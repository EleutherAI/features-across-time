import argparse
import os
import functools
import logging

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.utils.data import DataLoader

import pandas as pd
from tqdm import tqdm

from datasets import load_from_disk

from script_utils.load_model import get_auto_tokenizer, get_gpt_neo_with_bias
from script_utils.experiment import Experiment


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=None):
    model.train()
    ddp_loss = torch.zeros(2).to(rank)
    if sampler:
        sampler.set_epoch(epoch)

    for batch_idx, data in tqdm(enumerate(train_loader)):
        input = data['input_ids'][:, :-1]
        #     tokenizer = experiment.get_tokenizer("EleutherAI", "gpt-neox-20b")
        #     print(tokenizer.decode(input[0])) # Produces legible text
        target = data['input_ids'][:, 1:]
        input, target = input.to(rank), target.to(rank)
        optimizer.zero_grad()

        test_logits = model(input).logits
        test_loss = F.cross_entropy(test_logits.view(-1, test_logits.size(-1)), target.view(-1), reduction="none")
        print("mean", test_loss.mean())
        print("sum", test_loss.sum())

        loss = model(input, labels=target).loss
        print("regular loss", loss)
        
        if batch_idx % 100 == 0: print(loss)
        loss.backward()
        optimizer.step()
        break
        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(input.flatten())

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, ddp_loss[0] / ddp_loss[1]))

        
def test(model, rank, world_size, test_loader):
    model.eval()
    ddp_loss = torch.zeros(4).to(rank)
    with torch.no_grad():
        for data in test_loader:
            input = data['input_ids'][:, :-1]
            target = data['input_ids'][:, 1:]
            input, target = input.to(rank), target.to(rank)

            output = model(input, labels=target)
            pred = output.logits.argmax(dim=2, keepdim=True)
            ddp_loss[0] += output.loss.item()
            ddp_loss[1] += pred.eq(target.view_as(pred)).sum().item()
            ddp_loss[2] += len(input.view(-1))
            ddp_loss[3] += 1

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    if rank == 0:
        test_loss = ddp_loss[0] / ddp_loss[3]
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, int(ddp_loss[1]), int(ddp_loss[2]),
            100. * ddp_loss[1] / ddp_loss[2]))
        

def finetune(
    rank: int,
    world_size: int,
    experiment: Experiment,
    tmp_cache_path: str,
) -> pd.DataFrame:
    logging.getLogger('torch._dynamo').setLevel(logging.ERROR)

    setup(rank, world_size)

    # train_dataset = load_dataset("allenai/c4", "es", split='train')
    # train_dataset = load_dataset("NeelNanda/pile-10k", split='train')
    train_dataset = load_from_disk("/mnt/ssd-1/lucia/es_tokenized.hf")
    # test_dataset = load_dataset("allenai/c4", "es", split='validation')
    test_dataset = load_from_disk("/mnt/ssd-1/lucia/es_tokenized.hf")
    
    train_sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size)
    test_sampler = DistributedSampler(test_dataset, rank=rank, num_replicas=world_size)
    train_kwargs = {'batch_size': experiment.batch_size, 'sampler': train_sampler}
    test_kwargs = {'batch_size': experiment.test_batch_size, 'sampler': test_sampler}
    cuda_kwargs = {'num_workers': 2,
                    'pin_memory': True,
                    'shuffle': False}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_loader = DataLoader(train_dataset,**train_kwargs)
    test_loader = DataLoader(test_dataset, **test_kwargs)
    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=100
    )
    torch.cuda.set_device(rank)

    model = experiment.get_model(experiment.team, experiment.model_name, None, tmp_cache_path).float()
    for param in model.parameters():
        param.requires_grad = False
    model.embed_out.bias.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {trainable_params}")
    model = FSDP(model, use_orig_params=True, auto_wrap_policy=auto_wrap_policy).float()
    model = torch.compile(model)
    
    optimizer = optim.Adadelta(model.parameters(), lr=experiment.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=experiment.gamma)
    
    for epoch in range(1, experiment.epochs + 1):
        train(experiment, model, rank, world_size, train_loader, optimizer, epoch, sampler=train_sampler)
        test(model, rank, world_size, test_loader)
        scheduler.step()

    if experiment.save_model:
        dist.barrier()
        states = model.state_dict()
        if rank == 0:
            torch.save(states, f"{experiment.model_name}_finetune.pt")
    
    cleanup()


def main(tmp_cache_path: str):
    mp.set_start_method("spawn")

    experiments = [
        Experiment(
            num_samples=1024,
            batch_size=batch_size, 
            seq_len=2048, 
            team="EleutherAI", 
            model_name=model_name, 
            get_model=get_gpt_neo_with_bias, 
            get_tokenizer=get_auto_tokenizer,
            d_vocab=50_277,
            steps=[],
            ngram_orders=[1, 2],
            eod_index=get_auto_tokenizer("EleutherAI", model_name).eos_token_id,
            epochs=1,
            lr=1e-4,
            gamma=0.1,
            seed=1,
            test_batch_size=4,
            save_model=True,
        )
        for model_name, batch_size in [
            # ("pythia-14m", 4),
            # ("pythia-70m", 4),
            ("pythia-160m", 4),
            # ("pythia-410m", 4),
            # ("pythia-1b", 4),
            # ("pythia-1.4b", 4),
            # ("pythia-2.8b", 4),
            # ("pythia-6.9b", 1),
            # ("pythia-12b", 1),
        ]
    ]

    gpu_ids = None
    for experiment in experiments:
        torch.manual_seed(experiment.seed)
        if not gpu_ids:
            gpu_ids = list(range(torch.cuda.device_count()))

        print(f"Parallelising over {len(gpu_ids)} GPUs...")

        mp.spawn(finetune,
            args=(len(gpu_ids), experiment, tmp_cache_path),
            nprocs=len(gpu_ids),
            join=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--tmp_cache_path",
        default=".cache",
        help="Path to cache (repeatedly cleared to free disk space)",
    )
    args = parser.parse_args()
    main(args.tmp_cache_path)