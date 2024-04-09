import argparse
import math
import os
import shutil
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import tqdm.auto as tqdm
from scipy import stats
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_from_disk
from script_utils.ngram_model import NgramModel
from script_utils.divergences import kl_divergence, js_divergence, one_hot_js_divergence
from script_utils.load_model import get_auto_tokenizer, get_black_mamba, get_hails_mamba, get_zyphra_mamba, get_auto_model
from script_utils.experiment import Experiment

from dataclasses import dataclass
from typing import Callable, Any
import math
import os
import argparse
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
import pandas as pd

from script_utils.experiment import Experiment


import copy

import torch
import torch.nn.functional as F
import torch.backends.cuda as cuda
from torch.utils.data import DataLoader, IterableDataset

import wandb
from tqdm import tqdm
import bitsandbytes as bnb

from datasets import load_dataset
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention




def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def _attn_wrapper(self, query, key, value, attention_mask=None, head_mask=None):
    assert attention_mask is None and head_mask is None, "Not implemented"
    with cuda.sdp_kernel(enable_math=False):
        out = F.scaled_dot_product_attention(
            query.half(),
            key.half(),
            value.half(),
            is_causal=True,
        ).float()
    return out, None

# patch attention to save a lot of memory
GPTNeoXAttention._attn = _attn_wrapper


class DatasetWrapper(IterableDataset):
    def __init__(self, max_tokens):
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
        self.max_tokens = max_tokens

    def __iter__(self):
        buffer = []
        for sample in load_dataset(
            "EleutherAI/the_pile_deduplicated",
            # "togethercomputer/RedPajama-Data-1T",
            name="all",
            split="train",
            streaming=True,
        ).shuffle(buffer_size=10_000):
            buffer += self.tokenizer(sample["text"])["input_ids"]
            buffer += [self.tokenizer.eos_token_id]
            while len(buffer) > self.max_tokens:
                yield torch.tensor(buffer[: self.max_tokens])
                buffer = buffer[self.max_tokens :]


class Trainer:
    def __init__(self):
        self.max_tokens = 2**13
        self.grad = 64
        self.step = 0

        self.dataset = DatasetWrapper(self.max_tokens)
        self.tokenizer = self.dataset.tokenizer
        self.loader = DataLoader(
            self.dataset,
            batch_size=1,
            num_workers=8,
        )
        self.scaler = torch.cuda.amp.GradScaler()
        self.model = model = GPTNeoXForCausalLM.from_pretrained(
            "EleutherAI/pythia-1.4b-deduped",
        ).cuda()

        self.show_params()

        self.opt = bnb.optim.Lion(
            params=model.parameters(),
            lr=1e-5,
            weight_decay=1e-1,
            betas=(0.9, 0.95),
            optim_bits=8,
            # fused=True,
        )
        self.model = torch.compile(model)

    def show_params(self):
        model = self.model
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        emb_params = list(model.gpt_neox.embed_in.parameters())
        emb_params += list(model.embed_out.parameters())
        emb_params = sum(p.numel() for p in emb_params if p.requires_grad)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Params:", params - emb_params)
        print("Params (incl. embeddings):", params)
        print("Trainable params:", trainable_params)

    def train_step(self, batch):
        batch = batch.cuda()
        x, y = batch[:, :-1], batch[:, 1:]
        with torch.autocast(device_type="cuda", enabled=True):
            z = self.model(x).logits
            y = y.reshape(-1)
            z = z.view(-1, z.shape[-1])
            loss = F.cross_entropy(z, y)
        self.scaler.scale(loss / self.grad).backward()
        return loss

    def train(self):
        # wandb.init(
        #     project="pythia",
        #     entity="lovis",
        # )

        prog = tqdm(self.loader)
        self.opt.zero_grad()

        for i, batch in enumerate(prog):
            self.step = i + 1

            loss = self.train_step(batch)
            prog.set_description(f"loss: {loss.item():.3f}")
            # wandb.log(
            #     {
            #         "loss": loss.item(),
            #     },
            #     step=i,
            # )

            if (i + 1) % self.grad == 0:
                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad()

            if i % 1000 == 0:
                temp_model = copy.deepcopy(self.model).half()
                temp_model.save_pretrained(
                    "<your-hf-repo-id>",
                    push_to_hub=True,
                    max_shard_size="500MB",
                )
                del temp_model
                torch.cuda.empty_cache()


# if __name__ == "__main__":
#     trainer = Trainer()
#     trainer.train()


def train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=None):
    model.train()
    ddp_loss = torch.zeros(2).to(rank)
    if sampler:
        sampler.set_epoch(epoch)

    # pbar = tqdm.tqdm(total=len(data), position=args.gpu_id)
    for batch_idx, data in tqdm(enumerate(train_loader)):
        input = data['input_ids'][:, :-1]
        target = data['input_ids'][:, 1:]
        input, target = input.to(rank), target.to(rank)
        optimizer.zero_grad()
        loss = model(input, labels=target).loss
        loss.backward()
        optimizer.step()
        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(input)

    

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, ddp_loss[0] / ddp_loss[1]))

        
def test(model, rank, world_size, test_loader):
    model.eval()
    ddp_loss = torch.zeros(3).to(rank)
    with torch.no_grad():
        for data in test_loader:
            input = data['input_ids'][:, :-1]
            target = data['input_ids'][:, 1:]
            input, target = input.to(rank), target.to(rank)

            logits = model(input).logits
            ddp_loss[0] += F.nll_loss(logits.flatten(0, 1), target.flatten(), reduction='sum').item()
            pred = logits.argmax(dim=2, keepdim=True)
            ddp_loss[1] += pred.eq(target.view_as(pred)).sum().item()
            ddp_loss[2] += len(data)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    if rank == 0:
        test_loss = ddp_loss[0] / ddp_loss[2]
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, int(ddp_loss[1]), int(ddp_loss[2]),
            100. * ddp_loss[1] / ddp_loss[2]))
        

def finetune(
    rank: int,
    world_size: int,
    dfs: list,
    experiment: Experiment,
    model_path: str,
    pile_path: str,
    tmp_cache_path: str,
) -> pd.DataFrame:
    setup(rank, world_size)

    # train_dataset = load_dataset("allenai/c4", "es", split='train')
    # train_dataset = load_dataset("NeelNanda/pile-10k", split='train')
    train_dataset = load_from_disk("/home/lucia/features-across-time/scripts/es_tokenized.hf")
    # test_dataset = load_dataset("allenai/c4", "es", split='validation')
    test_dataset = load_from_disk("/home/lucia/features-across-time/scripts/es_tokenized.hf")
    
    train_sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size) # shuffle=True
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
    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=100
    )
    torch.cuda.set_device(rank)
    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    model = experiment.get_model(experiment.team, experiment.model_name, None, tmp_cache_path).float()
    model = FSDP(model).float()

    # model = torch.compile(model)
    
    optimizer = optim.Adadelta(model.parameters(), lr=experiment.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=experiment.gamma)
    init_start_event.record()
    
    for epoch in range(1, experiment.epochs + 1):
        train(experiment, model, rank, world_size, train_loader, optimizer, epoch, sampler=train_sampler)
        print("out")
        test(model, rank, world_size, test_loader)
        scheduler.step()

    init_end_event.record()

    if rank == 0:
        print(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
        print(f"{model}")

    if experiment.save_model:
        # use a barrier to make sure training is done on all ranks
        dist.barrier()
        states = model.state_dict()
        if rank == 0:
            torch.save(states, f"{experiment.model_name}_finetune.pt")
    
    # dfs.append(df)
    cleanup()


def main(ngram_path: str, pile_path: str, tmp_cache_path: str):
    experiments = [
        Experiment(
            num_samples=1024,
            batch_size=batch_size, 
            seq_len=2048, 
            team="EleutherAI", 
            model_name=model_name, 
            get_model=get_auto_model, 
            get_tokenizer=get_auto_tokenizer,
            d_vocab=50_277,
            steps=[],
            ngram_orders=[1, 2],
            eod_index=get_auto_tokenizer("EleutherAI", model_name).eos_token_id,
            epochs=1,
            lr=1e-4,
            gamma=1,
            seed=1,
            test_batch_size=4,
            save_model=True,
        )
        for model_name, batch_size in [
            ("pythia-14m", 4),
            # ("pythia-70m", 4),
            # ("pythia-160m", 4),
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
        # manager = mp.Manager()
        # dfs = manager.list()  # Shared list to collect DataFrames

        finetune(0, 1, [], experiment, ngram_path, pile_path, tmp_cache_path)
        # mp.spawn(finetune,
        #         args=(len(gpu_ids), dfs, experiment, ngram_path, pile_path, tmp_cache_path),
        #         nprocs=len(gpu_ids),
        #         join=True)

        # df = pd.concat(dfs)
        # df.to_csv(
        #     Path.cwd()
        #     / "output"
        #     / f"finetune_{experiment.model_name}_{experiment.num_samples}_{experiment.ngram_orders}.csv",
        #     index=False,
        # )


if __name__ == "__main__":
    # mp.set_start_method("spawn")

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--ngram_path",
        default="/mnt/ssd-1/lucia/pythia-deduped-bigrams.pkl",
        help="Path to pickled sparse scipy array of bigram counts over the Pile",
    )
    parser.add_argument(
        "--pile_path",
        default="/mnt/ssd-1/lucia/val_tokenized.hf",
        help="Path to Pile validation data",
    )
    parser.add_argument(
        "--tmp_cache_path",
        default=".cache",
        help="Path to cache (repeatedly cleared to free disk space)",
    )
    args = parser.parse_args()
    main(args.ngram_path, args.pile_path, args.tmp_cache_path)