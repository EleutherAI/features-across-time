import math
import os
import time
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Callable, Any
import random
from itertools import cycle

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from ml_dtypes import bfloat16 as np_bfloat16
from scripts.script_utils.divergences import (
    js_divergence_from_logits,
    kl_divergence_from_logits,
)


def set_seeds(seed=42):
    random.seed(seed)  # unused
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seeds()

class MemmapBFloat16Dataset(Dataset):
    def __init__(self, memmap_file, num_samples, seq_len, d_vocab):
        self.data = np.memmap(
            memmap_file,
            dtype=np_bfloat16,
            mode="r",
            shape=(num_samples * seq_len, d_vocab),
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class NgramTokensDataset(Dataset):
    def __init__(self, n, num_samples, seq_len):
        if n < 3:
            if num_samples == 4096:
                dataset = load_from_disk(f'/mnt/ssd-1/lucia/features-across-time/data/pile-deduped/{n}-gram-sequences-full-pile-4096.hf')
                print("Running 4096 samples for 1- to 2-grams")
            else:
                dataset = load_from_disk(f'/mnt/ssd-1/lucia/features-across-time/data/pile-deduped/{n}-gram-sequences-full-pile.hf')
            tokens = dataset['input_ids']
            self.data = torch.from_numpy(np.stack(tokens)).long()
        else:
            data_path = Path("/mnt/ssd-1/lucia/ngrams-across-time/data")
            if num_samples == 4096:
                tokens_path = data_path / f"{n}-gram-autoregressive-samples-4-shards-4096.npy"
                print("Running 4096 samples above 2-grams")
            else:
                tokens_path = data_path / f"{n}-gram-autoregressive-samples-4-shards.npy"
                print("Running 1024 samples")
            if not tokens_path.exists():
                raise ValueError(
                    f"Could not find tokens, please use Tokengrams to generate"
                )

            self.data = torch.from_numpy(
                np.memmap(
                    str(tokens_path), dtype=np.int32, mode="r+", shape=(num_samples, seq_len)
                ).copy()
            ).long()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_np(batch):
    return torch.from_numpy(np.stack(batch))


class MultiDatasetLoader:
    def __init__(self, datasets):
        self.datasets = datasets
        self.iterators = {name: iter(dataloader) for name, dataloader in datasets.items()}

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return {name: next(iterator) for name, iterator in self.iterators.items()}
        except StopIteration:
            self.iterators = {name: iter(dataloader) for name, dataloader in self.datasets.items()}
            raise StopIteration


def create_dataloaders(
        val_path: Path,
        ngram_path: Path,
        ngram_orders: list[int], 
        num_samples: int, 
        seq_len: int, 
        batch_size: int, 
        d_vocab: int, 
        div=True, 
        loss=True
    ) -> MultiDatasetLoader:
    dataloaders: dict[str, Any] = {}

    dataloaders['val'] = DataLoader(
        load_from_disk(str(val_path)).select(range(num_samples)), batch_size=batch_size  # type: ignore
    )
    
    if div:
        div_loaders = {
            f'ngram_logprobs_{n}': DataLoader(
                MemmapBFloat16Dataset(
                    str(ngram_path / f"{n}-gram-pile-logprobs-21_shards-bfloat16.npy"),
                    num_samples, seq_len, d_vocab),
                batch_size=batch_size * seq_len, # TODO check this
                collate_fn=collate_np,
            ) for n in ngram_orders
        }
        dataloaders.update(div_loaders)
    
    if loss:
        dataloaders.update({
            f'ngram_tokens_{n}': DataLoader(
                NgramTokensDataset(n, num_samples, seq_len),
                batch_size=batch_size,
            ) for n in ngram_orders
        })
    
    return MultiDatasetLoader(dataloaders)



@torch.inference_mode()
def worker(
    gpu_id: int,
    steps: list[int],
    model_name: str,
    batch_size: int,
    num_samples: int,
    d_vocab: int,
    seq_len: int,
    ngram_orders: list[int],
    ngram_path: Path,
    val_path: Path,
    div: bool,
    loss: bool,
) -> dict[str, list[float] | list[torch.Tensor]]:
    print(f"Calculating loss? {loss}. Calculating divs? {div}")
    torch.cuda.set_device(gpu_id)
    
    multi_loader = create_dataloaders(
        val_path,
        ngram_path,
        ngram_orders, 
        num_samples, 
        seq_len, 
        batch_size, 
        d_vocab, 
        div, 
        loss
    )
    cycled_loader = cycle(multi_loader)
    print("Loaded data...")

    # Do inference
    data = defaultdict(list)
    data["step"] = steps

    num_iters = math.ceil(num_samples / batch_size)
    for name, dataloader in multi_loader.datasets.items():
        assert len(dataloader) == num_iters, f"Dataloader '{name}' has {len(dataloader)} batches, expected {num_iters}"

    pbar = tqdm(total=len(steps) * num_iters, position=gpu_id)
    for step in steps:
        success = False
        for retry in range(3):
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    f"EleutherAI/{model_name}",
                    torch_dtype=torch.float16,
                    revision=f"step{step}",
                    cache_dir=".cache",
                    quantization_config=BitsAndBytesConfig(load_in_8bit=True) if "12b" in model_name else None,
                )
                if not "12b" in model_name:
                    model.cuda()
                success = True
                break
            except Exception as e:
                if retry < 2:
                    print(f"Attempt {retry + 1} failed, retrying in 2 seconds...", e)
                    time.sleep(2)
                else:
                    print("Failed to retrieve model at step", step, e)
                    for n in ngram_orders:
                        if loss:
                            data[f"mean_{n}_gram_loss"].append(np.nan)
                            data[f"mean_{n}_gram_accuracy"].append(np.nan)
                        if div:
                            data[f"mean_{n}_gram_kl_div"].append(np.nan)
                            data[f"mean_{n}_gram_js_div"].append(np.nan)
        if not success:
            continue

        running_means = defaultdict(float)
        # for n in ngram_orders:
        #     running_means[f'mean_{n}_per_token_loss'] = torch.zeros(seq_len - 1) # type: ignore

        for _ in range(num_iters):
            batch = next(cycled_loader)

            tokens = batch['val']['input_ids'].cuda()
            logits = model(tokens).logits[:, :, :d_vocab].flatten(0, 1)
            del tokens

            for n in ngram_orders:
                if f'ngram_logprobs_{n}' in batch:
                    ngram_logprobs = batch[f'ngram_logprobs_{n}'].cuda()
                    running_means[f"mean_{n}_gram_kl_div"] += (
                        kl_divergence_from_logits(ngram_logprobs, logits).mean().item()
                        / num_iters
                    )
                    running_means[f"mean_{n}_gram_js_div"] += (
                        js_divergence_from_logits(ngram_logprobs, logits).mean().item()
                        / num_iters
                    )
                    del ngram_logprobs

                if f'ngram_tokens_{n}' in batch:
                    ngram_tokens = batch[f'ngram_tokens_{n}'].cuda()
                    logits = model(ngram_tokens).logits[:, :, :d_vocab]
                    # per_token_loss = F.cross_entropy(
                    #     logits[:, :-1].flatten(0, 1),
                    #     ngram_tokens[:, 1:].flatten(0, 1),
                    #     reduction="none"
                    # ).reshape(batch_size, seq_len - 1).mean(0)
                    # running_means[f'mean_{n}_per_token_loss'] += per_token_loss.cpu() / num_iters

                    mean_loss: float = F.cross_entropy(
                        logits[:, :-1].flatten(0, 1),
                        ngram_tokens[:, 1:].flatten(0, 1),
                        reduction="mean",
                    ).item()
                    running_means[f"mean_{n}_gram_loss"] += mean_loss / num_iters
                    mean_accuracy = (logits.argmax(-1)[:, :-1] == ngram_tokens[:, 1:]).float().mean().item()
                    running_means[f"mean_{n}_gram_accuracy"] += (
                        mean_accuracy / num_iters
                    )

            pbar.update(1)
        print(running_means[f"mean_{n}_gram_kl_div"])
        print(running_means[f"mean_{n}_gram_js_div"])
        print(running_means[f"mean_{n}_gram_loss"])
        print(running_means[f"mean_{n}_gram_accuracy"])

        for key in running_means.keys():
            data[key].append(running_means[key])

    # Convert defaultdict to dict for DataFrame
    return {key: value for (key, value) in data.items()}


def get_missing_steps(
    df: pd.DataFrame,
    ngram_orders: list[int],
    steps: list[int],
    loss: bool,
    div: bool,
) -> set[int]:
    missing_steps = set()
    for ngram_order in ngram_orders:
        if loss:
            if f"mean_{ngram_order}_gram_loss" not in df.columns:
                missing_steps.update(steps)
            elif f"mean_{ngram_order}_gram_loss" in df:
                missing_steps.update(
                    df[df[f"mean_{ngram_order}_gram_loss"].isnull()]["step"].tolist()
                )
        if div:
            if f"mean_{ngram_order}_gram_kl_div" not in df.columns:
                missing_steps.update(steps)
            elif f"mean_{ngram_order}_gram_kl_div" in df:
                missing_steps.update(
                    df[df[f"mean_{ngram_order}_gram_kl_div"].isnull()]["step"].tolist()
                )

    return missing_steps


def main(
    ngram_path: Path,
    val_path: Path,
    output_path: Path,
    ngram_orders: list[int],
    steps: list[int],
    output_prefix: str,
    div: bool,
    loss: bool,
    overwrite: bool,
    num_samples: int,
):
    debug = False
    output_path.mkdir(exist_ok=True, parents=True)

    mp.set_start_method("spawn")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    
    seq_len = 2049

    for model_name, batch_size in (
        [
            # ("pythia-14m", 8),
            # ("pythia-70m", 4),
            # ("pythia-160m", 4),
            # ("pythia-410m", 4),
            # ("pythia-1b", 4),
            # ("pythia-1.4b", 4),
            # ("pythia-2.8b", 2),
            # ("pythia-6.9b", 1),
            ("pythia-12b", 1),
            # ("pythia-14m-warmup01", 8),
            # ("pythia-70m-warmup01", 4),
        ]
        # + [(f"pythia-14m-seed{i}", 8) for i in range(1, 10)]
        # + [(f"pythia-70m-seed{i}", 4) for i in range(1, 10)]
        # + [(f"pythia-160m-seed{i}", 4) for i in range(1, 10)]
        # + [(f"pythia-410m-seed{i}", 4) for i in range(1, 5)]
    ):
        print("Collecting data for", model_name)

        (output_path / "backup").mkdir(exist_ok=True)

        current_df_path = output_path / f"ngram_{model_name}_{num_samples}.csv"
        if not current_df_path.exists():
            pd.DataFrame({"step": steps}).to_csv(current_df_path)
        current_df = pd.read_csv(current_df_path)
        
        current_df.to_csv(
            output_path
            / "backup"
            / f"{output_prefix}_{model_name}_{num_samples}_{time.time()}.csv"
        )  
        current_df.set_index("step", inplace=True, drop=False)

        if not overwrite:
            missing_steps = get_missing_steps(
                current_df, ngram_orders, steps, loss, div
            )
            if not missing_steps:
                continue

            steps = [int(step) for step in list(missing_steps)]

        if debug:
            worker(
                0,
                steps,
                model_name,
                batch_size,
                num_samples,
                len(tokenizer),
                seq_len,
                ngram_orders,
                ngram_path,
                val_path,
                div,
                loss,
            )
        else:
            data = run_workers(
                worker,
                # roughly log spaced steps + final step
                steps,
                model_name=model_name,
                batch_size=batch_size,
                num_samples=num_samples,
                d_vocab=len(tokenizer.vocab),  # type: ignore
                seq_len=seq_len,
                ngram_orders=ngram_orders,
                ngram_path=ngram_path,
                val_path=val_path,
                div=div,
                loss=loss,
            )

        # Extract out per token loss tensors and make them into a separate dataframe with both
        # step and token index. don't worry about merging with other dfs etc.
        # if loss:
        #     result = []
        #     for d in data:
        #         # for step in step:
        #         for step_index, step in enumerate(d["step"]):
        #             step_dict = {"step": [step] * (seq_len - 1),"token": list(range(1, seq_len)),**{f"mean_{n}_per_token_loss": d[f"mean_{n}_per_token_loss"][step_index].tolist() for n in ngram_orders}}
        #             result.append(step_dict)

        #     in_context_df = pd.concat(pd.DataFrame(item) for item in result)
        #     # in_context_df.to_csv(
        #     #     pd.concat([pd.DataFrame(r) for r in result])
        #     in_context_df.to_csv(
        #         output_path / f"context_{model_name}_{num_samples}.csv", index=False
        #     )


        df = pd.concat([pd.DataFrame(d) for d in data])
        df.set_index("step", inplace=True, drop=False)
        for col in df.columns:
            current_df[col] = (
                df[col].combine_first(current_df[col]) if col in current_df else df[col]
            )   

        if "step" not in current_df.columns:
            current_df.reset_index(inplace=True)
        current_df.to_csv(
            output_path / f"{output_prefix}_{model_name}_{num_samples}.csv", index=False
        )


def run_workers(worker: Callable, steps: list[int], **kwargs) -> list[dict]:
    """Parallelise inference over model checkpoints."""
    max_steps_per_chunk = math.ceil(len(steps) / torch.cuda.device_count())
    step_indices = [
        steps[i : i + max_steps_per_chunk]
        for i in range(0, len(steps), max_steps_per_chunk)
    ]

    args = [
        (
            i,
            step_indices[i],
            *kwargs.values(),
        )
        for i in range(len(step_indices))
    ]
    print(
        f"Inference on steps {step_indices} for model {kwargs['model_name']}, \
            GPU count: {len(step_indices)}"
    )
    with mp.Pool(len(step_indices)) as pool:
        return pool.starmap(worker, args)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--n",
        default=[3],
        nargs="+",
        type=int,
        help="Which n-gram orders to collect data for",
    )
    parser.add_argument(
        "--loss",
        action="store_true",
    )
    parser.add_argument(
        "--div",
        action="store_true",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    parser.add_argument(
        "--output_prefix",
        "-p",
        default="ngram",
        help="CSV name prefix",
        type=str,
    )
    parser.add_argument(
        "--ngram_path",
        default="/mnt/ssd-1/lucia/ngrams-across-time/data",
        help="Path to n-gram data",
    )
    parser.add_argument(
        "--val_path",
        default="data/pile-deduped/val_tokenized.hf",
        help="Path to validation data",
    )
    parser.add_argument(
        "--output_path",
        default="output",
        help="Path to save output CSVs",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=4096
    )
    parser.add_argument(
        "--steps",
        default=[
            1,
            2,
            4,
            8,
            # 16,
            32,
            64,
            128,
            # 256,
            512,
            # 1000,
            2000,
            5000,
            # 8000,
            16_000,
            33_000,
            # 66_000,
            131_000,
            # 143_000,
        ],
        nargs="+",
        help="Which model checkpoints to collect data for",
    )
    args = parser.parse_args()
    main(
        Path(args.ngram_path),
        Path(args.val_path),
        Path(args.output_path),
        args.n,
        args.steps,
        args.output_prefix,
        args.div,
        args.loss,
        args.overwrite,
        args.num_samples
    )
