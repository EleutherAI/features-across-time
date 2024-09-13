import math
import os
import time
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from ml_dtypes import bfloat16 as np_bfloat16
from scripts.script_utils.divergences import (
    js_divergence_from_logits,
    kl_divergence_from_logits,
)


def set_seeds(seed=42):
    import random

    random.seed(seed)  # unused
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seeds()

class NgramDistDatasetBFloat16(Dataset):
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
    def __init__(self, n, num_samples, seq_len, d_vocab):
        num_tokens = num_samples * seq_len

        data_path = Path("/mnt/ssd-1/lucia/ngrams-across-time/data")
        tokens_path = data_path / f"{n}-gram-autoregressive-samples-4-shards.npy"
        counts_path = data_path / f"{n}-gram-autoregressive-counts-4-shards.npy"

        if tokens_path.exists():
            self.data = torch.from_numpy(
                np.memmap(
                    str(tokens_path), dtype=np.int32, mode="r+", shape=(num_tokens)
                ).copy()
            ).long()
            tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
            samples = "".join(tokenizer.batch_decode(self.data[:seq_len]))
            print("for ", n, samples)
        elif counts_path.exists():
            tokens_data = np.memmap(
                str(tokens_path), dtype=np.int32, mode="w+", shape=(num_tokens)
            )

            ngram_counts = np.memmap(
                str(counts_path), dtype=np.int64, mode="r", shape=(num_tokens, d_vocab)
            )
            unigram_counts = torch.from_numpy(ngram_counts[0].copy()).float()
            for i in tqdm(
                range(num_samples), desc=f"Sampling {n}-gram tokens from counts"
            ):  
                seq_slice = slice(i * seq_len, (i + 1) * seq_len)
                counts_torch = torch.from_numpy(ngram_counts[seq_slice].copy()).float()
                if any(counts_torch.sum(dim=-1) <= 0):
                    # Fall back to unigram counts if n-gram counts are all zero, equal to the first row of counts
                    counts_torch[counts_torch.sum(dim=-1) <= 0] = unigram_counts

                tokens_data[seq_slice] = (
                    torch.multinomial(counts_torch, 1)
                    .squeeze()
                    .to(torch.int32)
                    .numpy()
                )
                self.data = torch.from_numpy(tokens_data.copy()).long()
        # elif logprobs_path.exists():
        #     tokens_data = np.memmap(
        #         str(tokens_path), dtype=np.int32, mode="w+", shape=(num_tokens)
        #     )
        #     ngram_logprobs = np.memmap(
        #         str(logprobs_path),
        #         dtype=np_bfloat16,
        #         mode="r",
        #         shape=(num_tokens, d_vocab),
        #     )
        #     for i in tqdm(
        #         range(num_samples), desc=f"Sampling {n}-gram tokens from logprobs"
        #     ):
        #         seq_slice = slice(i * seq_len, (i + 1) * seq_len)
        #         ngram_probs = np.exp(ngram_logprobs[seq_slice])
        #         tokens_data[seq_slice] = (
        #             torch.multinomial(torch.from_numpy(ngram_probs), 1)
        #             .squeeze()
        #             .to(torch.int32)
        #             .numpy()
        #         )
                # self.data = torch.from_numpy(tokens_data.copy()).long()
        else:
            raise ValueError(
                f"Could not find tokens, probs, or logprobs for {n}-grams.\
                please use Tokengrams to generate"
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_np(batch):
    return torch.from_numpy(np.stack(batch))


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
) -> dict[str, list[float]]:
    print(f"Calculating loss? {loss}. Calculating divs? {div}")
    torch.cuda.set_device(gpu_id)
    val_ngram_dists, val_ngram_tokens = {}, {}

    # Load input data
    val_ds = DataLoader(
        load_from_disk(str(val_path)), batch_size=batch_size  # type: ignore
    )
    if div:
        val_ngram_dists = {
            n: DataLoader(
                NgramDistDatasetBFloat16(
                    str(
                        ngram_path / f"{n}-gram-pile-logprobs-21_shards-bfloat16.npy"
                    ),
                    num_samples,
                    seq_len,
                    d_vocab,
                ),
                collate_fn=collate_np,
                batch_size=batch_size * seq_len,
            )
            for n in ngram_orders
        }
    if loss:
        print("Loading or generating ngram tokens...")
        val_ngram_tokens = {
            n: DataLoader(
                NgramTokensDataset(n, num_samples, seq_len, d_vocab),
                batch_size=batch_size * seq_len,
            )
            for n in ngram_orders
        }
    print("Loaded data...")

    # Collect inference data
    data = defaultdict(list)
    data["step"] = steps

    num_iters = math.ceil(num_samples / batch_size)
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
                    # quantization_config=BitsAndBytesConfig(load_in_4bit=True)
                ).cuda()
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
                        if div:
                            data[f"mean_{n}_gram_kl_div"].append(np.nan)
                            data[f"mean_{n}_gram_js_div"].append(np.nan)
        if not success:
            continue

        val = iter(val_ds)
        ngram_dist_iters = {}
        ngram_token_iters = {}
        if div:
            # Use a cleaner dataloader setup
            ngram_dist_iters = {n: iter(val_ngram_dists[n]) for n in ngram_orders}
        if loss:
            ngram_token_iters = {n: iter(val_ngram_tokens[n]) for n in ngram_orders}

        running_means = defaultdict(float)
        for _ in range(num_iters):
            tokens = next(val)["input_ids"].cuda()
            logits = model(tokens).logits[:, :, :d_vocab].flatten(0, 1)
            del tokens

            for n in ngram_orders:
                if div:
                    ngram_logprobs = next(ngram_dist_iters[n]).cuda()
                    running_means[f"mean_{n}_gram_kl_div"] += (
                        kl_divergence_from_logits(ngram_logprobs, logits).mean().item()
                        / num_iters
                    )
                    running_means[f"mean_{n}_gram_js_div"] += (
                        js_divergence_from_logits(ngram_logprobs, logits).mean().item()
                        / num_iters
                    )
                    del ngram_logprobs

                if loss:
                    # Loss on maximum entropy n-gram tokens
                    ngram_tokens = (
                        next(ngram_token_iters[n]).cuda().reshape(batch_size, seq_len)
                    )
                    print(ngram_tokens[0, :10])
                    logits = model(ngram_tokens).logits[:, :, :d_vocab]
                    mean_loss: float = F.cross_entropy(
                        logits[:-1].flatten(0, 1),
                        ngram_tokens[1:].flatten(0, 1),
                        reduction="mean",
                    ).item()
                    running_means[f"mean_{n}_gram_loss"] += mean_loss / num_iters

            pbar.update(1)

        print(running_means[f"mean_{n}_gram_kl_div"])
        print(running_means[f"mean_{n}_gram_js_div"])
        print(running_means[f"mean_{n}_gram_loss"])

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
):
    debug = False
    output_path.mkdir(exist_ok=True, parents=True)

    mp.set_start_method("spawn")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    num_samples = 1024
    seq_len = 2049

    for model_name, batch_size in (
        [
            ("pythia-14m", 8),
            ("pythia-70m", 4),
            ("pythia-160m", 4),
            ("pythia-410m", 4),
            ("pythia-1b", 4),
            ("pythia-1.4b", 4),
            ("pythia-2.8b", 2),
            ("pythia-6.9b", 1),
            ("pythia-12b", 1),
            ("pythia-14m-warmup01", 8),
            ("pythia-70m-warmup01", 4),
        ]
        + [(f"pythia-14m-seed{i}", 8) for i in range(1, 10)]
        + [(f"pythia-70m-seed{i}", 4) for i in range(1, 10)]
        + [(f"pythia-160m-seed{i}", 4) for i in range(1, 10)]
        + [(f"pythia-410m-seed{i}", 4) for i in range(1, 5)]
    ):
        print("Collecting data for", model_name)

        current_df_path = output_path / f"ngram_{model_name}_{num_samples}.csv"
        if not current_df_path.exists():
            pd.DataFrame({"step": steps}).to_csv(current_df_path)
        current_df = pd.read_csv(current_df_path)
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
        df = pd.concat([pd.DataFrame(d) for d in data])

        os.makedirs(output_path / "backup", exist_ok=True)
        current_df.to_csv(
            output_path
            / "backup"
            / f"{output_prefix}_{model_name}_{num_samples}_{time.time()}.csv"
        )
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
        "--steps",
        default=[
            1,
            2,
            4,
            8,
            16,
            32,
            64,
            128,
            256,
            512,
            1000,
            2000,
            5000,
            8000,
            16_000,
            33_000,
            66_000,
            131_000,
            143_000,
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
    )
