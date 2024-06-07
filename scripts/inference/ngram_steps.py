import math
import pickle
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
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.script_utils.divergences import (
    js_divergence_log_space,
    kl_divergence_log_space,
)


class NgramModel:
    def __init__(self, path: str, seq_len: int, d_vocab: int, device):
        self.d_vocab = d_vocab
        self.seq_len = seq_len
        self.device = device

        with open(path, "rb") as f:
            bigram_counts = pickle.load(f).toarray().astype(np.float32)

        self.bigram_probs = bigram_counts / (
            bigram_counts.sum(axis=1)[:, None] + np.finfo(np.float32).eps
        )
        self.bigram_probs = (
            torch.tensor(self.bigram_probs).to_sparse().to_sparse_csr().to(self.device)
        )

        self.unigram_probs = torch.tensor(bigram_counts).sum(dim=1).to(self.device)
        self.unigram_probs /= self.unigram_probs.sum()

    def get_bigram_prob(self, prev: Tensor) -> Tensor:
        starts = self.bigram_probs.crow_indices()[prev]
        ends = self.bigram_probs.crow_indices()[prev + 1]

        # 0 padding to batch rows with variable numbers of non-zero elements
        bigram_dists = torch.zeros(
            (len(prev), self.d_vocab), dtype=torch.float32, device=self.device
        )
        for i in range(len(prev)):
            filled_col_indices = self.bigram_probs.col_indices()[starts[i] : ends[i]]
            filled_col_values = self.bigram_probs.values()[starts[i] : ends[i]]
            bigram_dists[i][filled_col_indices] = filled_col_values
        return bigram_dists

    def get_ngram_prob(self, tokens: Tensor, n: int) -> Tensor:
        if n == 1:
            return (
                self.unigram_probs.expand((*tokens.shape, -1))
                + torch.finfo(torch.float32).eps
            )
        if n == 2:
            if len(tokens.shape) == 1:
                return self.get_bigram_prob(tokens) + torch.finfo(torch.float32).eps
            else:
                return (
                    torch.stack([self.get_bigram_prob(row) for row in tokens])
                    + torch.finfo(torch.float32).eps
                )
        if n >= 3:
            raise NotImplementedError


@torch.inference_mode()
def worker(
    gpu_id: int,
    steps: list[int],
    model_name: str,
    batch_size: str,
    num_samples: int,
    d_vocab: int,
    seq_len: int,
    ngram_orders: list[int],
    ngram_path: Path,
    pile_path: Path,
) -> pd.DataFrame:
    torch.cuda.set_device(gpu_id)

    # Load input data
    pile_ds = DataLoader(load_from_disk(str(pile_path)), batch_size=batch_size)
    ngram_data = {}
    for n in ngram_orders:
        ds = load_from_disk(str(ngram_path / f"{n}-gram-sequences.hf"))
        ds.set_format("torch", columns=["input_ids"])
        ngram_data[n] = DataLoader(ds, batch_size=batch_size)

    ngram_model = NgramModel(
        str(ngram_path / "bigrams.pkl"),
        seq_len,
        d_vocab,
        "cuda",
    )
    print("Loaded data...")

    # Collect inference data
    data = defaultdict(list)
    data["step"] = steps

    num_iters = math.ceil(num_samples / batch_size)
    pbar = tqdm(total=len(steps) * num_iters, position=gpu_id)
    for step in steps:
        for retry in range(3):
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    f"EleutherAI/{model_name}",
                    torch_dtype="auto",
                    revision=f"step{step}",
                ).cuda()
                break
            except Exception as e:
                if retry < 2:
                    print(f"Attempt {retry + 1} failed, retrying in 2 seconds...", e)
                    time.sleep(2)
                else:
                    print("Failed to retrieve model at step", step, e)
                    for n in ngram_orders:
                        data[f"mean_{n}_gram_loss"].append(np.nan)
                        data[f"mean_{n}_gram_kl_div"].append(np.nan)
                        data[f"mean_{n}_gram_js_div"].append(np.nan)

        pile = iter(pile_ds)
        ngram_iters = {n: iter(ds) for n, ds in ngram_data.items()}
        running_means = defaultdict(float)
        for _ in range(num_iters):
            for n in ngram_orders:
                tokens = next(ngram_iters[n])["input_ids"].cuda()
                logits = model(tokens).logits[:, :, :d_vocab]
                mean_loss = F.cross_entropy(
                    logits[:, :-1].flatten(0, 1),
                    tokens[:, 1:].flatten(0, 1),
                    reduction="mean",
                ).item()
                running_means[f"mean_{n}_gram_loss"] += mean_loss / num_iters

            tokens = next(pile)["input_ids"].cuda()
            logits = model(tokens).logits[:, :, :d_vocab].flatten(0, 1)
            for n in ngram_orders:
                ngram_dists = (
                    ngram_model.get_ngram_prob(tokens, n).cuda().log().flatten(0, 1)
                )
                running_means[f"mean_{n}_gram_kl_div"] += (
                    kl_divergence_log_space(ngram_dists, logits).mean().item()
                    / num_iters
                )
                running_means[f"mean_{n}_gram_js_div"] += (
                    js_divergence_log_space(ngram_dists, logits).mean().item()
                    / num_iters
                )

            pbar.update(1)

        for key in running_means.keys():
            data[key].append(running_means[key])

    # Convert defaultdict to dict for DataFrame
    return {key: value for (key, value) in data.items()}


def main(ngram_path: Path, pile_path: Path, output_path: Path):
    mp.set_start_method("spawn")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    num_samples = 1024

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
        + [(f"pythia-14m-seed{i}", 8) for i in range(3, 10)]
        + [(f"pythia-70m-seed{i}", 4) for i in range(1, 10)]
        + [(f"pythia-160m-seed{i}", 4) for i in range(8, 10)]
        + [(f"pythia-410m-seed{i}", 4) for i in range(1, 5)]
    ):
        print("Collecting data for", model_name)
        data = run_workers(
            worker,
            # roughly log spaced steps + final step
            [
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
            model_name=model_name,
            batch_size=batch_size,
            num_samples=num_samples,
            d_vocab=len(tokenizer.vocab),
            seq_len=2049,
            ngram_orders=[1, 2],
            ngram_path=ngram_path,
            pile_path=pile_path,
        )
        df = pd.concat([pd.DataFrame(d) for d in data])
        df.to_csv(
            output_path / f"ngram_{model_name}_{num_samples}.csv",
            index=False,
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
    print(f"Inference on steps {step_indices}, GPU count: {len(step_indices)}")
    with mp.Pool(len(step_indices)) as pool:
        return pool.starmap(worker, args)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--ngram_path",
        default="data/pile-deduped",
        help="Path to directory containing pickled sparse scipy array of \
            bigram counts and HF n-gram sequence datasets",
    )
    parser.add_argument(
        "--pile_path",
        default="data/pile-deduped/val_tokenized.hf",
        help="Path to Pile validation data",
    )
    parser.add_argument(
        "--output_path",
        default="output",
        help="Path to save output CSVs",
    )
    args = parser.parse_args()
    main(Path(args.ngram_path), Path(args.pile_path), Path(args.output_path))
