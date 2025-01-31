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
from tokengrams import MemmapIndex
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from scripts.script_utils.divergences import (
    js_divergence_from_logits,
    kl_divergence_from_logits,
)


def set_seeds(seed=42):
    import random; random.seed(seed) # unused
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class NgramDistDataset(Dataset):
    def __init__(self, memmap_file, num_samples, seq_len, d_vocab):
        self.data = np.memmap(
            memmap_file,
            dtype=np.float32,
            mode="r",
            shape=(num_samples, seq_len, d_vocab),
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx].copy()
        return torch.from_numpy(x)


class NgramModel:
    def __init__(
        self,
        path: Path,
        seq_len: int,
        d_vocab: int,
        tokengrams_filename: str,
        device: str,
    ):
        self.d_vocab = d_vocab
        self.seq_len = seq_len
        self.device = device

        with open(path / "bigrams.pkl", "rb") as f:
            bigram_counts = pickle.load(f).toarray().astype(np.float32)

        with np.errstate(divide='ignore',invalid='ignore'):
            self.bigram_probs = bigram_counts / bigram_counts.sum(axis=1)[:, None]
        self.bigram_probs = np.nan_to_num(self.bigram_probs)
        assert np.all(np.isfinite(self.bigram_probs))

        self.bigram_probs = (
            torch.from_numpy(self.bigram_probs)
            .to_sparse()
            .to_sparse_csr()
            .to(self.device)
        )

        self.unigram_probs = torch.from_numpy(bigram_counts).sum(dim=1).to(self.device)
        self.unigram_probs /= self.unigram_probs.sum()
        self.tokengrams = MemmapIndex(
            str(path / (tokengrams_filename + ".bin")),
            str(path / (tokengrams_filename + ".idx")),
        )

    def get_bigram_probs(self, prev: Tensor) -> Tensor:
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

    def get_unsmoothed_probs(self, tokens: Tensor, n: int) -> Tensor:
        if n == 1:
            return self.unigram_probs.expand((*tokens.shape, -1))
        if n == 2:
            return torch.stack([self.get_bigram_probs(row) for row in tokens])

        # split into sequences of up to n - 1 tokens that end with each token
        ngram_prefixes = []
        for row in tokens:
            ngram_prefixes.extend(
                [row[max(0, i - (n - 1)) : i].tolist() for i in range(len(row))]
            )
        counts = torch.tensor(
            self.tokengrams.batch_count_next(ngram_prefixes, self.d_vocab),
        ).reshape(-1, self.seq_len, self.d_vocab)
        return counts / (counts.sum(dim=1, keepdim=True))

    def get_smoothed_probs(self, tokens, n: int) -> Tensor:
        ngram_prefixes = []
        # split rows into sequences of up to n - 1 tokens that end with each token
        for row in tokens:
            ngram_prefixes.extend(
                [row[max(0, i - (n - 1)) : i].tolist() for i in range(len(row))]
            )

        return torch.tensor(
            self.tokengrams.batch_smoothed_probs(ngram_prefixes, self.d_vocab),
        ).reshape(-1, self.seq_len, self.d_vocab)


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
    tokengrams_filename: str,
) -> pd.DataFrame:
    torch.cuda.set_device(gpu_id)

    # Load input data
    val_ds = DataLoader(load_from_disk(str(val_path)), batch_size=batch_size)
    ngram_data = {}
    for n in ngram_orders:
        ds = load_from_disk(str(ngram_path / f"{n}-gram-sequences-unsmoothed.hf"))
        ds.set_format("torch", columns=["input_ids"])
        ngram_data[n] = DataLoader(ds, batch_size=batch_size)

    smooth_ngram_data = {}
    for n in ngram_orders:
        ds = load_from_disk(str(ngram_path / f"smoothed-{n}-gram-sequences.hf"))
        ds.set_format("torch", columns=["input_ids"])
        smooth_ngram_data[n] = DataLoader(ds, batch_size=batch_size)

    ngram_model = NgramModel(
        ngram_path, seq_len, d_vocab, tokengrams_filename, "cuda"
    )

    val_ngram_dists = {
        n: np.memmap(
            str(ngram_path / f"{n}-gram-pile-dists-32bit.npy"),
            dtype=np.float32,
            mode='r+',
            shape=(num_samples * seq_len, d_vocab)
        )
        for n in ngram_orders if n > 2
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
                    quantization_config=BitsAndBytesConfig(load_in_8bit=True) if "12b" in model_name else None
                )
                if not "12b" in model_name:
                    model = model.cuda()
                success = True
                break
            except Exception as e:
                if retry < 2:
                    print(f"Attempt {retry + 1} failed, retrying in 2 seconds...", e)
                    time.sleep(2)
                else:
                    print("Failed to retrieve model at step", step, e)
                    for n in ngram_orders:
                            data[f"mean_{n}_gram_loss"].append(np.nan)
                            data[f"mean_smooth_{n}_gram_loss"].append(np.nan)
                            data[f"mean_{n}_gram_kl_div"].append(np.nan)
                            data[f"mean_{n}_gram_js_div"].append(np.nan)
                            data[f"mean_smooth_{n}_gram_kl_div"].append(np.nan)
                            data[f"mean_smooth_{n}_gram_js_div"].append(np.nan)

        if not success:
            continue

        val = iter(val_ds)
        ngram_iters = {n: iter(ds) for n, ds in ngram_data.items()}
        smooth_ngram_iters = {n: iter(ds) for n, ds in smooth_ngram_data.items()}
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

                tokens = next(smooth_ngram_iters[n])["input_ids"].cuda()
                logits = model(tokens).logits[:, :, :d_vocab]
                mean_loss = F.cross_entropy(
                    logits[:, :-1].flatten(0, 1),
                    tokens[:, 1:].flatten(0, 1),
                    reduction="mean",
                ).item()
                running_means[f"mean_{n}_gram_smoothed_loss"] += (
                    mean_loss / num_iters
                )

            for n in ngram_orders:
                tokens = next(val)["input_ids"].cuda()
                logits = model(tokens).logits[:, :, :d_vocab].flatten(0, 1)
                ngram_logprobs = (
                    ngram_model.get_unsmoothed_probs(tokens, n)
                    .cuda()
                    .log()
                    .flatten(0, 1)
                )
                running_means[f"mean_{n}_gram_kl_div"] += (
                    kl_divergence_from_logits(ngram_logprobs, logits).mean().item()
                    / num_iters
                )
                running_means[f"mean_{n}_gram_js_div"] += (
                    js_divergence_from_logits(ngram_logprobs, logits).mean().item()
                    / num_iters
                )
                del ngram_logprobs

                ngram_logprobs = (
                    ngram_model.get_smoothed_probs(tokens, n)
                    .cuda()
                    .log()
                    .flatten(0, 1)
                )
                running_means[f"mean_smooth_{n}_gram_kl_div"] += (
                    kl_divergence_from_logits(ngram_logprobs, logits).mean().item()
                    / num_iters
                )
                running_means[f"mean_smooth_{n}_gram_js_div"] += (
                    js_divergence_from_logits(ngram_logprobs, logits).mean().item()
                    / num_iters
                )
            pbar.update(1)
        print(running_means[f"mean_smooth_{ngram_orders[-1]}_gram_kl_div"])

        for key in running_means.keys():
            data[key].append(running_means[key])

    return {key: value for (key, value) in data.items()}


def main(
    ngram_path: Path,
    val_path: Path,
    output_path: Path,
    ngram_orders: list[int],
    steps: list[int],
    tokengrams_filename: str,
):
    output_path.mkdir(exist_ok=True, parents=True)

    mp.set_start_method("spawn")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    num_samples = 4096
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
            ("pythia-12b", 2),
            ("pythia-14m-warmup01", 8),
            ("pythia-70m-warmup01", 4),
        ]
        + [(f"pythia-14m-seed{i}", 2) for i in range(1, 10)]
        + [(f"pythia-70m-seed{i}", 4) for i in range(1, 10)]
        + [(f"pythia-160m-seed{i}", 4) for i in range(1, 10)]
        + [(f"pythia-410m-seed{i}", 4) for i in range(1, 5)]
    ):
        print("Collecting data for", model_name)

        current_df_path = output_path / f"ngram_{model_name}_{num_samples}.csv"
        if not current_df_path.exists():
            pd.DataFrame({"step": steps}).to_csv(current_df_path)
        current_df = pd.read_csv(current_df_path)
        (output_path / 'backup').mkdir(exist_ok=True, parents=True)
        current_df.to_csv(
            output_path
            / "backup"
            / f"ngram_{model_name}_{num_samples}_{time.time()}.csv"
        )
        current_df.set_index("step", inplace=True, drop=False)

        data = run_workers(
            worker,
            steps, # roughly log spaced steps + final step
            model_name=model_name,
            batch_size=batch_size,
            num_samples=num_samples,
            d_vocab=len(tokenizer.vocab),
            seq_len=seq_len,
            ngram_orders=ngram_orders,
            ngram_path=ngram_path,
            val_path=val_path,
            tokengrams_filename=tokengrams_filename,
        )
        df = pd.concat([pd.DataFrame(d) for d in data])
        df.set_index("step", inplace=True)
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
        "--n",
        default=[3],
        nargs="+",
        type=int,
        help="N-gram orders to collect data for",
    )
    parser.add_argument(
        "--ngram_path",
        default="data/pile-deduped",
        help="Path to directory containing pickled sparse scipy array of \
            bigram counts and HF n-gram sequence datasets",
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
            4000,
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
        "document-10G",
    )
