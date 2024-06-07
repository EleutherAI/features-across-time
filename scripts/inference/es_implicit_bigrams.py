import argparse
import math
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import tqdm.auto as tqdm
from datasets import load_from_disk
from scipy import stats
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from scripts.script_utils.divergences import kl_divergence_log_space
from scripts.script_utils.experiment import (
    Experiment,
    run_checkpoint_experiment_workers,
)
from scripts.script_utils.load_model import get_es_finetune
from scripts.script_utils.ngram_model import NgramModel


def get_mean_divergences(
    tokens: torch.Tensor,
    logits: torch.Tensor,
    ngram_model: NgramModel,
    batch: int,
    d_vocab: int,
    ngram_orders: list[int],
    device: str,
) -> np.ndarray:
    assert ngram_orders == [
        1,
        2,
    ], "TODO add support for higher order n-gram KL divergences"

    divergences = []
    labels = []
    logits = logits[:, :, :d_vocab].flatten(0, 1)

    for n in ngram_orders:
        ngram_dist = (ngram_model.get_ngram_prob(tokens, n).to(device).log()).flatten(
            0, 1
        )
        divergences.append(kl_divergence_log_space(ngram_dist, logits).mean())
        labels.append(f"{n}-gram_logit_kl_div")

    return torch.stack(divergences), labels


def get_confidence_intervals(
    mean: float, num_items: int, confidence=0.95
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    sem = np.sqrt(mean / num_items)
    return stats.norm.interval(confidence, loc=mean, scale=sem)


@torch.inference_mode()
def finetuned_stats_worker(
    gpu_id: int,
    experiment: Experiment,
    bigrams_path: str,
    data_path: str,
    steps: list[int],
) -> pd.DataFrame:
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    with open(bigrams_path, "rb") as f:
        bigram_counts = pickle.load(f).toarray().astype(np.float32)
        dense_bigram_probs = torch.tensor(
            bigram_counts / (bigram_counts.sum(axis=1) + np.finfo(np.float32).eps),
            device="cpu",
        )
        del bigram_counts

    num_iters = math.ceil(experiment.num_samples / experiment.batch_size)
    pbar = tqdm.tqdm(total=len(steps) * num_iters, position=gpu_id)
    data_loader = DataLoader(
        load_from_disk(data_path), batch_size=experiment.batch_size
    )
    kl_div_means = []
    kl_div_stds = []

    for step in steps:
        model = experiment.get_model(
            experiment.team, experiment.model_name, step, device=device
        )
        running_bigram_stats = torch.zeros(
            experiment.d_vocab, experiment.d_vocab, device=device, dtype=torch.float32
        )
        data = iter(data_loader)
        for _ in range(num_iters):
            sample = next(data)["input_ids"].to(device)
            logits = model(sample).logits[:, :, : experiment.d_vocab]
            # Upcasting the probs to avoid numerical instability
            probs = F.softmax(logits.to(torch.float32).flatten(0, 1), dim=-1)
            # assert probs.sum(dim=-1).allclose(torch.ones_like(probs.sum(dim=-1)))
            running_bigram_stats.scatter_add_(
                0, sample.flatten().unsqueeze(1).expand(-1, experiment.d_vocab), probs
            )
            pbar.update(1)

        unigram_counts = running_bigram_stats.sum(dim=1).unsqueeze(-1)
        running_bigram_stats /= torch.where(
            unigram_counts >= 1, unigram_counts, torch.ones_like(unigram_counts)
        )
        running_bigram_stats = (
            running_bigram_stats.cpu() + torch.finfo(torch.float32).eps
        )

        # sums = running_bigram_stats.sum(dim=1)
        # if gpu_id == 0:
        #     print(
        #         "assert",
        #         torch.all(
        #             torch.isclose(sums, torch.ones_like(sums))
        #             | torch.isclose(sums, torch.zeros_like(sums))
        #         ),
        #     )

        kl_divs = kl_divergence_log_space(
            dense_bigram_probs.log(), running_bigram_stats.log()
        )
        kl_div_means.append(kl_divs.mean().item())
        kl_div_stds.append(kl_divs.std().item())
        print("train mean", kl_div_means[-1])

    df = pd.DataFrame(
        {
            "step": steps,
            "kl_mean": kl_div_means,
            "kl_std": kl_div_stds,
        }
    )
    df.to_csv(
        Path.cwd()
        / "output"
        / f"finetune_bigram_{experiment.model_name}_\
            {experiment.num_samples}_{gpu_id}.csv",
        index=False,
    )
    return df


def main(ngram_path: str, data_path: str, seed: int = 1):
    # Seed everything
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    experiments = [
        Experiment(
            num_samples=8192,
            batch_size=batch_size,
            seq_len=2048,
            team="EleutherAI",
            model_name=model_name,
            get_model=get_es_finetune,
            d_vocab=50_277,
            steps=[
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
                1024,
                2048,
                4096,
                8192,
                16384,
                32768,
            ],
            ngram_orders=[1, 2],
            eod_index=AutoTokenizer.from_pretrained(
                f"EleutherAI/{model_name}"
            ).eos_token_id,
        )
        for model_name, batch_size in [
            # ("pythia-14m", 2),
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

    for experiment in experiments:
        df = run_checkpoint_experiment_workers(
            experiment, finetuned_stats_worker, ngram_path, data_path
        )

        df.to_csv(
            Path.cwd()
            / "output"
            / f"finetune_bigram_{experiment.model_name}_{experiment.num_samples}-1.csv",
            index=False,
        )


if __name__ == "__main__":
    mp.set_start_method("spawn")

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--ngram_path",
        default="data/es/bigrams.pkl",
        help="Path to pickled sparse scipy array \
            of bigram counts for a data distribution",
    )
    parser.add_argument(
        "--data_path",
        default="data/es/es_tokenized.hf",
        help="Path to data",
    )
    parser.add_argument("--seed", default=1)
    args = parser.parse_args()
    main(args.ngram_path, args.data_path, args.seed)
