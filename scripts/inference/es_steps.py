import argparse
import math
import os
import random
import shutil
from collections import defaultdict
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

from scripts.script_utils.divergences import (
    kl_divergence_log_space,
)
from scripts.script_utils.experiment import (
    Experiment,
    run_checkpoint_experiment_workers,
)
from scripts.script_utils.load_model import get_auto_tokenizer, get_es_finetune
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
    dataset_path: str,
    tmp_cache_path: str,
    steps: list[int],
) -> pd.DataFrame:
    device = "cuda"
    torch.cuda.set_device(gpu_id)

    tmp_cache_dir = Path(tmp_cache_path) / str(gpu_id)
    shutil.rmtree(tmp_cache_dir, ignore_errors=True)
    os.makedirs(tmp_cache_dir, exist_ok=True)

    ngram_model = NgramModel(bigrams_path, experiment.batch_size, device=device)
    # print("Loaded n-gram model...")

    ngram_means = defaultdict(list)
    ngram_conf_intervals = defaultdict(list)

    div_labels = [f"{n}-gram_logit_kl_div" for n in experiment.ngram_orders]
    div_means = {label: [] for label in div_labels}
    div_conf_intervals = {label: [] for label in div_labels}

    num_iters = math.ceil(experiment.num_samples / experiment.batch_size)
    pbar = tqdm.tqdm(total=len(steps) * num_iters, position=gpu_id)
    data_loader = DataLoader(
        load_from_disk(dataset_path), batch_size=experiment.batch_size
    )
    for step in steps:
        data = iter(data_loader)
        model = experiment.get_model(
            experiment.team, experiment.model_name, step, tmp_cache_dir, device=device
        )

        running_step_ngram_loss_means = [0.0] * len(experiment.ngram_orders)
        running_step_div_means = torch.zeros(len(div_labels))
        for i in range(num_iters):
            for n_index, n in enumerate(experiment.ngram_orders):
                ngram_sample = ngram_model.get_ngram_seq(n, i, Path("data/es")).long()
                ngram_logits = model(ngram_sample).logits[:, :, : experiment.d_vocab]

                ngram_loss_mean = F.cross_entropy(
                    ngram_logits[:, :-1].reshape(
                        experiment.batch_size * experiment.seq_len, -1
                    ),
                    ngram_sample[:, 1:].reshape(
                        experiment.batch_size * experiment.seq_len
                    ),
                    reduction="mean",
                ).item()
                running_step_ngram_loss_means[n_index] += ngram_loss_mean / num_iters

            sample = next(data)["input_ids"].to(device).to(torch.int32)
            logits = model(sample).logits[:, :, : experiment.d_vocab]
            # Should be converted to a single n calculation then moved into the n_index loop above
            divergences, _ = get_mean_divergences(
                sample,
                logits,
                ngram_model,
                experiment.batch_size,
                experiment.d_vocab,
                experiment.ngram_orders,
                device,
            )

            running_step_div_means += (divergences / num_iters).cpu()
            pbar.update(1)

        for n_index, n in enumerate(experiment.ngram_orders):
            ngram_means[n].append(running_step_ngram_loss_means[n_index])
            ngram_conf_intervals[n].append(
                get_confidence_intervals(
                    running_step_ngram_loss_means[n_index],
                    num_iters * experiment.batch_size,
                )
            )

        for i, label in enumerate(div_labels):
            div_means[label].append(running_step_div_means[i].item())
            div_conf_intervals[label].append(
                get_confidence_intervals(
                    div_means[label][-1], num_iters * experiment.batch_size
                )
            )

        shutil.rmtree(
            tmp_cache_dir / f"models--{experiment.team}--{experiment.model_name}",
            ignore_errors=True,
        )

    div_mean_data = {f"mean_{label}": div_means[label] for label in div_labels}
    div_bottom_conf_data = {
        f"bottom_conf_{label}": [interval[0] for interval in div_conf_intervals[label]]
        for label in div_labels
    }
    div_top_conf_data = {
        f"top_conf_{label}": [interval[1] for interval in div_conf_intervals[label]]
        for label in div_labels
    }

    ngram_data = {}
    for n in experiment.ngram_orders:
        ngram_data[f"mean_{n}_gram_loss"] = ngram_means[n]
        ngram_data[f"top_conf_{n}_gram_loss"] = [
            interval[1] for interval in ngram_conf_intervals[n]
        ]
        ngram_data[f"bottom_conf_{n}_gram_loss"] = [
            interval[0] for interval in ngram_conf_intervals[n]
        ]

    df = pd.DataFrame(
        {
            "step": steps,
            **ngram_data,
            **div_mean_data,
            **div_bottom_conf_data,
            **div_top_conf_data,
        }
    )
    df.to_csv(
        Path.cwd()
        / "output"
        / f"finetune_bigram_{experiment.model_name}_{experiment.num_samples}_{gpu_id}.csv",
        index=False,
    )
    return df


def main(ngram_path: str, dataset_path: str, tmp_cache_path: str, seed: int = 1):
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
            get_tokenizer=get_auto_tokenizer,
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
            eod_index=get_auto_tokenizer("EleutherAI", model_name).eos_token_id,
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
            experiment,
            finetuned_stats_worker,
            ngram_path,
            dataset_path,
            tmp_cache_path,
            gpu_ids=list(range(8)),
        )

        df.to_csv(
            Path.cwd()
            / "output"
            / f"finetune_bigram_{experiment.model_name}_{experiment.num_samples}_{experiment.ngram_orders}.csv",
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
    parser.add_argument(
        "--tmp_cache_path",
        default=".cache",
        help="Path to cache (repeatedly cleared to free disk space)",
    )
    args = parser.parse_args()
    main(args.ngram_path, args.data_path, args.tmp_cache_path, args.seed)
