import argparse
import math
import os
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
    # js_divergence,
    kl_divergence_log_space,
)
from scripts.script_utils.experiment import (
    Experiment,
    run_checkpoint_experiment_workers,
)
from scripts.script_utils.load_model import get_auto_model, get_auto_tokenizer
from scripts.script_utils.ngram_model import NgramModel


def get_mean_divergences(
    tokens: torch.Tensor,
    logits: torch.Tensor,
    ngram_model: NgramModel,
    batch: int,
    d_vocab: int,
    ngram_orders: list[int],
) -> np.ndarray:
    divergences = []
    labels = [
        # "logit_token_js_div",
        # "bigram_token_js_div",
    ]
    logits = logits[:, :, :d_vocab].flatten(0, 1)
    # sample = tokens[:, 1:].flatten()
    # divergences.append(one_hot_js_divergence(logits, sample, batch).mean())
    # divergences.append(one_hot_js_divergence(bigram_dists, sample, batch).mean())
    # del sample

    for n in ngram_orders:
        ngram_dists = ngram_model.get_ngram_prob(tokens, n).cuda().log().flatten(0, 1)
        divergences.append(kl_divergence_log_space(ngram_dists, logits).mean())
        # divergences.append(js_divergence(ngram_dists, logits).mean())

        labels.append(f"{n}-gram_logit_kl_div")
        # labels.append(f"{n}-gram_logit_js_div")

    return torch.stack(divergences), labels


def get_confidence_intervals(
    mean: float, num_items: int, confidence=0.95
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    sem = np.sqrt(mean / num_items)
    return stats.norm.interval(confidence, loc=mean, scale=sem)


@torch.inference_mode()
def ngram_model_worker(
    gpu_id: int,
    experiment: Experiment,
    model_path: Path,
    pile_path: str,
    steps: list[int],
) -> pd.DataFrame:
    torch.cuda.set_device(gpu_id)

    ngram_model = NgramModel(str(model_path / "bigrams.pkl"), experiment.batch_size)
    print("Loaded n-gram model...")

    ngram_means = defaultdict(list)
    ngram_conf_intervals = defaultdict(list)

    div_labels = [f"{n}-gram_logit_kl_div" for n in experiment.ngram_orders]
    # Ordering bug in div fn do not uncomment
    # + [
    # f"{n}-gram_logit_js_div" for n in experiment.ngram_orders
    # ]

    div_means = {label: [] for label in div_labels}
    div_conf_intervals = {label: [] for label in div_labels}

    num_iters = math.ceil(experiment.num_samples / experiment.batch_size)
    pbar = tqdm.tqdm(total=len(steps) * num_iters, position=gpu_id)
    pile_data_loader = DataLoader(
        load_from_disk(pile_path), batch_size=experiment.batch_size
    )

    ngram_samples = [
        load_from_disk(str(model_path / f"{n}-gram-sequences.hf"))
        for n in experiment.ngram_orders
    ]
    for ngram_sample in ngram_samples:
        ngram_sample.set_format("torch", columns=["input_ids"])

    def get_ngram_seq(n: int, i: int, batch_size: int) -> torch.Tensor:
        """Fetch a precomputed batch of n-gram sequences"""
        return ngram_samples[n][i * batch_size : (i * batch_size) + batch_size][
            "input_ids"
        ]

    for step in steps:
        print(step)
        pile = iter(pile_data_loader)
        running_step_ngram_loss_means = [0.0] * len(experiment.ngram_orders)
        running_step_div_means = torch.zeros(len(div_labels))

        try:
            model = experiment.get_model(
                experiment.team, experiment.model_name, step
            )
        except:
            print("Failed to load model")
            for n_index, n in enumerate(experiment.ngram_orders):
                ngram_means[n].append(np.nan)
                ngram_conf_intervals[n].append((np.nan, np.nan))

            for i, label in enumerate(div_labels):
                div_means[label].append(np.nan)
                div_conf_intervals[label].append((np.nan, np.nan))
            continue

        for i in range(num_iters):
            for n_index, n in enumerate(experiment.ngram_orders):
                ngram_sample = (
                    get_ngram_seq(n_index, i, experiment.batch_size).cuda().long()
                )
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

            sample = next(pile)["input_ids"].cuda().to(torch.int32)
            logits = model(sample).logits[:, :, : experiment.d_vocab]
            divergences, _ = get_mean_divergences(
                sample,
                logits,
                ngram_model,
                experiment.batch_size,
                experiment.d_vocab,
                experiment.ngram_orders,
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
        / f"means_ngrams_model_{experiment.model_name}_\
{experiment.num_samples}_steps_{gpu_id}.csv",
        index=False,
    )
    return df


def main(ngram_path: str, pile_path: str):
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
            # roughly log spaced steps + final step
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
            ngram_orders=[1, 2],
            eod_index=get_auto_tokenizer("EleutherAI", "pythia-14m").eos_token_id,
        )
        for model_name, batch_size in [
            # ("pythia-14m", 8),
            # ("pythia-70m", 4),
            # ("pythia-160m", 4),
            # ("pythia-410m", 4),
            # ("pythia-1b", 4),
            # ("pythia-1.4b", 4),
            # ("pythia-2.8b", 4),
            # ("pythia-6.9b", 1),
            ("pythia-12b", 1),
        ]
        # + [(f"pythia-14m-seed{i}", 8) for i in range(1, 10)]
        # + [(f"pythia-70m-seed{i}", 2) for i in range(1, 10)]
        # + [(f"pythia-160m-seed{i}", 4) for i in range(1, 10)]
        # + [(f"pythia-410m-seed{i}", 2) for i in range(1, 5)]
        # + [(f"pythia-14m-warmup01", 8), (f"pythia-70m-warmup01", 4)]
    ]

    for experiment in experiments:
        existing_df = pd.read_csv(
            Path.cwd()
            / "output"
            / "24-06-05"
            / f"means_ngrams_model_{experiment.model_name}_\
{experiment.num_samples}.csv"
        )
        # get the steps of rows that contain nans in the existing df
        # steps = existing_df[existing_df.isnull().any(axis=1)]["step"].tolist()
        steps = existing_df[existing_df['mean_2-gram_logit_kl_div'].isnull()]["step"].tolist()

        if not steps:
            continue
        print(steps)
        experiment.steps = steps

        df = run_checkpoint_experiment_workers(
            experiment, ngram_model_worker, ngram_path, pile_path
        )

        df.to_csv(
            Path.cwd()
            / "output"
            / f"means_ngrams_model_{experiment.model_name}_\
{experiment.num_samples}_steps.csv",
            index=False,
        )


#         df = run_checkpoint_experiment_workers(
#             experiment,
#             ngram_model_worker,
#             ngram_path,
#             pile_path,
#         )

#         df.to_csv(
#             Path.cwd()
#             / "output"
#             / f"means_ngrams_model_{experiment.model_name}_\
# {experiment.num_samples}.csv",
#             index=False,
#         )


if __name__ == "__main__":
    mp.set_start_method("spawn")

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--ngram_path",
        default="data/pile-deduped",
        help="Path to n-gram data: pickled sparse scipy array of \
    bigram counts; hf datasets of n-gram sequences",
    )
    parser.add_argument(
        "--pile_path",
        default="/mnt/ssd-1/lucia/val_tokenized.hf",
        help="Path to Pile validation data",
    )
    args = parser.parse_args()
    main(Path(args.ngram_path), args.pile_path)
