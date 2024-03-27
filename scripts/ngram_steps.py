import argparse
import math
import os
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import tqdm.auto as tqdm
from scipy import stats
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_from_disk
from scripts.scriptutils.ngram_model import NgramModel
from scriptutils.divergences import kl_divergence, js_divergence, one_hot_js_divergence
from scriptutils.load_model import get_neo_tokenizer, get_black_mamba, get_hails_mamba, get_zyphra_mamba
from scriptutils.experiment import Experiment, run_experiment_workers

def batch_generator(dataset, batch_size):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]["input_ids"]


def get_mean_divergences(
    tokens: torch.Tensor,
    logits: torch.Tensor,
    ngram_model: NgramModel,
    batch: int,
    d_vocab: int,
) -> np.ndarray:
    divergences = []
    logits = logits[:, :-1, :d_vocab].flatten(0, 1)
    sample = tokens[:, 1:].flatten()
    bigram_dists = (
        ngram_model.get_bigram_dists(tokens[:, :-1].flatten())
        + torch.finfo(torch.float32).eps
    )
    divergences.append(one_hot_js_divergence(logits, sample, batch).mean())
    divergences.append(one_hot_js_divergence(bigram_dists, sample, batch).mean())
    del sample

    divergences.append(kl_divergence(bigram_dists, logits).mean())
    divergences.append(js_divergence(bigram_dists, logits).mean())
    del bigram_dists

    unigram_dist = ngram_model.unigrams.log() + torch.finfo(torch.float32).eps
    divergences.append(kl_divergence(unigram_dist, logits).mean())
    divergences.append(
        js_divergence(unigram_dist.repeat(2048 * batch, 1), logits).mean()
    )
    labels = [
        "logit_token_js_div",
        "bigram_token_js_div",
        "bigram_logit_kl_div",
        "bigram_logit_js_div",
        "unigram_logit_kl_div",
        "unigram_logit_js_div",
    ]
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
    model_path: str,
    pile_path: str,
    tmp_cache_path: str,
    steps: list[int],
) -> pd.DataFrame:
    torch.cuda.set_device(gpu_id)

    tmp_cache_dir = Path(tmp_cache_path) / str(gpu_id)
    shutil.rmtree(tmp_cache_dir, ignore_errors=True)
    os.makedirs(tmp_cache_dir, exist_ok=True)

    ngram_model = NgramModel(model_path, experiment.batch_size)
    print("Loaded n-gram data...")

    unigram_means = []
    bigram_means = []
    unigram_conf_intervals = []
    bigram_conf_intervals = []
    div_labels = [
        "logit_token_js_div",
        "bigram_token_js_div",
        "bigram_logit_kl_div",
        "bigram_logit_js_div",
        "unigram_logit_kl_div",
        "unigram_logit_js_div",
    ]
    div_means = {label: [] for label in div_labels}
    div_conf_intervals = {label: [] for label in div_labels}

    num_iters = math.ceil(experiment.num_samples / experiment.batch_size)
    pbar = tqdm.tqdm(total=len(steps) * num_iters, position=gpu_id)
    data_loader = DataLoader(load_from_disk(pile_path), batch_size=experiment.batch_size)
    for step in steps:
        pile = iter(data_loader)
        model = experiment.get_model(experiment.team, experiment.model_name, step, tmp_cache_dir)
        running_step_unigram_loss_mean = 0.0
        running_step_bigram_loss_mean = 0.0
        running_step_div_means = torch.zeros(len(div_labels))
        for i in range(num_iters):
            unigram_sample = ngram_model.generate_unigrams().long()
            unigram_logits = model(unigram_sample).logits[:, :, :experiment.d_vocab]

            unigram_loss_mean = F.cross_entropy(
                unigram_logits[:, :-1].reshape(experiment.batch_size * experiment.seq_len, -1),
                unigram_sample[:, 1:].reshape(experiment.batch_size * experiment.seq_len),
                reduction="mean",
            ).item()
            running_step_unigram_loss_mean += unigram_loss_mean / num_iters
            del bigram_sample, bigram_logits

            bigram_sample = ngram_model.generate_bigrams(i).long()
            bigram_logits = model(bigram_sample).logits[:, :, :experiment.d_vocab]
            bigram_loss_mean = F.cross_entropy(
                bigram_logits[:, :-1].reshape(experiment.batch_size * experiment.seq_len, -1),
                bigram_sample[:, 1:].reshape(experiment.batch_size * experiment.seq_len),
                reduction="mean",
            ).item()
            running_step_bigram_loss_mean += bigram_loss_mean / num_iters

            del unigram_sample, unigram_logits

            sample = next(pile)["input_ids"].cuda().to(torch.int32)
            logits = model(sample).logits[:, :, :experiment.d_vocab]
            divergences, _ = get_mean_divergences(
                sample, logits, ngram_model, experiment.batch_size, experiment.d_vocab
            )
            running_step_div_means += (divergences / num_iters).cpu()
            pbar.update(1)

        unigram_means.append(running_step_unigram_loss_mean)
        unigram_conf_intervals.append(
            get_confidence_intervals(running_step_unigram_loss_mean, num_iters * experiment.batch_size)
        )
        bigram_means.append(running_step_bigram_loss_mean)
        bigram_conf_intervals.append(
            get_confidence_intervals(running_step_bigram_loss_mean, num_iters * experiment.batch_size)
        )
        for i, label in enumerate(div_labels):
            div_means[label].append(running_step_div_means[i].item())
            div_conf_intervals[label].append(
                get_confidence_intervals(div_means[label][-1], num_iters * experiment.batch_size)
            )

        shutil.rmtree(
            tmp_cache_dir / f"models--{experiment.team}--{experiment.model_name}", ignore_errors=True
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

    df = pd.DataFrame(
        {
            "step": steps,
            "mean_unigram_loss": unigram_means,
            "mean_bigram_loss": bigram_means,
            "bottom_conf_unigram_loss": [
                interval[0] for interval in unigram_conf_intervals
            ],
            "top_conf_unigram_loss": [
                interval[1] for interval in unigram_conf_intervals
            ],
            "bottom_conf_bigram_loss": [
                interval[0] for interval in bigram_conf_intervals
            ],
            "top_conf_bigram_loss": [interval[1] for interval in bigram_conf_intervals],
            **div_mean_data,
            **div_bottom_conf_data,
            **div_top_conf_data,
        }
    )
    df.to_csv(
        Path.cwd()
        / "output"
        / f"means_ngrams_model_{experiment.model_name}_{experiment.num_samples}_{gpu_id}.csv",
        index=False,
    )
    return df


# zyphra_experiment = Experiment(
#     num_samples=1024,
#     batch_size=4, 
#     seq_len=2048, 
#     team="Zyphra", 
#     model_name="Mamba-370M", 
#     get_model=get_zyphra_mamba, 
#     get_tokenizer=get_neo_tokenizer,
#     d_vocab=50_277, # len(get_neo_tokenizer().vocab) 
#     # roughly log spaced steps + final step
#     steps=[2**i for i in range(int(math.log2(2048)) + 1)] + [10_000, 20_000, 40_000, 80_000, 160_000, 320_000, 610_000]
# )
def main(ngram_path: str, pile_path: str, tmp_cache_path: str):
    experiments = [Experiment(
        num_samples=1024,
        batch_size=4, 
        seq_len=2048, 
        team="hails", 
        model_name="mamba-160m-hf", 
        get_model=get_hails_mamba, 
        get_tokenizer=get_neo_tokenizer,
        d_vocab=50_277, # len(get_neo_tokenizer().vocab) 
        steps=[0] + [2**i for i in range(int(math.log2(256)) + 1)] + [1000, 2000, 4000, 8000, 16_000, 32_000, 61_000, 143_000]
    )]

    for experiment in experiments:
        df = run_experiment_workers(
            experiment, 
            ngram_model_worker, 
            ngram_path, 
            pile_path, 
            tmp_cache_path
        )
        
        df.to_csv(
            Path.cwd()
            / "output"
            / f"means_ngrams_model_{experiment.model_name}_{experiment.num_samples}.csv",
            index=False,
        )


if __name__ == "__main__":
    mp.set_start_method("spawn")

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