import argparse
import math
import os
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import tqdm.auto as tqdm
from datasets import load_from_disk
from optimum.bettertransformer import BetterTransformer
from scipy import stats
from torch.utils.data import DataLoader
from transformers import GPTNeoXForCausalLM

from scripts.scriptutils.ngram_model import NgramModel
from scriptutils.divergences import one_hot_js_divergence, js_divergence, kl_divergence


def get_mean_divergences(
    tokens: torch.Tensor,
    logits: torch.Tensor,
    ngram_model: NgramModel,
    batch: int,
    d_vocab: int,
) -> np.ndarray:
    divergences = []
    logits = logits[:, :-1, :d_vocab].flatten(
        0, 1
    ) 
    sample = tokens[:, 1:].flatten()
    bigram_log_dists = (
        ngram_model.get_bigram_prob(tokens[:, :-1].flatten()).log()
        + torch.finfo(torch.float32).eps
    )
    divergences.append(one_hot_js_divergence(logits, sample, batch).mean())
    divergences.append(
        one_hot_js_divergence(bigram_log_dists, sample, batch).mean()
    )
    del sample

    # divergences.append(kl_divergence(bigram_dists, logits).mean())
    divergences.append(
        js_divergence(bigram_log_dists, logits).mean()
    )  # uses extra 32-25 GB - mem bottleneck. unigrams might too
    del bigram_log_dists

    unigram_dist = ngram_model.unigram_probs + torch.finfo(torch.float32).eps
    # divergences.append(kl_divergence(unigram_dist, logits).mean())
    divergences.append(
        js_divergence(unigram_dist.repeat(2048 * batch, 1), logits).mean()
    )
    labels = [
        "logit_token_js_div",
        "bigram_token_js_div",
        # "bigram_logit_kl_div",
        "bigram_logit_js_div",
        # "unigram_logit_kl_div",
        "unigram_logit_js_div",
    ]
    return torch.stack(divergences), labels


def get_confidence_intervals(
    mean: float, num_items: int, confidence=0.95
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    sem = np.sqrt(mean / num_items)
    conf_intervals = stats.norm.interval(confidence, loc=mean, scale=sem)
    return conf_intervals


@torch.inference_mode()
def ngram_model_worker(
    gpu_id: int,
    steps: list[int],
    model_name: str,
    model_path: str,
    pile_path: str,
    num_samples: int,
    batch: int,
    d_vocab: int,
) -> pd.DataFrame:
    tmp_cache_dir = Path("/mnt/ssd-1/lucia/.cache") / str(gpu_id)
    shutil.rmtree(tmp_cache_dir, ignore_errors=True)

    os.makedirs(tmp_cache_dir, exist_ok=True)
    torch.cuda.set_device(gpu_id)
    ngram_model = NgramModel(model_path, batch)

    div_labels = [
        "logit_token_js_div",
        "bigram_token_js_div",
        # "bigram_logit_kl_div",
        "bigram_logit_js_div",
        # "unigram_logit_kl_div",
        "unigram_logit_js_div",
    ]
    div_means = {label: [] for label in div_labels}
    div_conf_intervals = {label: [] for label in div_labels}

    num_iters = math.ceil(num_samples / batch)
    pbar = tqdm.tqdm(total=len(steps) * num_iters, position=gpu_id)
    for step in steps:
        torch.cuda.synchronize()  # getting out of disk space error
        # possibly from downloading model before the old one is deleted
        pile = iter(DataLoader(load_from_disk(pile_path), batch_size=batch))
        model = GPTNeoXForCausalLM.from_pretrained(
            f"EleutherAI/{model_name}",
            revision=f"step{step}",
            torch_dtype="auto",
            cache_dir=tmp_cache_dir,
        ).cuda()
        model = BetterTransformer.transform(model)
        # running_step_unigram_loss_mean = 0.0
        # running_step_bigram_loss_mean = 0.0
        running_step_div_means = torch.zeros(len(div_labels))
        for i in range(num_iters):
            # unigram_sample = ngram_model.get_ngrams(i, 1)
            # unigram_outputs = model(unigram_sample)
            # unigram_loss_mean = F.cross_entropy(
            #     unigram_outputs.logits[:, :-1].flatten(0, 1),
            #     unigram_sample[:, 1:].flatten(),
            #     reduction="mean",
            # ).item()
            # running_step_unigram_loss_mean += unigram_loss_mean / num_iters
            # bigram_sample = ngram_model.get_ngrams(i, 2)
            # bigram_outputs = model(bigram_sample)
            # bigram_loss_mean = F.cross_entropy(
            #     bigram_outputs.logits[:, :-1].flatten(0, 1),
            #     bigram_sample[:, 1:].flatten(),
            #     reduction="mean",
            # ).item()
            # del bigram_outputs, bigram_sample, unigram_outputs, unigram_sample
            # running_step_bigram_loss_mean += bigram_loss_mean / num_iters
            sample = next(pile)["input_ids"].cuda().to(torch.int32)
            logits = model(sample).logits
            divergences, _ = get_mean_divergences(
                sample, logits, ngram_model, batch, d_vocab
            )
            running_step_div_means += (divergences / num_iters).cpu()
            pbar.update(1)

        # unigram_means.append(running_step_unigram_loss_mean)
        # unigram_conf_intervals.append(
        #     get_confidence_intervals(running_step_unigram_loss_mean, num_iters * batch)
        # )
        # bigram_means.append(running_step_bigram_loss_mean)
        # bigram_conf_intervals.append(
        #     get_confidence_intervals(running_step_bigram_loss_mean, num_iters * batch)
        # )
        for i, label in enumerate(div_labels):
            div_means[label].append(running_step_div_means[i].item())
            div_conf_intervals[label].append(
                get_confidence_intervals(div_means[label][-1], num_iters * batch)
            )

        shutil.rmtree(
            tmp_cache_dir / f"models--EleutherAI--{model_name}", ignore_errors=True
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

    return pd.DataFrame(
        {
            "step": steps,
            # "mean_unigram_loss": unigram_means,
            # "mean_bigram_loss": bigram_means,
            # "bottom_conf_unigram_loss": [
            #     interval[0] for interval in unigram_conf_intervals
            # ],
            # "top_conf_unigram_loss": [
            #     interval[1] for interval in unigram_conf_intervals
            # ],
            # "bottom_conf_bigram_loss": [
            #     interval[0] for interval in bigram_conf_intervals
            # ],
            # "top_conf_bigram_loss": [interval[1] for interval in bigram_conf_intervals],
            **div_mean_data,
            **div_bottom_conf_data,
            **div_top_conf_data,
        }
    )


def main(ngram_path: str, pile_path: str):
    model_batch_sizes = {
        # "pythia-14m": 8,
        # "pythia-70m": 8,
        # "pythia-160m": 4,
        # "pythia-410m": 4,
        # "pythia-1b": 4,
        # "pythia-1.4b": 4,
        # "pythia-2.8b": 4,
        # "pythia-6.9b": 2,
        "pythia-12b": 1,
    }
    # tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    d_vocab = 50277  # len(tokenizer.vocab)
    num_samples = 4096

    log_steps = [0] + [2**i for i in range(int(math.log2(256)) + 1)]
    linear_step_samples = [1000, 2000, 4000, 8000, 16_000, 33_000, 66_000, 131_000]
    # linear_steps = list(range(1000, 144000, 1000))
    # linear_steps.remove(116_000) # missing in 1B
    steps = log_steps + linear_step_samples + [143_000]

    num_gpus = torch.cuda.device_count()
    mp.set_start_method("spawn")

    max_steps_per_chunk = math.ceil(len(steps) / num_gpus)
    step_indices = [
        steps[i : i + max_steps_per_chunk]
        for i in range(0, len(steps), max_steps_per_chunk)
    ]

    for model_name, batch in model_batch_sizes.items():
        args = [
            (
                i,
                step_indices[i],
                model_name,
                ngram_path,
                pile_path,
                num_samples,
                batch,
                d_vocab,
            )
            for i in range(len(step_indices))
        ]
        print(f"Parallelising over {len(step_indices)} GPUs...")
        with mp.Pool(len(step_indices)) as pool:
            dfs = pool.starmap(ngram_model_worker, args)

        df = pd.concat(dfs)
        df.to_csv(
            Path.cwd()
            / "output"
            / f"js_divs_ngrams_model_{model_name}_{num_samples}.csv",
            index=False,
        )


if __name__ == "__main__":
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
        default="val_tokenized.hf",  # '/mnt/ssd-1/lucia/val_tokenized.hf',
        help="Path to Pile validation data",
    )
    args = parser.parse_args()
    main(args.ngram_path, args.pile_path)
