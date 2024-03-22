import argparse
import math
import os
import pickle
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Callable


import numpy as np
import pandas as pd
import scipy
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import tqdm.auto as tqdm
from mamba_ssm import MambaLMHeadModel
from datasets import load_from_disk
from scipy import stats
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from utils.divergences import kl_divergence, js_divergence, one_hot_js_divergence


def batch_generator(dataset, batch_size):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]["input_ids"]


class NgramModel:
    def __init__(self, path: str, batch=1, seq_len=2049):
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        self.d_vocab = len(self.tokenizer.vocab)
        self.batch = batch
        self.seq_len = seq_len

        with open(path, "rb") as f:
            bigram_counts = pickle.load(f)
        bigram_counts = bigram_counts.toarray().astype(np.float32)

        self.unigrams = torch.tensor(bigram_counts).sum(dim=1).cuda()
        self.unigrams /= self.unigrams.sum()

        # Conver to sparse CSR tensor in a dumb way
        sparse_bigram_probs = (
            torch.tensor(
                bigram_counts / (bigram_counts.sum(axis=1) + np.finfo(np.float32).eps)
            )
            .log()
            .to_sparse()
        )

        indices = sparse_bigram_probs.indices().numpy()
        values = sparse_bigram_probs.values().numpy()
        shape = sparse_bigram_probs.shape
        sparse_csr_bigram_probs = scipy.sparse.coo_matrix(
            (values, (indices[0], indices[1])), shape=shape
        ).tocsr()
        self.log_bigrams = torch.sparse_csr_tensor(
            sparse_csr_bigram_probs.indptr.astype(np.int64),
            sparse_csr_bigram_probs.indices.astype(np.int64),
            sparse_csr_bigram_probs.data.astype(np.float32),
            dtype=torch.float32,
            device="cuda",
        )

        self.bigram_samples = np.load("bigram-sequences.npy")

    def generate_unigrams(self) -> torch.Tensor:
        return torch.multinomial(
            self.unigrams, self.batch * self.seq_len, replacement=True
        ).reshape(self.batch, self.seq_len)

    def generate_unigram_strs(self) -> list[str]:
        tokens = self.generate_unigrams()
        return [self.tokenizer.decode(row.tolist()) for row in tokens]

    def get_bigram_dists(self, prev: torch.Tensor) -> torch.Tensor:
        starts = self.log_bigrams.crow_indices()[prev]
        ends = self.log_bigrams.crow_indices()[prev + 1]

        # 0 padding to batch rows with variable numbers of non-zero elements
        bigram_dists = torch.zeros(
            (len(prev), self.d_vocab), dtype=torch.float32, device="cuda"
        )
        for i in range(len(prev)):
            filled_col_indices = self.log_bigrams.col_indices()[starts[i] : ends[i]]
            filled_col_values = self.log_bigrams.values()[starts[i] : ends[i]]
            bigram_dists[i][filled_col_indices] = filled_col_values
        return bigram_dists

    # # separate slice sparse array function with test
    # def sample_bigram(self, prev: torch.Tensor) -> torch.Tensor:
    #     """Given a batch of previous tokens, sample from a bigram model
    #     using conditional distributions stored in a sparse CSR tensor."""
    #     starts = self.bigrams.crow_indices()[prev]
    #     ends = self.bigrams.crow_indices()[prev + 1]

    #     # 0 padding to batch rows with variable numbers of non-zero elements
    #     token_probs = torch.zeros(
    #         (self.batch, self.d_vocab), dtype=torch.float32, device="cuda"
    #     )
    #     token_col_indices = torch.zeros(
    #         (self.batch, self.d_vocab), dtype=torch.int32, device="cuda"
    #     )
    #     for i in range(self.batch):
    #         token_probs[i, : ends[i] - starts[i]] = self.bigrams.values()[
    #             starts[i] : ends[i]
    #         ]
    #         token_col_indices[i, : ends[i] - starts[i]] = self.bigrams.col_indices()[
    #             starts[i] : ends[i]
    #         ]

    #     sampled_value_indices = torch.multinomial(token_probs, 1)
    #     return torch.gather(token_col_indices, 1, sampled_value_indices)

    def generate_bigrams(self, i: int) -> torch.Tensor:
        """Auto-regressively generate bigram model sequence. Initialize each
        sequence by sampling from a unigram model."""
        # i should range from 0 to 1024/2
        batch = self.bigram_samples[
            i * self.batch : (i * self.batch) + self.batch, :50277
        ]
        return torch.tensor(batch, device="cuda").long()
        # result = [
        #     torch.multinomial(self.unigrams, self.batch, replacement=True).unsqueeze(1)
        # ]
        # for _ in range(self.seq_len - 1):
        #     prev = result[-1]
        #     result.append(self.sample_bigram(prev))
        # return torch.cat(result, dim=1)

    def generate_bigram_strs(self) -> list[str]:
        tokens = self.generate_bigrams()
        return [self.tokenizer.decode(row.tolist()) for row in tokens]


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
    conf_intervals = stats.norm.interval(confidence, loc=mean, scale=sem)
    return conf_intervals


@dataclass
class Experiment:
    def __init__(self, team: str, model_name: str, batch_size: int, seq_len: int, get_model: Callable, get_tokenizer: Callable):
        self.team = team
        self.model_name = model_name
        self.batch_size = batch_size
        self.get_model = get_model
        self.get_tokenizer = get_tokenizer
        self.seq_len = seq_len

    
@torch.inference_mode()
def ngram_model_worker(
    gpu_id: int,
    steps: list[int],
    experiment: Experiment,
    model_path: str,
    pile_path: str,
    tmp_cache_path: str,
    num_samples: int,
    d_vocab: int,
) -> pd.DataFrame:
    tmp_cache_dir = Path(tmp_cache_path) / str(gpu_id)
    shutil.rmtree(tmp_cache_dir, ignore_errors=True)
    os.makedirs(tmp_cache_dir, exist_ok=True)
    torch.cuda.set_device(gpu_id)
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

    num_iters = math.ceil(num_samples / experiment.batch_size)
    pbar = tqdm.tqdm(total=len(steps) * num_iters, position=gpu_id)
    for step in steps:
        pile = iter(DataLoader(load_from_disk(pile_path), batch_size=experiment.batch_size))
        model = experiment.get_model(experiment.team, experiment.model_name, step)
        running_step_unigram_loss_mean = 0.0
        running_step_bigram_loss_mean = 0.0
        running_step_div_means = torch.zeros(len(div_labels))
        for i in range(num_iters):
            unigram_sample = ngram_model.generate_unigrams()
            unigram_logits = model(unigram_sample).logits[:, :, :d_vocab]

            unigram_loss_mean = F.cross_entropy(
                unigram_logits[:, :-1].reshape(experiment.batch_size * experiment.seq_len, -1),
                unigram_sample[:, 1:].reshape(experiment.batch_size * experiment.seq_len),
                reduction="mean",
            ).item()
            running_step_unigram_loss_mean += unigram_loss_mean / num_iters

            bigram_sample = ngram_model.generate_bigrams(i)
            bigram_logits = model(bigram_sample).logits[:, :, :d_vocab]
            bigram_loss_mean = F.cross_entropy(
                bigram_logits[:, :-1].reshape(experiment.batch_size * experiment.seq_len, -1),
                bigram_sample[:, 1:].reshape(experiment.batch_size * experiment.seq_len),
                reduction="mean",
            ).item()
            running_step_bigram_loss_mean += bigram_loss_mean / num_iters

            del bigram_sample, unigram_sample

            sample = next(pile)["input_ids"].cuda().to(torch.int32)
            logits = model(sample).logits[:, :, :d_vocab]
            divergences, _ = get_mean_divergences(
                sample, logits, ngram_model, experiment.batch_size, d_vocab
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

    return pd.DataFrame(
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
        

def get_neo_tokenizer():
    return AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")


def get_zyphra_mamba(team: str, model_name: str, step: int):
    return MambaLMHeadModel.from_pretrained(
        f"{team}/{model_name}",
        iteration=step,
        device="cuda"
    )


# def get_eleuther_mamba(team: str, model_name: str, step: int, tmp_cache_dir: str):
#     # AutoModel, MambaLMHeadModel, MambaForCausalLM
#     return MambaLMHeadModel.from_pretrained(
#         f"{team}/{model_name}", 
#         revision=f"step{step}",
#         # torch_dtype="auto",
#         # cache_dir=tmp_cache_dir,
#         device="cuda"
#     )


def main(ngram_path: str, pile_path: str, tmp_cache_path: str):
    get_zyphra_mamba("Zyphra", "Mamba-370M", 0)
    experiments = [Experiment(
        team="Zyphra", 
        model_name="Mamba-370M", 
        batch_size=2, 
        seq_len=2048, 
        get_model=get_zyphra_mamba, 
        get_tokenizer=get_neo_tokenizer
    )]
    
    d_vocab = 50277  # len(tokenizer.vocab) 
    num_samples = 1024

    log_steps = [0] + [2**i for i in range(int(math.log2(256)) + 1)]
    linear_step_samples = [1000, 2000, 4000, 8000, 16_000, 33_000, 66_000, 131_000]
    steps = log_steps + linear_step_samples + [143_000]

    num_gpus = torch.cuda.device_count()
    mp.set_start_method("spawn")

    max_steps_per_chunk = math.ceil(len(steps) / num_gpus)
    step_indices = [
        steps[i : i + max_steps_per_chunk]
        for i in range(0, len(steps), max_steps_per_chunk)
    ]

    for experiment in experiments:
        args = [
            (
                i,
                step_indices[i],
                experiment,
                ngram_path,
                pile_path,
                tmp_cache_path,
                num_samples,
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
            / f"means_ngrams_model_{experiment.model_name}_{num_samples}_kl_div.csv",
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