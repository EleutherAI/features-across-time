import math
import pickle
from pathlib import Path
import argparse
import os
import shutil

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import tqdm.auto as tqdm
from plot_steps import plot_ngram_model_bpb
from scipy import stats
from transformers import AutoTokenizer, GPTNeoXForCausalLM, logging


class Pile:
    def __init__(self, path: str, batch=1):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.index = len(self.data)
        self.chunk_size = 2049
        self.batch = batch

    def __iter__(self):
        return self

    def __next__(self):
        if self.index <= 0:
            raise StopIteration

        start_index = max(self.index - (self.chunk_size * self.batch), 0)
        sample = self.data[start_index : self.index]
        self.index = start_index

        return torch.from_numpy(sample.astype(np.int64)).reshape(self.batch, -1).cuda()


class NgramModel:
    def __init__(self, path: str, d_vocab: int, batch=1, seq_len=2049):
        self.d_vocab = d_vocab
        self.batch = batch
        self.seq_len = seq_len

        with open(path, "rb") as f:
            bigram_counts = pickle.load(f)

        self.unigrams = (
            torch.tensor(bigram_counts.toarray().astype(np.float32)).sum(dim=1).cuda()
        )
        self.bigrams = torch.sparse_csr_tensor(
            bigram_counts.indptr.astype(np.int64),
            bigram_counts.indices.astype(np.int64),
            bigram_counts.data.astype(np.float32),
            dtype=torch.float32,
            device="cuda",
        )

    def generate_unigrams(self) -> torch.Tensor:
        return torch.multinomial(self.unigrams, self.batch * self.seq_len).reshape(
            self.batch, self.seq_len
        )

    def generate_bigrams(self) -> torch.Tensor:
        """Auto-regressively sample tokens using conditional distributions stored
        in a sparse CSR tensor. Each sequence is initialized by sampling from a
        unigram distribution."""
        result = [torch.multinomial(self.unigrams, self.batch).unsqueeze(1)]
        for _ in range(self.seq_len - 1):
            prev = result[-1]
            starts = self.bigrams.crow_indices()[prev]
            ends = self.bigrams.crow_indices()[prev + 1]

            # 0 padding to batch rows with variable numbers of non-zero elements
            token_probs = torch.zeros((self.batch, self.d_vocab), device="cuda")
            token_col_indices = torch.zeros(
                (self.batch, self.d_vocab), dtype=torch.int32, device="cuda"
            )
            for i in range(self.batch):
                token_probs[i, : ends[i] - starts[i]] = self.bigrams.values()[
                    starts[i] : ends[i]
                ]
                token_col_indices[
                    i, : ends[i] - starts[i]
                ] = self.bigrams.col_indices()[starts[i] : ends[i]]

            sampled_value_indices = torch.multinomial(token_probs, 1)
            sampled_col_indices = torch.gather(
                token_col_indices, 1, sampled_value_indices
            )
            result.append(sampled_col_indices)
        return torch.cat(result, dim=1)



def kl_divergence(
    logit_p: torch.Tensor, logit_q: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """Compute the KL divergence between two sets of logits."""
    log_p = logit_p.log_softmax(dim)
    log_q = logit_q.log_softmax(dim)
    return torch.nansum(log_p.exp() * (log_p - log_q), dim)


def js_divergence(
    logit_p: torch.Tensor, logit_q: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """Compute the Jensen-Shannon divergence between two sets of logits."""
    log_p = logit_p.log_softmax(dim)
    log_q = logit_q.log_softmax(dim)

    # Mean of P and Q
    log_m = torch.stack([log_p, log_q]).sub(math.log(2)).logsumexp(0)

    kl_p = torch.nansum(log_p.exp() * (log_p - log_m), dim)
    kl_q = torch.nansum(log_q.exp() * (log_q - log_m), dim)
    return 0.5 * (kl_p + kl_q)


def get_divergences(
    tokens: torch.Tensor,
    logits: torch.Tensor,
    bigrams: torch.Tensor,
    unigrams: torch.Tensor,
    batch: int,
    d_vocab: int,
) -> np.ndarray:
    bigram_dists = (
        bigrams[tokens[:, :-1].flatten()].log()
        + torch.finfo(torch.float32).eps
    )
    unigram_dist = unigrams.log() + torch.finfo(torch.float32).eps
    sample = tokens[:, 1:].view(batch * 2048)
    token_dists = F.one_hot(sample, num_classes=d_vocab).float().log()

    logits = logits[:, :-1, :d_vocab].view(batch * 2048, -1)

    labels = [
        "bigram_logit_kl_div",
        "unigram_logit_kl_div",
        "unigram_logit_js_div",
        "bigram_logit_js_div",
        "bigram_token_js_div",
        "logit_token_js_div",
    ]

    divergences = [
        kl_divergence(bigram_dists, logits).view(batch, -1),
        kl_divergence(unigram_dist, logits).view(batch, -1),
        js_divergence(unigram_dist.repeat(2048, 1), logits).view(batch, -1),
        js_divergence(bigram_dists, logits).view(batch, -1),
        js_divergence(bigram_dists, token_dists).view(batch, -1),
        js_divergence(logits, token_dists).view(batch, -1),
    ]
    divergences = torch.stack(divergences, dim=-1)
    return divergences, labels


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
    torch.cuda.set_device(gpu_id)
    ngram_model = NgramModel(model_path, d_vocab, batch)

    unigram_means = []
    bigram_means = []
    unigram_conf_intervals = []
    bigram_conf_intervals = []
    
    div_labels = [
        "bigram_logit_kl_div",
        "unigram_logit_kl_div",
        "unigram_logit_js_div",
        "bigram_logit_js_div",
        "bigram_token_js_div",
        "logit_token_js_div",
    ]
    div_means = {label: [] for label in div_labels}
    div_conf_intervals = {label: [] for label in div_labels}

    for step in tqdm.tqdm(steps, position=gpu_id):
        pile = Pile(pile_path, batch)

        os.makedirs(f'/root/.cache/{gpu_id}', exist_ok=True)
        model = GPTNeoXForCausalLM.from_pretrained(
            f'EleutherAI/{model_name}', revision=f"step{step}", torch_dtype="auto", cache_dir=f'/root/.cache/{gpu_id}'
        ).cuda()

        running_step_unigram_loss_mean = 0.0
        running_step_bigram_loss_mean = 0.0
        running_step_div_means = torch.zeros(len(div_labels))
        num_iters = math.ceil(num_samples / batch)
        for _ in range(num_iters):
            unigram_sample = ngram_model.generate_unigrams()
            unigram_outputs = model(unigram_sample)
            unigram_loss_mean = F.cross_entropy(
                unigram_outputs.logits[:, :-1].flatten(0, 1),
                unigram_sample[:, 1:].flatten(),
                reduction="mean",
            ).item()
            running_step_unigram_loss_mean += (unigram_loss_mean / num_iters)
            del unigram_sample, unigram_outputs

            bigram_sample = ngram_model.generate_bigrams()
            bigram_outputs = model(bigram_sample)
            bigram_loss_mean = F.cross_entropy(
                bigram_outputs.logits[:, :-1].flatten(0, 1),
                bigram_sample[:, 1:].flatten(),
                reduction="mean",
            ).item()
            running_step_bigram_loss_mean += (bigram_loss_mean / num_iters)

            sample = next(pile)
            outputs = model(sample)
            divergences, _ = get_divergences(
                sample, outputs.logits, ngram_model.unigrams, ngram_model.bigrams, batch, d_vocab
            )
            running_step_div_means += (divergences.mean(dim=0) / num_iters).cpu()

        unigram_means.append(running_step_unigram_loss_mean)
        unigram_conf_intervals.append(
            get_confidence_intervals(
                running_step_unigram_loss_mean, num_iters * batch * 2048
            )
        )
        bigram_means.append(running_step_bigram_loss_mean)
        bigram_conf_intervals.append(
            get_confidence_intervals(
                running_step_bigram_loss_mean, num_iters * batch
            )
        )
        for i, label in enumerate(div_labels):
            div_means[label].append(running_step_div_means[i])
            div_conf_intervals[label].append(
                get_confidence_intervals(
                    running_step_bigram_loss_mean[i], num_iters * batch
                )
            )

        shutil.rmtree(f'/root/.cache/{gpu_id}', ignore_errors=True)

    div_mean_data = {f"mean_{label}": div_means[label] for label in div_labels}
    div_bottom_conf_data = {
        f"bottom_conf_{label}": bigram_conf_intervals[label][0] for label in div_labels
    }
    div_top_conf_data = {
        f"top_conf_{label}": bigram_conf_intervals[label][1] for label in div_labels
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
            **div_top_conf_data
        }
    )


def main(ngram_path: str, pile_path: str):
    logging.set_verbosity_error()
    model_batch_sizes = {
        # "pythia-14m": 32,
        # "pythia-70m": 32,
        # "pythia-160m": 16,
        # "pythia-410m": 16,
        # "pythia-1b": 8,
        # "pythia-1.4b": 8,
        # "pythia-2.8b": 4,
        "pythia-6.9b": 1,   
        "pythia-12b": 1,
    }
    # tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    d_vocab = 50277 # len(tokenizer.vocab)
    num_samples = 1024

    log_steps = [0] + [2**i for i in range(int(math.log2(512)) + 1)]
    linear_steps = list(range(1000, 144000, 1000))
    linear_steps.remove(116_000) # missing
    steps = log_steps + linear_steps

    num_gpus = torch.cuda.device_count()
    mp.set_start_method("spawn")

    max_steps_per_chunk = math.ceil(len(steps) / num_gpus)
    step_indices = [
        steps[i : i + max_steps_per_chunk]
        for i in range(0, len(steps), max_steps_per_chunk)
    ]

    for model_name, batch in model_batch_sizes.items():
        args = [
            (i, step_indices[i], model_name, ngram_path, pile_path, num_samples, batch, d_vocab)
            for i in range(num_gpus)
        ]
        print(f"Parallelising over {num_gpus} GPUs...")
        with mp.Pool(num_gpus) as pool:
            dfs = pool.starmap(ngram_model_worker, args)

        with open(Path.cwd() / "output" / f"step_ngrams_model_means_{model_name}.pkl", "wb") as f:
            pickle.dump(pd.concat(dfs), f)

    plot_ngram_model_bpb()


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
        default="/mnt/ssd-1/lucia",
        help="Path to Pile validation data",
    )
    args = parser.parse_args()
    main(args.ngram_path, args.pile_path)
