import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import tqdm.auto as tqdm
from plot_steps import plot_divs
from scipy import stats
from transformers import AutoTokenizer, GPTNeoXForCausalLM


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


def split_by_eod(
    data: torch.Tensor,
    eod_indices: list[torch.Tensor],
) -> list[np.ndarray]:
    result = []
    start_idx = 0
    for i in range(data.shape[0]):
        for zero_idx in eod_indices[i]:
            if zero_idx > start_idx:
                result.append(data[i, start_idx : zero_idx + 1].cpu().numpy())
            start_idx = zero_idx + 1
        if start_idx < len(data):
            result.append(data[i, start_idx:].cpu().numpy())
    return result


def matrix_summary_stats(
    matrices: list[np.ndarray], target_length=2048
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    padded_matrices = [
        np.pad(m, ((0, target_length - len(m)), (0, 0)), constant_values=np.nan)
        for m in matrices
    ]
    stacked_matrices = np.stack(padded_matrices, axis=0).astype(np.float64)
    means = np.nanmean(stacked_matrices, axis=0)

    standard_errors = stats.sem(stacked_matrices, axis=0, nan_policy="omit")
    conf_intervals = stats.norm.interval(
        0.95,
        loc=means,
        scale=standard_errors,
    )

    return means, conf_intervals


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
    normalized_bigrams: torch.Tensor,
    unigrams: torch.Tensor,
    batch: int,
    d_vocab: int,
) -> np.ndarray:
    bigram_dists = (
        normalized_bigrams[tokens[:, :-1].flatten().cpu()].cuda().log()
        + torch.finfo(torch.float32).eps
    )
    unigram_dist = unigrams.cuda().log() + torch.finfo(torch.float32).eps
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


@torch.inference_mode()
def worker(
    gpu_id: str,
    steps: list[int],
    model_name: str,
    pile_path: str,
    bigrams_path: str,
    num_samples: int,
    d_vocab: int,
) -> pd.DataFrame:
    batch = 1

    with open(bigrams_path, "rb") as f:
        bigrams = pickle.load(f)

    bigrams = torch.tensor(bigrams.toarray(), dtype=torch.float32)
    unigrams = bigrams.sum(dim=1)

    row_sums = unigrams.unsqueeze(-1)
    normalized_bigrams = torch.where(
        (row_sums == 0), 1 / bigrams.shape[0], bigrams / row_sums
    )

    torch.cuda.set_device(gpu_id)
    div_labels = [
        "bigram_logit_kl_div",
        "unigram_logit_kl_div",
        "unigram_logit_js_div",
        "bigram_logit_js_div",
        "bigram_token_js_div",
        "logit_token_js_div",
    ]
    means = {label: [] for label in div_labels}
    bottom_conf_intervals = {label: [] for label in div_labels}
    top_conf_intervals = {label: [] for label in div_labels}
    step_data = []
    token_indices = []
    for step in tqdm.tqdm(steps, position=gpu_id):
        pile = Pile(pile_path, batch)
        model = GPTNeoXForCausalLM.from_pretrained(
            model_name, revision=f"step{step}", torch_dtype="auto"
        ).cuda()

        step_divergences = []
        for _ in range(num_samples // batch):
            sample = next(pile)
            eod_indices = [torch.where(sample[i] == 0)[0] for i in range(len(sample))]
            outputs = model(sample)
            divergences, labels = get_divergences(
                sample, outputs.logits, normalized_bigrams, unigrams, batch, d_vocab
            )
            step_divergences.extend(split_by_eod(divergences, eod_indices))

        mean_step_divs, conf_intervals = matrix_summary_stats(
            step_divergences, target_length=2048
        )
        for i in range(len(div_labels)):
            means[div_labels[i]].extend(mean_step_divs[:, i])
            bottom_conf_intervals[div_labels[i]].extend(conf_intervals[0][:, i])
            top_conf_intervals[div_labels[i]].extend(conf_intervals[1][:, i])

        step_data.extend([step] * 2048)
        token_indices.extend(list(range(2048)))

    index_data = {"step": step_data, "index": token_indices}
    mean_data = {f"mean_{label}": means[label] for label in div_labels}
    bottom_conf_data = {
        f"bottom_conf_{label}": bottom_conf_intervals[label] for label in div_labels
    }
    top_conf_data = {
        f"top_conf_{label}": top_conf_intervals[label] for label in div_labels
    }
    return pd.DataFrame(
        {**index_data, **mean_data, **bottom_conf_data, **top_conf_data}
    )


def main():
    model_name = "EleutherAI/pythia-410m"
    pile_path = "/mnt/ssd-1/pile_preshuffled/standard/document.bin"
    bigrams_path = "/mnt/ssd-1/lucia/pythia-deduped-bigrams.pkl"
    num_samples = 16
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    d_vocab = len(tokenizer.vocab)  # 50277

    log_steps = [0] + [2**i for i in range(int(math.log2(512)) + 1)]
    linear_steps = list(range(1000, 144000, 1000))
    steps = log_steps + linear_steps

    num_gpus = torch.cuda.device_count()
    max_steps_per_chunk = math.ceil(len(steps) / num_gpus)
    step_indices = [
        steps[i : i + max_steps_per_chunk]
        for i in range(0, len(steps), max_steps_per_chunk)
    ]
    args = [
        (i, step_indices[i], model_name, pile_path, bigrams_path, num_samples, d_vocab)
        for i in range(num_gpus)
    ]

    print(f"Parallelising over {num_gpus} GPUs...")
    mp.set_start_method("spawn")
    with mp.Pool(num_gpus) as pool:
        dfs = pool.starmap(worker, args)

    with open(Path.cwd() / "output" / "step_divergences.pkl", "wb") as f:
        pickle.dump(pd.concat(dfs), f)

    plot_divs()


if __name__ == "__main__":
    main()
