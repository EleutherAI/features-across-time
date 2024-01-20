import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import tqdm.auto as tqdm
from scipy import stats
from transformers import AutoTokenizer, GPTNeoXForCausalLM


class BigramModel:
    def __init__(self, path: str, batch=1, seq_len=2049):
        with open(path, "rb") as f:
            bigrams = pickle.load(f)

        self.bigrams = torch.tensor(bigrams.toarray(), dtype=torch.float32)
        self.unigrams = self.bigrams.sum(dim=1).cuda()
        self.batch = batch
        self.seq_len = seq_len

    def __iter__(self):
        return self

    def __next__(self) -> torch.Tensor:
        result = [torch.multinomial(self.unigrams, self.batch)]
        for _ in range(self.seq_len - 1):
            token_dists = self.bigrams[result[-1].cpu()].cuda()
            result.append(torch.multinomial(token_dists, 1).squeeze())

        assert result[-1].shape == result[0].shape
        print(torch.stack(result, dim=-1))
        return torch.stack(result, dim=-1)


class UnigramModel:
    def __init__(self, path: str, batch=1, seq_len=2049):
        with open(path, "rb") as f:
            bigrams = pickle.load(f)

        bigrams = torch.tensor(bigrams.toarray()).float()
        self.unigrams = bigrams.sum(dim=1).cuda()
        self.batch = batch
        self.seq_len = seq_len

    def __iter__(self):
        return self

    def __next__(self) -> torch.Tensor:
        result = []
        for _ in self.batch():
            result.append([torch.multinomial(self.unigrams, self.seq_len)])

        assert result[-1].shape == result[0].shape
        return torch.stack(result, dim=0)


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


def summary_stats(
    vectors: list[np.ndarray], target_length=2048
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    padded_vectors = [
        np.pad(v, (0, target_length - len(v)), constant_values=np.nan) for v in vectors
    ]
    stacked_vectors = np.vstack(padded_vectors).astype(np.float64)
    means = np.nanmean(stacked_vectors, axis=0)
    standard_errors = stats.sem(stacked_vectors, axis=0, nan_policy="omit")
    conf_intervals = stats.norm.interval(0.95, loc=means, scale=standard_errors)

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


def get_metrics(
    tokens: torch.Tensor,
    logits: torch.Tensor,
    normalized_bigrams: torch.Tensor,
    unigrams: torch.Tensor,
    batch: int,
    d_vocab: int,
) -> np.ndarray:
    """We can calculate divergence from bigram stats at every position
    but we throw away the final position to maintain consistency with
    divergence from tokens"""
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
        "unigram_logit_kl_div" "unigram_logit_js_div" "bigram_logit_js_div",
        "bigram_token_js_div",
        "logit_token_js_div",
    ]

    metrics = [
        kl_divergence(bigram_dists, logits).view(batch, -1),
        kl_divergence(unigram_dist, logits).view(batch, -1),
        js_divergence(unigram_dist.repeat(2048, 1), logits).view(batch, -1),
        js_divergence(bigram_dists, logits).view(batch, -1),
        js_divergence(bigram_dists, token_dists).view(batch, -1),
        js_divergence(logits, token_dists).view(batch, -1),
    ]
    metrics = torch.stack(metrics, dim=-1)
    return metrics, labels


@torch.inference_mode()
def unigram_model_worker(
    gpu_id: str,
    steps: list[int],
    model_name: str,
    pile_path: str,
    num_samples: int,
    d_vocab: int,
) -> pd.DataFrame:
    batch = 4
    torch.cuda.set_device(gpu_id)
    unigram_model = UnigramModel("/mnt/ssd-1/lucia/pythia-deduped-bigrams.pkl", batch)

    mean_losses = []
    bottom_conf_intervals = []
    top_conf_intervals = []
    step_data = []
    token_indices = []
    for step in tqdm.tqdm(steps, position=gpu_id):
        model = GPTNeoXForCausalLM.from_pretrained(
            model_name, revision=f"step{step}", torch_dtype="auto"
        ).cuda()

        step_losses = []
        for _ in range(num_samples // batch):
            sample = next(unigram_model)
            eod_indices = [torch.where(sample[i] == 0)[0] for i in range(len(sample))]
            outputs = model(sample)
            loss = F.cross_entropy(
                outputs.logits[:, :-1].reshape(batch * 2048, -1),
                sample[:, 1:].reshape(batch * 2048),
                reduction="none",
            ).reshape(batch, 2048)
            step_losses.extend(split_by_eod(loss, eod_indices))

        mean_step_loss, conf_intervals = summary_stats(step_losses, target_length=2048)
        mean_losses.extend(mean_step_loss)
        bottom_conf_intervals.extend(conf_intervals[0])
        top_conf_intervals.extend(conf_intervals[1])

        step_data.extend([step] * 2048)
        token_indices.extend(list(range(2048)))

    index_data = {"step": step_data, "index": token_indices}
    unigram_data = {
        "mean_losses": mean_losses,
        "bottom_conf_intervals": bottom_conf_intervals,
        "top_conf_intervals": top_conf_intervals,
    }
    return pd.DataFrame({**index_data, **unigram_data})


@torch.inference_mode()
def bigram_model_worker(
    gpu_id: str,
    steps: list[int],
    model_name: str,
    pile_path: str,
    num_samples: int,
    d_vocab: int,
) -> pd.DataFrame:
    batch = 4
    torch.cuda.set_device(gpu_id)
    bigram_model = BigramModel("/mnt/ssd-1/lucia/pythia-deduped-bigrams.pkl", batch)

    mean_bigram_losses = []
    bigram_bottom_conf_intervals = []
    bigram_top_conf_intervals = []
    step_data = []
    token_indices = []
    for step in tqdm.tqdm(steps, position=gpu_id):
        model = GPTNeoXForCausalLM.from_pretrained(
            model_name, revision=f"step{step}", torch_dtype="auto"
        ).cuda()

        step_bigram_losses = []
        for _ in range(num_samples // batch):
            bigram_sample = next(bigram_model)
            eod_indices = [
                torch.where(bigram_sample[i] == 0)[0] for i in range(len(bigram_sample))
            ]
            bigram_outputs = model(bigram_sample)
            bigram_loss = F.cross_entropy(
                bigram_outputs.logits[:, :-1].reshape(batch * 2048, -1),
                bigram_sample[:, 1:].reshape(batch * 2048),
                reduction="none",
            ).reshape(batch, 2048)
            step_bigram_losses.extend(split_by_eod(bigram_loss, eod_indices))

        mean_step_bigram_loss, bigram_conf_intervals = summary_stats(
            step_bigram_losses, target_length=2048
        )
        mean_bigram_losses.extend(mean_step_bigram_loss)
        bigram_bottom_conf_intervals.extend(bigram_conf_intervals[0])
        bigram_top_conf_intervals.extend(bigram_conf_intervals[1])

        step_data.extend([step] * 2048)
        token_indices.extend(list(range(2048)))

    index_data = {"step": step_data, "index": token_indices}
    bigram_data = {
        "mean_bigram_losses": mean_bigram_losses,
        "bigram_bottom_conf_intervals": bigram_bottom_conf_intervals,
        "bigram_top_conf_intervals": bigram_top_conf_intervals,
    }
    return pd.DataFrame({**index_data, **bigram_data})


@torch.inference_mode()
def worker(
    gpu_id: str,
    steps: list[int],
    model_name: str,
    pile_path: str,
    num_samples: int,
    d_vocab: int,
) -> pd.DataFrame:
    batch = 1

    with open("/mnt/ssd-1/lucia/tmp_pythia-deduped-bigrams.pkl", "rb") as f:
        normalized_bigrams = pickle.load(f)
    normalized_bigrams = torch.tensor(normalized_bigrams)
    with open("/mnt/ssd-1/lucia/pythia-deduped-bigrams.pkl", "rb") as f:
        bigrams = pickle.load(f)
    unigrams = torch.tensor(bigrams.sum(axis=1)).float()

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

        step_metrics = []
        for _ in range(num_samples // batch):
            sample = next(pile)
            eod_indices = [torch.where(sample[i] == 0)[0] for i in range(len(sample))]
            outputs = model(sample)
            metrics, labels = get_metrics(
                sample, outputs.logits, normalized_bigrams, unigrams, batch, d_vocab
            )
            step_metrics.extend(split_by_eod(metrics, eod_indices))

        mean_step_divs, conf_intervals = matrix_summary_stats(
            step_metrics, target_length=2048
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


def write_normalized_bigram_frequencies(path: str):
    with open(path, "r+b") as f:
        bigrams = pickle.load(f)

    bigrams = bigrams.toarray().astype(np.float64)
    row_sums = bigrams.sum(axis=1)
    zero_rows = row_sums == 0

    uniform_value = 1 / bigrams.shape[0]
    bigrams = np.where(
        zero_rows[:, np.newaxis], uniform_value, bigrams / row_sums[:, np.newaxis]
    )

    with open("/mnt/ssd-1/lucia/tmp_pythia-deduped-bigrams.pkl", "wb") as f:
        pickle.dump(bigrams, f)


def main():
    model_name = "EleutherAI/pythia-410m"
    path = "/mnt/ssd-1/pile_preshuffled/standard/document.bin"
    num_samples = 16
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    d_vocab = len(tokenizer.vocab)  # 50277
    write_normalized_bigram_frequencies("/mnt/ssd-1/lucia/pythia-deduped-bigrams.pkl")

    log_steps = [0] + [2**i for i in range(int(math.log2(512)) + 1)]
    linear_steps = [i for i in range(1000, 144000, 1000)]
    steps = log_steps + linear_steps

    num_gpus = torch.cuda.device_count()
    max_steps_per_chunk = math.ceil(len(steps) / num_gpus)
    step_indices = [
        steps[i : i + max_steps_per_chunk]
        for i in range(0, len(steps), max_steps_per_chunk)
    ]

    print(f"Parallelising over {num_gpus} GPUs...")
    mp.set_start_method("spawn")
    with mp.Pool(num_gpus) as pool:
        args = [
            (i, step_indices[i], model_name, path, num_samples, d_vocab)
            for i in range(num_gpus)
        ]
        dfs = pool.starmap(bigram_model_worker, args)

    output_path = Path.cwd() / "output" / "checkpoint_bigrams_model.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(pd.concat(dfs), f)


if __name__ == "__main__":
    main()
