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
from transformers import GPTNeoXForCausalLM


class NgramModel:
    def __init__(self, path: str, batch=1, seq_len=2049):
        with open(path, "rb") as f:
            bigram_counts = pickle.load(f)

        bigram_counts = torch.tensor(bigram_counts.toarray(), dtype=torch.float32)
        unigrams = bigram_counts.sum(dim=1)
        row_sums = unigrams.unsqueeze(-1)

        self.bigrams = torch.where(
            (row_sums == 0), 1 / bigram_counts.shape[0], bigram_counts / row_sums
        )
        self.unigrams = unigrams.cuda()
        self.batch = batch
        self.seq_len = seq_len

    def generate_bigrams(self) -> torch.Tensor:
        result = [torch.multinomial(self.unigrams, self.batch)]
        for _ in range(self.seq_len - 1):
            token_dists = self.bigrams[result[-1].cpu()].cuda()
            result.append(torch.multinomial(token_dists, 1).squeeze())
        return torch.stack(result, dim=-1)

    def generate_unigrams(self) -> torch.Tensor:
        return torch.multinomial(self.unigrams, self.batch * self.seq_len).reshape(
            self.batch, self.seq_len
        )


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


def get_sequence_losses(
    model: GPTNeoXForCausalLM, sample: torch.Tensor, batch: int
) -> list[np.ndarray]:
    """Get sequence losses. Start a new sequence at each EOD token."""
    eod_indices = [torch.where(sample[i] == 0)[0] for i in range(len(sample))]
    outputs = model(sample)
    loss = F.cross_entropy(
        outputs.logits[:, :-1].reshape(batch * 2048, -1),
        sample[:, 1:].reshape(batch * 2048),
        reduction="none",
    ).reshape(batch, 2048)
    return split_by_eod(loss, eod_indices)


@torch.inference_mode()
def ngram_model_worker(
    gpu_id: str,
    steps: list[int],
    model_name: str,
    model_path: str,
    num_samples: int,
) -> pd.DataFrame:
    batch = 4
    torch.cuda.set_device(gpu_id)
    ngram_model = NgramModel(model_path, batch)

    labels = ["unigram_loss", "bigram_loss"]
    means = {label: [] for label in labels}
    bottom_conf_intervals = {label: [] for label in labels}
    top_conf_intervals = {label: [] for label in labels}
    step_data = []
    token_indices = []
    for step in tqdm.tqdm(steps, position=gpu_id):
        model = GPTNeoXForCausalLM.from_pretrained(
            model_name, revision=f"step{step}", torch_dtype="auto"
        ).cuda()

        step_unigram_losses = []
        step_bigram_losses = []
        for _ in range(num_samples // batch):
            bigram_sample = ngram_model.generate_bigrams()
            step_bigram_losses.extend(get_sequence_losses(model, bigram_sample, batch))

            unigram_sample = ngram_model.generate_unigrams()
            step_unigram_losses.extend(
                get_sequence_losses(model, unigram_sample, batch)
            )

        mean_unigram_step_loss, unigram_conf_intervals = summary_stats(
            step_unigram_losses, target_length=2048
        )
        means["unigram_loss"].extend(mean_unigram_step_loss)
        bottom_conf_intervals["unigram_loss"].extend(unigram_conf_intervals[0])
        top_conf_intervals["unigram_loss"].extend(unigram_conf_intervals[1])

        mean_bigram_step_loss, bigram_conf_intervals = summary_stats(
            step_bigram_losses, target_length=2048
        )
        means["bigram_loss"].extend(mean_bigram_step_loss)
        bottom_conf_intervals["bigram_loss"].extend(bigram_conf_intervals[0])
        top_conf_intervals["bigram_loss"].extend(bigram_conf_intervals[1])

        step_data.extend([step] * 2048)
        token_indices.extend(list(range(2048)))

    index_data = {"step": step_data, "index": token_indices}
    mean_data = {f"mean_{label}": means[label] for label in labels}
    bottom_conf_data = {
        f"bottom_conf_{label}": bottom_conf_intervals[label] for label in labels
    }
    top_conf_data = {f"top_conf_{label}": top_conf_intervals[label] for label in labels}
    return pd.DataFrame(
        {**index_data, **mean_data, **bottom_conf_data, **top_conf_data}
    )


def main():
    model_name = "EleutherAI/pythia-410m"
    ngrams_path = "/mnt/ssd-1/lucia/pythia-deduped-bigrams.pkl"
    num_samples = 32

    log_steps = [0] + [2**i for i in range(int(math.log2(512)) + 1)]
    linear_steps = list(range(1000, 144000, 1000))
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
            (i, step_indices[i], model_name, ngrams_path, num_samples)
            for i in range(num_gpus)
        ]
        dfs = pool.starmap(ngram_model_worker, args)

    output_path = Path.cwd() / "output" / "step_ngrams_model.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(pd.concat(dfs), f)


if __name__ == "__main__":
    main()
