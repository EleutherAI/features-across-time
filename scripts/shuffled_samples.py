import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import tqdm.auto as tqdm
from scipy import stats
from transformers import GPTNeoXForCausalLM


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


def get_bow_sample(tokens: torch.Tensor, eod_indices: list[torch.Tensor]):
    """Shuffle tokens within documents. The end of document token is 0."""
    sample = tokens.clone()
    for i in range(len(sample)):
        start_idx = 0
        for idx in eod_indices[i]:
            if idx > start_idx:
                perm = torch.randperm(int(idx - start_idx)).cuda()
                sample[i, start_idx:idx] = sample[i, start_idx:idx][perm]
            start_idx = idx + 1
        if start_idx < len(sample[i]):
            perm = torch.randperm(int(len(sample[i]) - start_idx)).cuda()
            sample[i, start_idx:] = sample[i, start_idx:][perm]
    return sample


def split_by_eod(loss: torch.Tensor, eod_indices: [torch.Tensor]) -> list[list]:
    result = []
    start_idx = 0
    for i in range(loss.shape[0]):
        for zero_idx in eod_indices[i]:
            if zero_idx > start_idx:
                result.append(loss[i, start_idx : zero_idx + 1].cpu().numpy())
            start_idx = zero_idx + 1
        if start_idx < len(loss):
            result.append(loss[i, start_idx:].cpu().numpy())
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


@torch.inference_mode()
def worker(
    gpu_id: str, steps: list[int], model_name: str, pile_path: str, num_samples: int
):
    batch = 2

    torch.cuda.set_device(gpu_id)
    step_data = []
    token_indices = []
    losses = []
    bow_losses = []
    bottom_conf_intervals, top_conf_intervals = [], []
    bow_bottom_conf_intervals, bow_top_conf_intervals = [], []
    for step in tqdm.tqdm(steps, position=gpu_id):
        pile = Pile(pile_path, batch)
        model = GPTNeoXForCausalLM.from_pretrained(
            model_name, revision=f"step{step}", torch_dtype="auto"
        ).cuda()

        step_bow_losses = []
        step_losses = []
        for _ in range(num_samples // batch):
            sample = next(pile)
            eod_indices = [torch.where(sample[i] == 0)[0] for i in range(len(sample))]
            outputs = model(sample)
            loss = torch.nn.functional.cross_entropy(
                outputs.logits[:, :-1].reshape(batch * 2048, -1),
                sample[:, 1:].reshape(batch * 2048),
                reduction="none",
            ).reshape(batch, 2048)
            step_losses.extend(split_by_eod(loss, eod_indices))

            bow_sample = get_bow_sample(sample, eod_indices)
            bow_outputs = model(bow_sample)
            bow_loss = torch.nn.functional.cross_entropy(
                bow_outputs.logits[:, :-1].reshape(batch * 2048, -1),
                bow_sample[:, 1:].reshape(batch * 2048),
                reduction="none",
            ).reshape(batch, 2048)

            eod_indices = [
                torch.where(sample[i, :-1] == 0)[0] for i in range(len(sample))
            ]
            step_bow_losses.extend(split_by_eod(bow_loss, eod_indices))

        mean_bow_sequence_loss, bow_conf_intervals = summary_stats(step_bow_losses)
        bow_losses.extend(mean_bow_sequence_loss)
        bow_bottom_conf_intervals.extend(bow_conf_intervals[0])
        bow_top_conf_intervals.extend(bow_conf_intervals[1])

        mean_sequence_loss, conf_intervals = summary_stats(step_losses)
        losses.extend(mean_sequence_loss)
        bottom_conf_intervals.extend(conf_intervals[0])
        top_conf_intervals.extend(conf_intervals[1])

        step_data.extend([step] * len(mean_bow_sequence_loss))
        token_indices.extend(list(range(2048)))

    return pd.DataFrame(
        {
            "step": step_data,
            "index": token_indices,
            "mean_losses": losses,
            "bottom_conf_intervals": bottom_conf_intervals,
            "top_conf_intervals": top_conf_intervals,
            "token_bow_mean_losses": bow_losses,
            "token_bottom_conf_intervals": bow_bottom_conf_intervals,
            "token_top_conf_intervals": bow_top_conf_intervals,
        }
    )


def main():
    model_name = "EleutherAI/pythia-410m"
    pile_path = "/mnt/ssd-1/pile_preshuffled/standard/document.bin"
    num_samples = 100

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
            (i, step_indices[i], model_name, pile_path, num_samples)
            for i in range(num_gpus)
        ]
        dfs = pool.starmap(worker, args)

    output_path = Path.cwd() / "output" / "step_shuffled.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(pd.concat(dfs), f)


if __name__ == "__main__":
    main()
