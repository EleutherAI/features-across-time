import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from transformers import GPTNeoXForCausalLM


class Pile:
    def __init__(self, path: str):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.index = len(self.data)
        self.chunk_size = 2049

    def __iter__(self):
        return self

    def __next__(self):
        if self.index <= 0:
            raise StopIteration

        start_index = max(self.index - self.chunk_size, 0)
        sample = self.data[start_index : self.index]
        self.index = start_index

        return torch.from_numpy(sample.astype(np.int64))


def get_bow_sample(tokens: torch.tensor):
    sample = tokens.clone()
    zero_indices = torch.where(sample == 0)[0]
    start_idx = 0
    for idx in zero_indices:
        if idx > start_idx:
            perm = torch.randperm(idx - start_idx)
            sample[start_idx:idx] = sample[start_idx:idx][perm]
        start_idx = idx + 1
    if start_idx < len(sample):
        perm = torch.randperm(len(sample) - start_idx)
        sample[start_idx:] = sample[start_idx:][perm]
    return sample


def cross_entropy_loss(tokens, logits):
    log_probs = F.log_softmax(logits, dim=-1)
    predicted_log_probs = log_probs[..., :-1, :].gather(
        dim=-1, index=tokens[..., 1:, None]
    )[..., 0]
    return -predicted_log_probs


def split_loss(loss: torch.tensor, zero_indices: torch.tensor):
    result = []
    start_idx = 0
    for zero_idx in zero_indices:
        if zero_idx > start_idx:
            result.append(loss[start_idx : zero_idx + 1])
        start_idx = zero_idx + 1
    if start_idx < len(loss):
        result.append(loss[start_idx:])
    return result


def worker(
    gpu_idx: int, steps: list[int], model_name: str, pile_path: str, num_samples: int
):
    attention_mask = torch.ones((1, 2049), dtype=torch.int32)

    token_bow_losses = []

    for step in steps:
        pile = Pile(pile_path)  # currently same sample for all steps
        model = GPTNeoXForCausalLM.from_pretrained(model_name, revision=f"step{step}")

        for _ in range(num_samples):
            sample = next(pile)
            zero_indices = torch.nonzero(sample == 0)
            bow_sample = get_bow_sample(sample)
            bow_outputs = model(bow_sample.unsqueeze(0), attention_mask)
            bow_loss = cross_entropy_loss(bow_sample.unsqueeze(0), bow_outputs.logits)
            token_bow_losses.extend(split_loss(bow_loss), zero_indices)

    print(token_bow_losses)
    df = pd.DataFrame({"step": [], "bow_loss": []})
    df.to_csv(Path.cwd() / "output" / f"checkpoint_token_data_{gpu_idx}")


def main():
    model_name = "EleutherAI/pythia-160m"
    path = "/mnt/ssd-1/pile_preshuffled/standard/document.bin"
    num_samples = 30_000

    log_steps = [0] + [2**i for i in range(int(math.log2(512)) + 1)]
    linear_steps = [i for i in range(1000, 143000, 1000)]
    steps = log_steps + linear_steps

    num_gpus = torch.cuda.device_count()
    max_steps_per_chunk = math.ceil(len(steps) // num_gpus)
    step_indices = [
        steps[i : i + max_steps_per_chunk]
        for i in range(0, len(steps), max_steps_per_chunk)
    ]

    num_gpus = 1
    print(f"Parallelising over {num_gpus} GPUs...")
    for i in range(num_gpus):
        mp.spawn(
            worker, args=(step_indices[i], model_name, path, num_samples), nprocs=1
        )


def test():
    test = np.array([0, 1, 5, 3, 0, 5, 2, 0, 2])

    print(get_bow_sample(test))
    print(get_bow_sample(test))
    print(get_bow_sample(test))

    # zero_indices = torch.tensor([0, 3, 5])
    # print(split_loss(torch.tensor([0, 1, 3, 0, 1, 0])))


if __name__ == "__main__":
    main()
