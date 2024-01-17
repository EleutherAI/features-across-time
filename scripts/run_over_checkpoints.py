import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import tqdm.auto as tqdm
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


def get_bow_sample(tokens: torch.Tensor):
    sample = tokens.clone()
    for i in range(len(sample)):
        zero_indices = torch.nonzero(sample[i] == 0).cuda()
        if zero_indices.numel():
            zero_indices = zero_indices[0]

        start_idx = 0
        for idx in zero_indices:
            if idx > start_idx:
                perm = torch.randperm(int(idx - start_idx)).cuda()
                sample[i, start_idx:idx] = sample[i, start_idx:idx][perm]
            start_idx = idx + 1
        if start_idx < len(sample[i]):
            perm = torch.randperm(int(len(sample[i]) - start_idx)).cuda()
            sample[i, start_idx:] = sample[i, start_idx:][perm]
        return sample


def split_loss(loss: torch.Tensor, zero_indices: torch.Tensor) -> list[list]:
    result = []
    start_idx = 0
    for zero_idx in zero_indices:
        if zero_idx > start_idx:
            result.append(loss[start_idx : zero_idx + 1].tolist())
        start_idx = zero_idx + 1
    if start_idx < len(loss):
        result.append(loss[start_idx:].tolist())
    return result


# compile
@torch.inference_mode()
def worker(
    gpu_id: str, steps: list[int], model_name: str, pile_path: str, num_samples: int
):
    output_path = Path.cwd() / "output" / f"checkpoint_token_data_{gpu_id}.pkl"

    torch.cuda.set_device(gpu_id)
    step_data = []
    token_losses = []
    token_bow_losses = []
    for step in tqdm.tqdm(steps, position=gpu_id):
        pile = Pile(pile_path, batch=4)
        model = GPTNeoXForCausalLM.from_pretrained(
            model_name, revision=f"step{step}", torch_dtype="auto"
        ).cuda()

        for _ in range(num_samples):
            sample = next(pile)
            zero_indices = torch.nonzero(sample.flatten(0) == 0).cuda()
            outputs = model(sample)

            loss = torch.nn.functional.cross_entropy(
                outputs.logits[:, :-1].flatten(0, 1),
                sample[:, 1:].flatten(0, 1),
                reduction="none",
            )
            sequence_losses = split_loss(loss, zero_indices)
            token_losses.extend(sequence_losses)

            bow_sample = get_bow_sample(sample)
            bow_outputs = model(bow_sample)
            bow_loss = torch.nn.functional.cross_entropy(
                bow_outputs.logits[:, :-1].flatten(0, 1),
                bow_sample[:, 1:].flatten(0, 1),
                reduction="none",
            )
            bow_sequence_losses = split_loss(bow_loss, zero_indices)
            token_bow_losses.extend(bow_sequence_losses)
            step_data.extend([step] * len(bow_sequence_losses))

    df = pd.DataFrame(
        {
            "step": step_data,
            "token_loss": token_losses,
            "token_bow_losses": token_bow_losses,
        }
    )

    with open(output_path, "wb") as f:
        pickle.dump(df, f)


def main():
    model_name = "EleutherAI/pythia-160m-v0"
    path = "/mnt/ssd-1/pile_preshuffled/standard/document.bin"
    num_samples = 500

    # log_steps = [0] + [2**i for i in range(int(math.log2(512)) + 1)]
    linear_steps = [i for i in range(1000, 144000, 1000)]
    # steps = log_steps + linear_steps
    steps = linear_steps

    num_gpus = torch.cuda.device_count()
    max_steps_per_chunk = math.ceil(len(steps) / num_gpus)
    step_indices = [
        steps[i : i + max_steps_per_chunk]
        for i in range(0, len(steps), max_steps_per_chunk)
    ]

    print(f"Parallelising over {num_gpus} GPUs...")
    mp.set_start_method("spawn")
    processes = []
    for i in range(num_gpus):
        p = mp.Process(
            target=worker, args=(i, step_indices[i], model_name, path, num_samples)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
