import math

import torch
import torch.multiprocessing as mp
from transformers import GPTNeoXForCausalLM


def get_device():
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def get_model(model_name: str, step: int) -> GPTNeoXForCausalLM:
    return GPTNeoXForCausalLM.from_pretrained(
        model_name,
        revision=f"step{step}",
        cache_dir=f"./pythia-160m-deduped/step{step}",
    )


def worker(gpu_idx: int, steps: list[int], model_name: str):
    for step in steps:
        get_model(model_name, step)


def main():
    model_name = "EleutherAI/pythia-160m-deduped"

    log_steps = [0] + [2**i for i in range(int(math.log2(512)) + 1)]
    linear_steps = [i for i in range(1000, 143000, 1000)]
    steps = log_steps + linear_steps

    num_gpus = torch.cuda.device_count()
    max_steps_per_chunk = math.ceil(len(steps) // num_gpus)
    step_indices = [
        steps[i : i + max_steps_per_chunk]
        for i in range(0, len(steps), max_steps_per_chunk)
    ]

    print(f"Parallelising over {num_gpus} GPUs...")
    for i in range(num_gpus):
        mp.spawn(worker, args=(step_indices[i], model_name), nprocs=1)


if __name__ == "__main__":
    main()
