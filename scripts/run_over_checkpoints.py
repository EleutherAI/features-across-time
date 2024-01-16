import math

import numpy as np
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
    return GPTNeoXForCausalLM.from_pretrained(model_name, revision=f"step{step}")


class Pile:
    def __init__(self, path: str):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.index = -1

    def __iter__(self):
        return self

    def __next__(self):
        sample = self.data[self.index - 2049 : self.index]
        self.index -= 2049
        return sample


def get_bow_sample(tokens: np.ndarray):
    sample = tokens.copy()
    zero_indices = np.where(sample == 0)[0]
    start_idx = 0
    for idx in zero_indices:
        if idx > start_idx:
            np.random.shuffle(sample[start_idx:idx])
        start_idx = idx + 1
    if start_idx < len(sample):
        np.random.shuffle(sample[start_idx:])
    return sample


def worker(
    gpu_idx: int, steps: list[int], model_name: str, pile_path: str, num_samples: int
):
    attention_mask = torch.ones((1, 2049), dtype=torch.int32)

    for step in steps:
        pile = Pile(pile_path)  # currently same sample for all steps
        model = get_model(model_name, step)

        token_losses = []
        token_bow_losses = []
        for _ in range(num_samples):
            sample = next(pile)
            zero_indices = np.where(sample == 0)[0]

            outputs = model(
                torch.from_numpy(sample.astype(np.int32)).unsqueeze(0), attention_mask
            )
            print(type(outputs))
            return

            def split_loss(loss: str):
                result = []
                start_idx = 0
                for zero_idx in zero_indices:
                    if zero_idx > start_idx:
                        result.append(loss[start_idx:zero_idx])
                    start_idx = zero_idx + 1
                if start_idx < len(loss):
                    result.append(loss[start_idx:])
                return result

            print(split_loss(outputs))
            print(split_loss(sample))
            return

            token_losses.append(outputs.loss)

            bow_sample = get_bow_sample(sample)
            bow_outputs = model(
                torch.from_numpy(bow_sample.astype(np.int32)).unsqueeze(0),
                attention_mask,
            )
            token_bow_losses.append(bow_outputs.loss)

    # df = pd.DataFrame({
    #     "step": [],
    #     "original_loss": [],
    #     "bow_loss": []
    # })
    # df.to_csv(f"checkpoint_aggregated_data_{gpu_idx}")
    # df = pd.DataFrame({
    #     "step": [],
    #     "original_loss": [],
    #     "bow_loss": []
    # })
    # df.to_csv(f"{output_dir}/checkpoint_token_data_{gpu_idx}")


def main():
    model_name = "EleutherAI/pythia-160m-deduped"
    path = "/mnt/ssd-1/pile_preshuffled/deduped/document.bin"
    num_samples = 30

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
        mp.spawn(
            worker, args=(step_indices[i], model_name, path, num_samples), nprocs=1
        )


def test():
    test = np.array([0, 1, 5, 3, 0, 5, 2, 0, 2])
    print(get_bow_sample(test))
    print(get_bow_sample(test))
    print(get_bow_sample(test))


if __name__ == "__main__":
    main()
