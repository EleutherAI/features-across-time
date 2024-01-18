import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import tqdm.auto as tqdm
from scipy import stats
from torch.nn import LogSoftmax
from torch.nn.functional import kl_div
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


def split_by_eod(data: torch.Tensor, eod_indices: [torch.Tensor]) -> list[np.ndarray]:
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
    vectors: list[np.ndarray], target_length=2049
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    padded_vectors = [
        np.pad(v, (0, target_length - len(v)), constant_values=np.nan) for v in vectors
    ]
    stacked_vectors = np.vstack(padded_vectors).astype(np.float64)
    means = np.nanmean(stacked_vectors, axis=0)
    standard_errors = stats.sem(stacked_vectors, axis=0, nan_policy="omit")
    conf_intervals = stats.norm.interval(0.95, loc=means, scale=standard_errors)

    return means, conf_intervals


def get_bigram_kl_divergences(
    tokens: torch.Tensor,
    logits: torch.Tensor,
    bigrams: np.ndarray,
    batch: int,
    d_vocab: int,
) -> np.ndarray:
    tokens = tokens.view(batch * 2049)
    logits = logits.view(batch * 2049, -1)[:, :d_vocab]

    bigram_dists = bigrams[tokens.cpu()].cuda() + torch.finfo(torch.float32).eps
    bigram_dists = bigram_dists
    log_softmax_logits = LogSoftmax(dim=1)(logits)

    kl_divergences = kl_div(log_softmax_logits, bigram_dists, reduction="none").sum(
        dim=1
    )

    return kl_divergences.view(batch, -1)


@torch.inference_mode()
def worker(
    gpu_id: str,
    steps: list[int],
    model_name: str,
    pile_path: str,
    num_samples: int,
    d_vocab: int,
):
    output_path = Path.cwd() / "output" / f"checkpoint_bigrams_{gpu_id}.pkl"
    batch = 1

    with open("tmp_pythia-deduped-bigrams.pkl", "rb") as f:
        bigrams = pickle.load(f)
        bigrams = torch.tensor(bigrams)

    torch.cuda.set_device(gpu_id)
    step_data = []
    token_indices = []
    kl_divergences = []
    bottom_conf_intervals, top_conf_intervals = [], []
    for step in tqdm.tqdm(steps, position=gpu_id):
        pile = Pile(pile_path, batch)
        model = GPTNeoXForCausalLM.from_pretrained(
            model_name, revision=f"step{step}", torch_dtype="auto"
        ).cuda()

        step_kl_divergences = []
        for _ in range(num_samples // batch):
            sample = next(pile)
            eod_indices = [torch.where(sample[i] == 0)[0] for i in range(len(sample))]
            outputs = model(sample)
            step_kl_divergence = get_bigram_kl_divergences(
                sample, outputs.logits, bigrams, batch, d_vocab
            ).reshape(batch, 2049)
            step_kl_divergences.extend(split_by_eod(step_kl_divergence, eod_indices))

        assert isinstance(step_kl_divergences[0], np.ndarray)
        mean_step_kl_divergence, conf_intervals = summary_stats(step_kl_divergences)
        kl_divergences.extend(mean_step_kl_divergence)
        bottom_conf_intervals.extend(conf_intervals[0])
        top_conf_intervals.extend(conf_intervals[1])
        step_data.extend([step] * len(mean_step_kl_divergence))
        token_indices.extend(
            list(range(2049))
        )  # can calculate divergence from bigram stats at every position

    df = pd.DataFrame(
        {
            "step": step_data,
            "index": token_indices,
            "mean_kl_divergence": kl_divergences,
            "bottom_conf_intervals": bottom_conf_intervals,
            "top_conf_intervals": top_conf_intervals,
        }
    )

    with open(output_path, "wb") as f:
        pickle.dump(df, f)


def main():
    model_name = "EleutherAI/pythia-410m"
    path = "/mnt/ssd-1/pile_preshuffled/standard/document.bin"
    num_samples = 30
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    d_vocab = len(tokenizer.vocab)
    logging.set_verbosity_error()
    # normalize bigram frequencies once at start
    # dctx = zstd.ZstdDecompressor()
    # with open('pythia-deduped-bigrams.zst', 'rb') as compressed_file, \
    #     open('tmp_pythia-deduped-bigrams.pkl', 'wb') as temp_file:
    #     dctx.copy_stream(compressed_file, temp_file)

    # with open('tmp_pythia-deduped-bigrams.pkl', 'r+b') as f:
    #     bigrams = pickle.load(f)
    #     bigrams = bigrams.toarray().astype(np.float64)

    #     row_sums = bigrams.sum(axis=1)
    #     zero_rows = row_sums == 0
    #     d_vocab = bigrams.shape[1]

    #     if np.any(~zero_rows):
    #         bigrams[~zero_rows] /= row_sums[~zero_rows][:, np.newaxis]

    #     if np.any(zero_rows):
    #         uniform_value = 1 / d_vocab
    #         bigrams[zero_rows] = uniform_value

    #     f.seek(0)
    #     pickle.dump(bigrams, f)
    #     f.truncate()

    log_steps = [0] + [2**i for i in range(int(math.log2(512)) + 1)]
    linear_steps = [i for i in range(1000, 144000, 1000)]
    steps = log_steps + linear_steps

    num_gpus = torch.cuda.device_count()
    max_steps_per_chunk = math.ceil(len(steps) / num_gpus)
    step_indices = [
        steps[i : i + max_steps_per_chunk]
        for i in range(0, len(steps), max_steps_per_chunk)
    ]
    # data = ()

    print(f"Parallelising over {num_gpus} GPUs...")
    mp.set_start_method("spawn")
    processes = []
    for i in range(num_gpus):
        p = mp.Process(
            target=worker,
            args=(i, step_indices[i], model_name, path, num_samples, d_vocab),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
