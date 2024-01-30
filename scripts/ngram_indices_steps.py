import argparse
import math
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import tqdm.auto as tqdm
from ngram_steps import NgramModel
from scipy import stats
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


def encode(input: list[str], encoder: PreTrainedTokenizer, seq_len: int):
    result = []
    for row in input:
        encoded_tokens = torch.tensor(encoder.encode(row), device="cuda")[:seq_len]
        result.append(encoded_tokens)
        assert (
            len(result[-1]) == seq_len
        ), "Encoded sequence length too short; increase input length"
    return torch.stack(result)


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


def positional_summary_stats(
    vectors: list[np.ndarray], target_length: int
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
    model: PreTrainedModel,
    sample: torch.Tensor,
    batch: int,
    seq_len: int,
    eod_token_index: int,
) -> list[np.ndarray]:
    """Get sequence losses. Start a new sequence at each EOD token."""
    outputs = model(sample)
    eod_indices = [
        torch.where(sample[i] == eod_token_index)[0] for i in range(len(sample))
    ]
    loss = F.cross_entropy(
        outputs.logits[:, :-1].reshape(batch * (seq_len - 1), -1),
        sample[:, 1:].reshape(batch * (seq_len - 1)),
        reduction="none",
    ).reshape(batch, seq_len - 1)
    return split_by_eod(loss, eod_indices)


@torch.inference_mode()
def multi_step_worker(
    gpu_id: int,
    model_name: str,
    team: str,
    ngram_path: str,
    pile_path: str,
    tmp_cache_path: str,
    num_samples: int,
    batch: int,
    seq_len: int,
    d_vocab: int,
    eod_index: int,
    steps: list[str],
    tokenizer: PreTrainedTokenizer,
) -> pd.DataFrame:
    hf_model_name = f"{team}/{model_name}"

    tmp_cache_dir = f"{tmp_cache_path}/{gpu_id}"
    shutil.rmtree(tmp_cache_dir, ignore_errors=True)
    os.makedirs(tmp_cache_dir, exist_ok=True)

    ngram_model = NgramModel(ngram_path, batch=batch, seq_len=seq_len)
    use_encode = not (
        isinstance(tokenizer, AutoTokenizer)
        and NgramModel.tokenizer.name_or_path == tokenizer.name_or_path
    )
    if not use_encode:
        del tokenizer

    token_indices = []
    step_indices = []
    labels = ["unigram_loss", "bigram_loss", "random_loss"]
    means = {label: [] for label in labels}
    bottom_conf_intervals = {label: [] for label in labels}
    top_conf_intervals = {label: [] for label in labels}

    num_iters = math.ceil(num_samples / batch)
    pbar = tqdm.tqdm(total=len(steps) * num_iters, position=gpu_id)
    for step in steps:
        model = LlamaForCausalLM.from_pretrained(
            hf_model_name,
            revision=f"ckpt_{step}",
            torch_dtype="auto",
            cache_dir=tmp_cache_dir,
        ).cuda()

        step_unigram_losses = []
        step_bigram_losses = []
        step_random_losses = []

        for _ in range(num_iters):
            step_unigram_sample = (
                encode(ngram_model.generate_unigram_strs(), tokenizer, seq_len)
                if use_encode
                else ngram_model.generate_unigrams()
            )
            step_unigram_losses.extend(
                get_sequence_losses(
                    model, step_unigram_sample, batch, seq_len, eod_index
                )
            )
            step_bigram_sample = (
                encode(ngram_model.generate_bigram_strs(), tokenizer, seq_len)
                if use_encode
                else ngram_model.generate_bigrams()
            )
            step_bigram_losses.extend(
                get_sequence_losses(
                    model, step_bigram_sample, batch, seq_len, eod_index
                )
            )
            step_random_sample = torch.randint(
                0, d_vocab, [batch, seq_len], device="cuda"
            )
            step_random_losses.extend(
                get_sequence_losses(
                    model, step_random_sample, batch, seq_len, eod_index
                )
            )
            pbar.update(1)

        token_indices.extend(list(range(seq_len - 1)))
        step_indices.extend([int(step)] * (seq_len - 1))
        mean_unigram_loss, unigram_conf_intervals = positional_summary_stats(
            step_unigram_losses, (seq_len - 1)
        )
        means["unigram_loss"].extend(mean_unigram_loss)
        bottom_conf_intervals["unigram_loss"].extend(unigram_conf_intervals[0])
        top_conf_intervals["unigram_loss"].extend(unigram_conf_intervals[1])

        mean_bigram_loss, bigram_conf_intervals = positional_summary_stats(
            step_bigram_losses, (seq_len - 1)
        )
        means["bigram_loss"].extend(mean_bigram_loss)
        bottom_conf_intervals["bigram_loss"].extend(bigram_conf_intervals[0])
        top_conf_intervals["bigram_loss"].extend(bigram_conf_intervals[1])

        mean_random_loss, random_conf_intervals = positional_summary_stats(
            step_random_losses, (seq_len - 1)
        )
        means["random_loss"].extend(mean_random_loss)
        bottom_conf_intervals["random_loss"].extend(random_conf_intervals[0])
        top_conf_intervals["random_loss"].extend(random_conf_intervals[1])

        shutil.rmtree(tmp_cache_dir, ignore_errors=True)

    index_data = {"index": token_indices}
    mean_data = {f"mean_{label}": means[label] for label in labels}
    bottom_conf_data = {
        f"bottom_conf_{label}": bottom_conf_intervals[label] for label in labels
    }
    top_conf_data = {f"top_conf_{label}": top_conf_intervals[label] for label in labels}
    return pd.DataFrame(
        {**index_data, **mean_data, **bottom_conf_data, **top_conf_data}
    )


def main(ngram_path: str, pile_path: str, tmp_cache_path: str):
    team = "LLM360"
    model_batch_sizes = {"Amber": 1}
    tokenizer = LlamaTokenizer.from_pretrained("LLM360/Amber")

    vocab_size = tokenizer.vocab_size
    eod_index = tokenizer.eos_token_id
    num_samples = 1024
    batch = 1
    seq_len = 2048
    # Amber steps go from 0 to 359. Assuming linearly spaced (not specified)
    steps = ["000"] + [f"{2**i:03}" for i in range(int(math.log2(359)) + 1)] + ["358"]
    print(steps)
    num_gpus = torch.cuda.device_count()

    max_steps_per_chunk = math.ceil(len(steps) / num_gpus)
    step_indices = [
        steps[i : i + max_steps_per_chunk]
        for i in range(0, len(steps), max_steps_per_chunk)
    ]

    for model_name, batch in model_batch_sizes.items():
        args = [
            (
                i,
                model_name,
                team,
                ngram_path,
                pile_path,
                tmp_cache_path,
                num_samples,
                batch,
                seq_len,
                vocab_size,
                eod_index,
                step_indices[i],
                tokenizer,
            )
            for i in range(len(step_indices))
        ]
        print(f"Parallelising over {len(step_indices)} GPUs...")
        with mp.Pool(len(step_indices)) as pool:
            dfs = pool.starmap(multi_step_worker, args)

    df = pd.concat(dfs)
    df.to_csv(
        Path.cwd() / "output" / f"{model_name}_{num_samples}_steps.csv",
        index=False,
    )


if __name__ == "__main__":
    mp.set_start_method("spawn")

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--ngram_path",
        default="pythia-deduped-bigrams.pkl",
        help="Path to pickled sparse scipy array of bigram counts over the Pile",
    )
    parser.add_argument(
        "--pile_path",
        default="val_tokenized.hf",
        help="Path to Pile validation data",
    )
    parser.add_argument(
        "--tmp_cache_path",
        default=".cache",
        help="Path to cache (repeatedly cleared to free disk space)",
    )

    args = parser.parse_args()
    main(args.ngram_path, args.pile_path, args.tmp_cache_path)
