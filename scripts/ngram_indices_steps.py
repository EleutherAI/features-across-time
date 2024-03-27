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
from scipy import stats
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
)
from scriptutils.ngram_model import NgramModel
from scriptutils.load_model import get_neo_tokenizer, get_black_mamba, get_hails_mamba, get_zyphra_mamba
from scriptutils.experiment import Experiment, run_experiment_workers


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
    d_vocab=50277,
) -> list[np.ndarray]:
    """Get sequence losses. Start a new sequence at each EOD token."""
    outputs = model(sample)
    eod_indices = [
        torch.where(sample[i].eq(eod_token_index))[0] for i in range(len(sample))
    ]
    loss = F.cross_entropy(
        outputs.logits[:, :-1, :d_vocab].reshape(batch * (seq_len - 1), -1),
        sample[:, 1:].reshape(batch * (seq_len - 1)),
        reduction="none",
    ).reshape(batch, seq_len - 1)
    return split_by_eod(loss, eod_indices)


@torch.inference_mode()
def multi_step_worker(
    gpu_id: int,
    experiment: Experiment,
    ngram_path: str,
    pile_path: str,
    tmp_cache_path: str,
    steps: list[str]
) -> pd.DataFrame:
    torch.cuda.set_device(gpu_id)

    tokenizer = experiment.get_tokenizer()
    tmp_cache_dir = f"{tmp_cache_path}/{gpu_id}"
    shutil.rmtree(tmp_cache_dir, ignore_errors=True)
    os.makedirs(tmp_cache_dir, exist_ok=True)
    ngram_model = NgramModel(
        ngram_path, 
        batch=experiment.batch_size, 
        seq_len=experiment.seq_len, 
        tokenizer=tokenizer
    )
    # use_encode = not (
    #     isinstance(tokenizer, AutoTokenizer)
    #     and NgramModel.tokenizer.name_or_path == tokenizer.name_or_path
    # )
    use_encode = False
    if not use_encode:
        del tokenizer

    token_indices = []
    step_indices = []
    labels = ["unigram_loss", "bigram_loss"]  #  "random_loss"
    means = {label: [] for label in labels}
    bottom_conf_intervals = {label: [] for label in labels}
    top_conf_intervals = {label: [] for label in labels}

    num_iters = math.ceil(experiment.num_samples / experiment.batch_size)
    pbar = tqdm.tqdm(total=len(steps) * num_iters, position=gpu_id)
    for step in steps:
        model = experiment.get_model(experiment.team, experiment.model_name, step, tmp_cache_dir)

        step_unigram_losses = []
        step_bigram_losses = []
        # step_random_losses = []

        for i in range(num_iters):
            step_unigram_sample = (
                encode(ngram_model.generate_unigram_strs(), tokenizer, experiment.seq_len)
                if use_encode
                else ngram_model.generate_unigrams()
            )
            step_unigram_losses.extend(
                get_sequence_losses(
                    model, step_unigram_sample, experiment.batch_size, experiment.seq_len, experiment.eod_index, experiment.d_vocab
                )
            )
            step_bigram_sample = (
                encode(ngram_model.generate_bigram_strs(i), tokenizer, experiment.seq_len)
                if use_encode
                else ngram_model.generate_bigrams(i)
            )
            step_bigram_losses.extend(
                get_sequence_losses(
                    model, step_bigram_sample, experiment.batch_size, experiment.seq_len, experiment.eod_index, experiment.d_vocab
                )
            )
            # step_random_sample = torch.randint(
            #     0, experiment.d_vocab, [experiment.batch_size, experiment.seq_len], device="cuda"
            # )
            # step_random_losses.extend(
            #     get_sequence_losses(
            #         model, step_random_sample, experiment.batch_size, experiment.seq_len, experiment.eod_index
            #     )
            # )
            pbar.update(1)

        token_indices.extend(list(range(experiment.seq_len - 1)))
        step_indices.extend([int(step)] * (experiment.seq_len - 1))
        mean_unigram_loss, unigram_conf_intervals = positional_summary_stats(
            step_unigram_losses, (experiment.seq_len - 1)
        )
        means["unigram_loss"].extend(mean_unigram_loss)
        bottom_conf_intervals["unigram_loss"].extend(unigram_conf_intervals[0])
        top_conf_intervals["unigram_loss"].extend(unigram_conf_intervals[1])

        mean_bigram_loss, bigram_conf_intervals = positional_summary_stats(
            step_bigram_losses, (experiment.seq_len - 1)
        )
        means["bigram_loss"].extend(mean_bigram_loss)
        bottom_conf_intervals["bigram_loss"].extend(bigram_conf_intervals[0])
        top_conf_intervals["bigram_loss"].extend(bigram_conf_intervals[1])

        # mean_random_loss, random_conf_intervals = positional_summary_stats(
        #     step_random_losses, (seq_len - 1)
        # )
        # means["random_loss"].extend(mean_random_loss)
        # bottom_conf_intervals["random_loss"].extend(random_conf_intervals[0])
        # top_conf_intervals["random_loss"].extend(random_conf_intervals[1])

        shutil.rmtree(tmp_cache_dir, ignore_errors=True)

    index_data = {
        "index": token_indices, 
        "step": step_indices
    }
    mean_data = {f"mean_{label}": means[label] for label in labels}
    bottom_conf_data = {
        f"bottom_conf_{label}": bottom_conf_intervals[label] for label in labels
    }
    top_conf_data = {f"top_conf_{label}": top_conf_intervals[label] for label in labels}
    df = pd.DataFrame(
        {**index_data, **mean_data, **bottom_conf_data, **top_conf_data}
    )
    df.to_csv(
        Path.cwd()
        / "output"
        / f"{experiment.model_name}_{experiment.num_samples}_steps_{gpu_id}_{steps}.csv",
        index=False,
    )
    
    return df

# experiment = Experiment(
#     num_samples=1024,
#     team="hails", 
#     model_name="mamba-160m-hf", 
#     batch_size=1,
#     seq_len=2049, 
#     steps=[0, 1, 2, 4, 8, 16, 256, 1000, 8000, 33_000, 143_000], # done: 66_000, 131_000, 
#     d_vocab=50_277, # len(get_neo_tokenizer().vocab) 
#     get_model=get_hails_mamba, 
#     get_tokenizer=get_neo_tokenizer,
#     eod_index=0 # tokenizer.eos_token_id
# )
def main(ngram_path: str, pile_path: str, tmp_cache_path: str):
    experiment = Experiment(
        num_samples=1024,
        batch_size=2, 
        seq_len=2049, 
        team="Zyphra", 
        model_name="Mamba-370M", 
        get_model=get_zyphra_mamba, 
        get_tokenizer=get_neo_tokenizer,
        d_vocab=50_277, # len(get_neo_tokenizer().vocab) 
        # roughly log spaced steps + final step
        steps=[2**i for i in range(int(math.log2(2048)) + 1)] + [10_000, 20_000, 40_000, 80_000, 160_000, 320_000, 610_000],
        eod_index=get_neo_tokenizer().eos_token_id
    )

    df = run_experiment_workers(experiment, multi_step_worker, ngram_path, pile_path, tmp_cache_path)
    df.to_csv(
        Path.cwd()
        / "output"
        / f"{experiment.model_name}_{experiment.num_samples}_steps.csv",
        index=False,
    )


if __name__ == "__main__":
    mp.set_start_method("spawn")

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--ngram_path",
        default="/mnt/ssd-1/lucia/pythia-deduped-bigrams.pkl",
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
