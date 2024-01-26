import math
import pickle
from pathlib import Path
import argparse
import os


import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import tqdm.auto as tqdm
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.bettertransformer import BetterTransformer

class MistralNgramModel:
    def __init__(self, path: str, batch=1, seq_len=2049):
        self.decode_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        self.encode_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        self.d_vocab = len(self.decode_tokenizer.vocab)
        self.batch = batch
        self.seq_len = seq_len

        with open(path, "rb") as f:
            bigram_counts = pickle.load(f)

        self.unigrams = (
            torch.tensor(bigram_counts.toarray().astype(np.float32)).sum(dim=1).cuda()
        )  # small
        self.bigrams = torch.sparse_csr_tensor(
            bigram_counts.indptr.astype(np.int64),
            bigram_counts.indices.astype(np.int64),
            bigram_counts.data.astype(np.float32),
            dtype=torch.float32,
            device="cuda",
        )  # 0.7 GB

    def generate_unigrams(self) -> torch.Tensor:
        tokens = torch.multinomial(
            self.unigrams, self.batch * self.seq_len, replacement=True
        ).reshape(self.batch, self.seq_len)

        mistral_result = []
        for i in range(self.batch):
            token_strs = self.decode_tokenizer.decode(tokens[i].tolist())
            tokens = torch.tensor(self.encode_tokenizer.encode(token_strs), device='cuda')
            mistral_result.append(tokens[:2049])
            if len(mistral_result[-1]) < 2049:
                print("short tensor unigrams! make sample longer evey time")
        return torch.stack(mistral_result)


    # separate slice sparse array function with test
    def sample_bigram(self, prev: torch.Tensor) -> torch.Tensor:
        """Given a batch of previous tokens, sample from a bigram model
        using conditional distributions stored in a sparse CSR tensor."""
        starts = self.bigrams.crow_indices()[prev]
        ends = self.bigrams.crow_indices()[prev + 1]

        # 0 padding to batch rows with variable numbers of non-zero elements
        token_probs = torch.zeros(
            (self.batch, self.d_vocab), dtype=torch.float32, device="cuda"
        )
        token_col_indices = torch.zeros(
            (self.batch, self.d_vocab), dtype=torch.int32, device="cuda"
        )
        for i in range(self.batch):
            token_probs[i, : ends[i] - starts[i]] = self.bigrams.values()[
                starts[i] : ends[i]
            ]
            token_col_indices[i, : ends[i] - starts[i]] = self.bigrams.col_indices()[
                starts[i] : ends[i]
            ]

        sampled_value_indices = torch.multinomial(token_probs, 1)
        return torch.gather(token_col_indices, 1, sampled_value_indices)

    def generate_bigrams(self) -> torch.Tensor:
        """Auto-regressively generate bigram model sequence. Initialize each
        sequence by sampling from a unigram model."""
        result = [
            torch.multinomial(self.unigrams, self.batch, replacement=True).unsqueeze(1)
        ]
        for _ in range(self.seq_len - 1):
            prev = result[-1]
            result.append(self.sample_bigram(prev))

        result = torch.cat(result, dim=-1)
        mistral_result = []
        for i in range(self.batch):
            token_strs = self.decode_tokenizer.decode(result[i].tolist())
            tokens = torch.tensor(self.encode_tokenizer.encode(token_strs), device='cuda')
            mistral_result.append(tokens[:2049])
            if len(mistral_result[-1]) < 2049:
                print("short tensor! make sample longer evey time")
        return torch.stack(mistral_result)


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
    model: AutoModelForCausalLM, sample: torch.Tensor, batch: int
) -> list[np.ndarray]:
    """Get sequence losses. Start a new sequence at each EOD token."""
    eod_indices = [torch.where(sample[i] == 2)[0] for i in range(len(sample))]
    outputs = model(sample)
    loss = F.cross_entropy(
        outputs.logits[:, :-1].reshape(batch * 2048, -1),
        sample[:, 1:].reshape(batch * 2048),
        reduction="none",
    ).reshape(batch, 2048)
    return split_by_eod(loss, eod_indices)


@torch.inference_mode()
def ngram_model_worker(
    model_path: str,
    num_samples: int,
    batch: int,
) -> pd.DataFrame:
    tmp_cache_dir = Path(".cache")
    os.makedirs(tmp_cache_dir, exist_ok=True)
    ngram_model = MistralNgramModel(model_path, batch=batch)

    labels = ["unigram_loss", "bigram_loss"]
    means = {label: [] for label in labels}
    bottom_conf_intervals = {label: [] for label in labels}
    top_conf_intervals = {label: [] for label in labels}
    step_data = []
    token_indices = []
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-v0.1",
        torch_dtype="auto",
        cache_dir=tmp_cache_dir,
    ).cuda()
    step_unigram_losses = []
    step_bigram_losses = []
    for _ in tqdm.tqdm(range(num_samples // batch)):
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


def main(ngram_path: str):
    num_samples = 1024
    batch = 1

    df = ngram_model_worker(
            ngram_path,
            num_samples,
            batch)
    df.to_csv(
        Path.cwd()
        / "output"
        / f"means_ngrams_model_{'mistral-7b'}_{num_samples}.csv",
        index=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--ngram_path",
        default="pythia-deduped-bigrams.pkl", # /mnt/ssd-1/lucia/
        help="Path to pickled sparse scipy array of bigram counts over the Pile",
    )
    args = parser.parse_args()
    main(args.ngram_path)
