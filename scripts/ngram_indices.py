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


class NgramModel:
    def __init__(self, path: str, encode_tokenizer=None, batch=1, seq_len=2048):
        self.decode_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        self.encode_tokenizer = AutoTokenizer.from_pretrained(encode_tokenizer)
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
        return self.transcode(tokens)
    

    def generate_random(self) -> torch.Tensor:
        return torch.randint(0, len(self.encode_tokenizer.vocab), [self.batch, self.seq_len], device='cuda')


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
        return self.transcode(result)


    def transcode(self, tokens: torch.Tensor):
        encoded_result = []
        for i in range(len(tokens)):
            token_strs = self.decode_tokenizer.decode(tokens[i].tolist())
            encoded_tokens = torch.tensor(self.encode_tokenizer.encode(token_strs), device='cuda')
            encoded_result.append(encoded_tokens[:self.seq_len])
            assert len(encoded_result[-1]) >= self.seq_len, "Transcoded tokens too short; increase seq_length"
        return torch.stack(encoded_result)


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
    model: AutoModelForCausalLM, sample: torch.Tensor, batch: int, seq_len=2048
) -> list[np.ndarray]:
    """Get sequence losses. Start a new sequence at each EOD token."""
    eod_indices = [torch.where(sample[i] == 2)[0] for i in range(len(sample))]
    outputs = model(sample)
    loss = F.cross_entropy(
        outputs.logits[:, :-1].reshape(batch * seq_len, -1),
        sample[:, 1:].reshape(batch * seq_len),
        reduction="none",
    ).reshape(batch, seq_len)
    return split_by_eod(loss, eod_indices)


@torch.inference_mode()
def ngram_model_worker(
    model_name: str,
    team: str,
    model_path: str,
    num_samples: int,
    batch: int,
    seq_len: int,
) -> pd.DataFrame:
    hf_model_name = f'{team}/{model_name}',
    tmp_cache_dir = Path(".cache")
    os.makedirs(tmp_cache_dir, exist_ok=True)
    ngram_model = NgramModel(model_path, encode_tokenizer=hf_model_name, batch=batch, seq_len=seq_len + 1)
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_name,
        torch_dtype="auto",
        cache_dir=tmp_cache_dir,
    ).cuda()

    token_indices = []
    labels = ["unigram_loss", "bigram_loss", "random_loss"]
    means = {label: [] for label in labels}
    bottom_conf_intervals = {label: [] for label in labels}
    top_conf_intervals = {label: [] for label in labels}

    unigram_losses = []
    bigram_losses = []
    random_losses = []
    for _ in tqdm.tqdm(range(num_samples // batch)):
        bigram_sample = ngram_model.generate_bigrams()
        bigram_losses.extend(get_sequence_losses(model, bigram_sample, batch, seq_len))
        unigram_sample = ngram_model.generate_unigrams()
        unigram_losses.extend(
            get_sequence_losses(model, unigram_sample, batch, seq_len)
        )
        random_sample = ngram_model.generate_random()
        random_losses.extend(
            get_sequence_losses(model, random_sample, batch, seq_len)
        )

    mean_unigram_loss, unigram_conf_intervals = positional_summary_stats(
        unigram_losses, target_length=seq_len
    )
    means["unigram_loss"].extend(mean_unigram_loss)
    bottom_conf_intervals["unigram_loss"].extend(unigram_conf_intervals[0])
    top_conf_intervals["unigram_loss"].extend(unigram_conf_intervals[1])

    mean_bigram_loss, bigram_conf_intervals = positional_summary_stats(
        bigram_losses, target_length=seq_len
    )
    means["bigram_loss"].extend(mean_bigram_loss)
    bottom_conf_intervals["bigram_loss"].extend(bigram_conf_intervals[0])
    top_conf_intervals["bigram_loss"].extend(bigram_conf_intervals[1])

    mean_random_loss, random_conf_intervals = positional_summary_stats(
        random_losses, target_length=seq_len
    )
    means["random_loss"].extend(mean_random_loss)
    bottom_conf_intervals["random_loss"].extend(random_conf_intervals[0])
    top_conf_intervals["random_loss"].extend(random_conf_intervals[1])

    token_indices.extend(list(range(seq_len)))

    index_data = {"index": token_indices}
    mean_data = {f"mean_{label}": means[label] for label in labels}
    bottom_conf_data = {
        f"bottom_conf_{label}": bottom_conf_intervals[label] for label in labels
    }
    top_conf_data = {f"top_conf_{label}": top_conf_intervals[label] for label in labels}
    return pd.DataFrame(
        {**index_data, **mean_data, **bottom_conf_data, **top_conf_data}
    )


def main(ngram_path: str):
    model_name = "Mistral-7B-v0.1"
    team = "mistralai"
    
    num_samples = 1024
    batch = 1
    seq_len = 2048 # (2048 * 4)

    df = ngram_model_worker(
            model_name,
            team,
            ngram_path,
            num_samples,
            batch,
            seq_len)
    df.to_csv(
        Path.cwd()
        / "output"
        / f"{model_name}_{num_samples}.csv",
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
