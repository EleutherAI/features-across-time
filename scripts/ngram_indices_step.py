import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm.auto as tqdm
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

from scriptutils.experiment import Experiment
from scriptutils.load_model import get_auto_model, get_neo_tokenizer
from ngram_indices_steps import split_by_eod, positional_summary_stats, get_sequence_losses

class NgramModel:
    def __init__(self, path: str, encode_tokenizer, batch=1, seq_len=2048):
        self.decode_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        self.encode_tokenizer = encode_tokenizer
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
        return torch.randint(
            0,
            len(self.encode_tokenizer.vocab),
            [self.batch, self.seq_len],
            device="cuda",
        )

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
            encoded_tokens = torch.tensor(
                self.encode_tokenizer.encode(token_strs), device="cuda"
            )
            encoded_result.append(encoded_tokens[:self.seq_len])
            assert (
                len(encoded_result[-1]) >= self.seq_len
            ), "Transcoded tokens too short; increase seq_length"
        return torch.stack(encoded_result)


@torch.inference_mode()
def worker(
    experiment: Experiment,
    ngram_path: str,
) -> pd.DataFrame:
    tmp_cache_dir = Path(".cache")
    os.makedirs(tmp_cache_dir, exist_ok=True)
    ngram_model = NgramModel(
        ngram_path, encode_tokenizer=experiment.get_tokenizer(), batch=experiment.batch_size, seq_len=experiment.seq_len
    )
    model = experiment.get_model(experiment.team, experiment.model_name, 0, tmp_cache_dir)
    token_indices = []
    labels = ["unigram_loss", "bigram_loss", "random_loss"]
    means = {label: [] for label in labels}
    bottom_conf_intervals = {label: [] for label in labels}
    top_conf_intervals = {label: [] for label in labels}

    unigram_losses = []
    bigram_losses = []
    random_losses = []
    for _ in tqdm.tqdm(range(experiment.num_samples // experiment.batch_size)):
        bigram_sample = ngram_model.generate_bigrams()
        bigram_losses.extend(get_sequence_losses(model, bigram_sample, experiment.batch_size, experiment.seq_len, experiment.eod_index))
        unigram_sample = ngram_model.generate_unigrams()
        unigram_losses.extend(
            get_sequence_losses(model, unigram_sample, experiment.batch_size, experiment.seq_len, experiment.eod_index)
        )
        random_sample = ngram_model.generate_random()
        random_losses.extend(get_sequence_losses(model, random_sample, experiment.batch_size, experiment.seq_len, experiment.eod_index))

    mean_unigram_loss, unigram_conf_intervals = positional_summary_stats(
        unigram_losses, target_length=experiment.seq_len
    )
    means["unigram_loss"].extend(mean_unigram_loss)
    bottom_conf_intervals["unigram_loss"].extend(unigram_conf_intervals[0])
    top_conf_intervals["unigram_loss"].extend(unigram_conf_intervals[1])

    mean_bigram_loss, bigram_conf_intervals = positional_summary_stats(
        bigram_losses, target_length=experiment.seq_len
    )
    means["bigram_loss"].extend(mean_bigram_loss)
    bottom_conf_intervals["bigram_loss"].extend(bigram_conf_intervals[0])
    top_conf_intervals["bigram_loss"].extend(bigram_conf_intervals[1])

    mean_random_loss, random_conf_intervals = positional_summary_stats(
        random_losses, target_length=experiment.seq_len
    )
    means["random_loss"].extend(mean_random_loss)
    bottom_conf_intervals["random_loss"].extend(random_conf_intervals[0])
    top_conf_intervals["random_loss"].extend(random_conf_intervals[1])

    token_indices.extend(list(range(experiment.seq_len - 1)))

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
    experiment = Experiment(
        num_samples=1024,
        team="mistralai", 
        model_name="Mistral-7B-v0.1",
        batch_size=1,
        seq_len=2049, 
        steps=[0, 1, 2, 4, 8, 16, 256, 1000, 8000, 33_000, 143_000], # done: 66_000, 131_000, 
        d_vocab=50_277, # len(get_neo_tokenizer().vocab) 
        get_model=get_auto_model, 
        get_tokenizer=get_neo_tokenizer,
        eod_index=2 # tokenizer.eos_token_id
    )

    df = worker(experiment, ngram_path)
    df.to_csv(
        Path.cwd() / "output" / f"{experiment.model_name}_{experiment.num_samples}.csv",
        index=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--ngram_path",
        default="/mnt/ssd-1/lucia/pythia-deduped-bigrams.pkl",
        help="Path to pickled sparse scipy array of bigram counts over the Pile",
    )
    args = parser.parse_args()
    main(args.ngram_path)
