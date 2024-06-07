import pickle
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer


class NgramModel:
    def __init__(self, path: str, batch=1, seq_len=2049, tokenizer=None, device="cuda"):
        self.tokenizer = (
            tokenizer
            if tokenizer is not None
            else AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        )
        self.d_vocab = len(self.tokenizer.vocab)
        self.batch_size = batch
        self.seq_len = seq_len
        self.device = device

        with open(path, "rb") as f:
            bigram_counts = pickle.load(f).toarray().astype(np.float32)

        self.bigram_probs = bigram_counts / (
            bigram_counts.sum(axis=1)[:, None] + np.finfo(np.float32).eps
        )
        self.bigram_probs = (
            torch.tensor(self.bigram_probs).to_sparse().to_sparse_csr().to(self.device)
        )

        self.unigram_probs = torch.tensor(bigram_counts).sum(dim=1).to(self.device)
        self.unigram_probs /= self.unigram_probs.sum()

        # forty_billion_tokens_index_path = "/mnt/ssd-1/nora/pile-40B.idx"
        # forty_billion_tokens_path = "/mnt/ssd-1/nora/pile-40B.bin"
        # self.mmap_index = MemmapIndex(
        #     forty_billion_tokens_path,
        #     forty_billion_tokens_index_path)

    def generate_unigram_seq(self) -> torch.Tensor:
        return torch.multinomial(
            self.unigram_probs, self.batch_size * self.seq_len, replacement=True
        ).reshape(self.batch_size, self.seq_len)

    def get_ngram_seq(self, n: int, i: int, sequence_path=Path.cwd()) -> torch.Tensor:
        """Fetch a precomputed batch of n-gram sequences"""
        ngram_samples = load_from_disk(str(sequence_path / f"{n}-gram-sequences.hf"))
        ngram_samples.set_format("torch", columns=["input_ids"])

        return ngram_samples[
            i * self.batch_size : (i * self.batch_size) + self.batch_size
        ]["input_ids"]

    def get_ngram_str(self, n: int, i: int) -> list[str]:
        """Fetch a precomputed batch of n-gram sequences and convert to strs"""
        if n == 1:
            tokens = self.get_ngram_seq(n, i) if n != 1 else self.generate_unigram_seq()
        return [self.tokenizer.decode(row.tolist()) for row in tokens]

    def get_bigram_prob(self, prev: torch.Tensor) -> torch.Tensor:
        """Non-ideal behaviour with flattened tensors where the end of once batch and
        the start of the next forms a bigram - should be extended to accept a batch
        dimension
        """
        starts = self.bigram_probs.crow_indices()[prev]
        ends = self.bigram_probs.crow_indices()[prev + 1]

        # 0 padding to batch rows with variable numbers of non-zero elements
        bigram_dists = torch.zeros(
            (len(prev), self.d_vocab), dtype=torch.float32, device=self.device
        )
        for i in range(len(prev)):
            filled_col_indices = self.bigram_probs.col_indices()[starts[i] : ends[i]]
            filled_col_values = self.bigram_probs.values()[starts[i] : ends[i]]
            bigram_dists[i][filled_col_indices] = filled_col_values
        return bigram_dists

    def get_ngram_prob(self, tokens: torch.Tensor, n: int) -> torch.Tensor:
        if n == 1:
            return (
                self.unigram_probs.expand((*tokens.shape, -1))
                + torch.finfo(torch.float32).eps
            )
        if n == 2:
            if len(tokens.shape) == 1:
                return self.get_bigram_prob(tokens) + torch.finfo(torch.float32).eps
            else:
                return (
                    torch.stack([self.get_bigram_prob(row) for row in tokens])
                    + torch.finfo(torch.float32).eps
                )

        ngram_prefixes = []
        for row in tokens:
            ngram_prefixes.extend(
                [row[i : i + (n - 1)].tolist() for i in range(len(row) - (n - 2))]
            )

        counts = torch.tensor(
            self.mmap_index.batch_next_token_counts(ngram_prefixes, self.d_vocab)
        )[:, : self.d_vocab]
        return counts / (
            counts.sum(dim=1).unsqueeze(1) + torch.finfo(torch.float64).eps
        )
