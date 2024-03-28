import pickle

import numpy as np
import scipy
import torch
from transformers import AutoTokenizer


class NgramModel:
    def __init__(
            self, 
            path: str, 
            batch=1, 
            seq_len=2049, 
            tokenizer=None,
            use_bigram_dists=True
        ):
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        self.d_vocab = len(self.tokenizer.vocab)
        self.batch_size = batch
        self.seq_len = seq_len

        if use_bigram_dists:
            with open(path, "rb") as f:
                bigram_counts = pickle.load(f).toarray().astype(np.float32)

            self.unigrams = torch.tensor(bigram_counts).sum(dim=1).cuda()
            self.unigrams /= self.unigrams.sum()

            # Conver to sparse CSR tensor in a dumb way
            sparse_bigram_probs = (
                torch.tensor(
                    bigram_counts / (bigram_counts.sum(axis=1) + np.finfo(np.float32).eps)
                )
                .log()
                .to_sparse()
            )
            del bigram_counts

            indices = sparse_bigram_probs.indices().numpy()
            values = sparse_bigram_probs.values().numpy()
            shape = sparse_bigram_probs.shape
            sparse_csr_bigram_probs = scipy.sparse.coo_matrix(
                (values, (indices[0], indices[1])), shape=shape
            ).tocsr()
            del sparse_bigram_probs, indices, values, shape
            self.log_bigrams = torch.sparse_csr_tensor(
                sparse_csr_bigram_probs.indptr.astype(np.int64),
                sparse_csr_bigram_probs.indices.astype(np.int64),
                sparse_csr_bigram_probs.data.astype(np.float32),
                dtype=torch.float32,
                device="cuda",
            )
        else:
            self.log_bigrams = None


    def generate_unigrams(self) -> torch.Tensor:
        return torch.multinomial(
            self.unigrams, self.batch_size * self.seq_len, replacement=True
        ).reshape(self.batch_size, self.seq_len)


    def get_bigram_dists(self, prev: torch.Tensor) -> torch.Tensor:
        starts = self.log_bigrams.crow_indices()[prev]
        ends = self.log_bigrams.crow_indices()[prev + 1]

        # 0 padding to batch rows with variable numbers of non-zero elements
        bigram_dists = torch.zeros(
            (len(prev), self.d_vocab), dtype=torch.float32, device="cuda"
        )
        for i in range(len(prev)):
            filled_col_indices = self.log_bigrams.col_indices()[starts[i] : ends[i]]
            filled_col_values = self.log_bigrams.values()[starts[i] : ends[i]]
            bigram_dists[i][filled_col_indices] = filled_col_values
        return bigram_dists
    

    def get_ngrams(self, n: int, i: int) -> torch.Tensor:
        """Fetch a precomputed batch of n-gram sequences"""
        ngram_samples = np.memmap(f"{n}-gram-sequences.npy", dtype=np.int64, mode='r', shape=(1024, self.seq_len))
        batch = ngram_samples[
            i * self.batch_size:(i * self.batch_size) + self.batch_size
        ]

        return torch.tensor(batch, device="cuda").long()


    def get_ngram_strs(self, n: int, i: int) -> list[str]:
        """Fetch a precomputed batch of n-gram sequences and convert to strs"""
        if n == 1:
            tokens = self.get_ngrams(n, i) if n != 1 else self.generate_unigrams()
        return [self.tokenizer.decode(row.tolist()) for row in tokens]


    def get_ngram_dists(self, query: list[int], n: int) -> torch.Tensor:
        return torch.zeros()