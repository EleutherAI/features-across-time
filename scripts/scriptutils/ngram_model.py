import numpy as np
import scipy
import torch
from transformers import AutoTokenizer


import pickle


class NgramModel:
    def __init__(self, path: str, batch=1, seq_len=2049, tokenizer=None):
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        self.d_vocab = len(self.tokenizer.vocab)
        self.batch = batch
        self.seq_len = seq_len

        with open(path, "rb") as f:
            bigram_counts = pickle.load(f)
        bigram_counts = bigram_counts.toarray().astype(np.float32)

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
        self.bigram_samples = np.load("bigram-sequences.npy")

    def generate_unigrams(self) -> torch.Tensor:
        return torch.multinomial(
            self.unigrams, self.batch * self.seq_len, replacement=True
        ).reshape(self.batch, self.seq_len)

    def generate_unigram_strs(self) -> list[str]:
        tokens = self.generate_unigrams()
        return [self.tokenizer.decode(row.tolist()) for row in tokens]

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

    # # separate slice sparse array function with test
    # def sample_bigram(self, prev: torch.Tensor) -> torch.Tensor:
    #     """Given a batch of previous tokens, sample from a bigram model
    #     using conditional distributions stored in a sparse CSR tensor."""
    #     starts = self.bigrams.crow_indices()[prev]
    #     ends = self.bigrams.crow_indices()[prev + 1]

    #     # 0 padding to batch rows with variable numbers of non-zero elements
    #     token_probs = torch.zeros(
    #         (self.batch, self.d_vocab), dtype=torch.float32, device="cuda"
    #     )
    #     token_col_indices = torch.zeros(
    #         (self.batch, self.d_vocab), dtype=torch.int32, device="cuda"
    #     )
    #     for i in range(self.batch):
    #         token_probs[i, : ends[i] - starts[i]] = self.bigrams.values()[
    #             starts[i] : ends[i]
    #         ]
    #         token_col_indices[i, : ends[i] - starts[i]] = self.bigrams.col_indices()[
    #             starts[i] : ends[i]
    #         ]

    #     sampled_value_indices = torch.multinomial(token_probs, 1)
    #     return torch.gather(token_col_indices, 1, sampled_value_indices)

    def generate_bigrams(self, i: int) -> torch.Tensor:
        """Auto-regressively generate bigram model sequence. Initialize each
        sequence by sampling from a unigram model."""
        # i should range from 0 to 1024/2
        batch = self.bigram_samples[
            i * self.batch : (i * self.batch) + self.batch, :50277
        ]
        return torch.tensor(batch, device="cuda").long()
        # result = [
        #     torch.multinomial(self.unigrams, self.batch, replacement=True).unsqueeze(1)
        # ]
        # for _ in range(self.seq_len - 1):
        #     prev = result[-1]
        #     result.append(self.sample_bigram(prev))
        # return torch.cat(result, dim=1)

    def generate_bigram_strs(self) -> list[str]:
        tokens = self.generate_bigrams()
        return [self.tokenizer.decode(row.tolist()) for row in tokens]