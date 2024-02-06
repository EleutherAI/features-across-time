import pickle

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from scipy.stats import entropy
from tqdm.auto import tqdm
from transformers import AutoTokenizer


class NgramModel:
    def __init__(self, path: str, batch=1, seq_len=2049):
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        self.batch = batch
        self.seq_len = seq_len

        with open(path, "rb") as f:
            bigram_counts = pickle.load(f)

        self.unigrams = (
            torch.tensor(bigram_counts.toarray().astype(np.float32)).sum(dim=1).cuda()
        )
        self.bigrams = bigram_counts.toarray().astype(np.float32)
        self.bigrams += np.finfo(self.bigrams.dtype).eps

        self.prob_matrix = self.bigrams / self.bigrams.sum(axis=1)[:, None]

    def generate_bigrams(self) -> torch.Tensor:
        """Auto-regressively generate bigram model sequence. Initialize each
        sequence by sampling from a unigram model."""
        # TODO expand unigrams to avoid unsqueeze
        result = [
            torch.multinomial(self.unigrams, self.batch, replacement=True).unsqueeze(1)
        ]
        for _ in range(self.seq_len - 1):
            prev = result[-1]
            rows = self.bigrams[prev[:, 0].cpu().numpy()]
            next = torch.multinomial(torch.tensor(rows, device="cuda"), 1)
            result.append(next)
        return torch.cat(result, dim=1)

    def perplexity(self, sample: NDArray):
        nll = -np.log(
            self.prob_matrix[sample[:, :-1].flatten(), sample[:, 1:].flatten()]
        )
        return np.exp(nll.mean())

    def cross_entropy(self, sample: NDArray):
        rows = np.log(self.prob_matrix[sample])

        F.cross_entropy(
            torch.tensor(rows[:-1]),
            torch.tensor(sample[1:]),
            reduction="mean",
        ).item()


def conditional_entropy(arr: NDArray):
    """H(Y|X) = H(X, Y) - H(X)"""
    H = entropy(arr.data) - entropy(arr.sum(1))
    print("Entropy: ", H)  # 5.59
    bpb_ratio = 0.3366084909549386
    print("Entropy (bpb):", H * bpb_ratio)


def main():
    bigrams_path = "pythia-deduped-bigrams.pkl"
    batch = 4
    ngram_model = NgramModel(bigrams_path, batch)  # /mnt/ssd-1/lucia/

    def generate_samples():
        data = np.zeros((1024, 2049), dtype=np.int64)
        for i in tqdm(range(0, 1024, batch)):
            sample = ngram_model.generate_bigrams()
            data[i : i + batch, :] = sample.cpu().numpy()

        np.save("bigram-sequences.npy", data)

    generate_samples()

    # data = np.load('bigram-sequences.npy')
    # print(data[1022, :].shape)
    # for row in data:
    # breakpoint()
    # ngram_model.cross_entropy(data[0])

    # with open(bigrams_path, "rb") as f:
    #     arr = pickle.load(f)
    # conditional_entropy(arr)

    # perplexities = []
    # for i in tqdm(range(0, 1024 // batch, batch)):
    #     perplexity = ngram_model.perplexity(mmap[i:i + batch])
    #     perplexities.append(perplexity.item())
    # print(np.mean(perplexities)) # entropy = 5.594


if __name__ == "__main__":
    main()
