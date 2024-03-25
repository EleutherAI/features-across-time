import pickle
import time

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from scipy.stats import entropy
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from tokengrams import InMemoryIndex, MemmapIndex


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


        one_billion_tokens_path = "/mnt/ssd-1/nora/pile-head.bin"
        # ten_billion_tokens_path = "/mnt/ssd-1/nora/pile-10g.bin"

        one_billion_tokens_index_path = "/mnt/ssd-1/nora/pile-head.idx"
        self.mmap_index = MemmapIndex(one_billion_tokens_path, one_billion_tokens_index_path)

    def generate_bigrams(self) -> torch.Tensor:
        """Auto-regressively generate bigram model sequence. Initialize each
        sequence by sampling from a unigram model."""
        result = [
            torch.multinomial(self.unigrams, self.batch, replacement=True).unsqueeze(1)
        ]
        for _ in range(self.seq_len - 1):
            prev = result[-1]
            rows = self.bigrams[prev[:, 0].cpu().numpy()]
            next = torch.multinomial(torch.tensor(rows, device="cuda"), 1)
            result.append(next)
        return torch.cat(result, dim=1)

    def generate_trigrams(self) -> torch.Tensor:
        """Auto-regressively generate trigram model sequence. Initialize each
        sequence by sampling from a unigram and then a bigram model."""
        start = time.time()
        result = torch.zeros(self.batch, self.seq_len, dtype=torch.int64)
        result[:, 0] = torch.multinomial(self.unigrams, self.batch, replacement=True)
        rows = self.bigrams[result[:, 0].cpu().numpy()]
        result[:, 1] = torch.multinomial(torch.tensor(rows, device="cuda"), 1).squeeze(-1)

        # Build as lists
        for i in tqdm(range(2, self.seq_len)):
            next = torch.zeros(self.batch, dtype=torch.int64)
            for j in range(self.batch): # todo vectorized mmap_index.sample over batch
                item = result[j, i - 2:i].tolist() # last two tokens of each batch [batch, 2]
                # todo try catch if we hit a sequence that only ever exists at the end of lines
                # the index is currently built on unchunked data so this shouldn't happen
                next[j] = self.mmap_index.sample(item)
            result[:, i] = next
        # print(result.shape, "time taken: ", time.time() - start)

        return result


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


def generate_bigram_samples(batch, ngram_model):
        data = np.zeros((1024, 2049), dtype=np.int64)
        for i in tqdm(range(0, 1024, batch)):
            sample = ngram_model.generate_bigrams()
            data[i : i + batch, :] = sample.cpu().numpy()

        np.save("bigram-sequences.npy", data)

def generate_trigram_samples(batch, ngram_model):
        data = np.zeros((1024, 2049), dtype=np.int64)
        for i in tqdm(range(0, 1024, batch)):
            sample = ngram_model.generate_trigrams()
            data[i : i + batch, :] = sample.cpu().numpy()

        np.save("trigram-sequences.npy", data)

def main():
    bigrams_path = "pythia-deduped-bigrams.pkl"
    batch = 4
    ngram_model = NgramModel(bigrams_path, batch)  # /mnt/ssd-1/lucia/

    # generate_bigram_samples(batch, ngram_model)
    generate_trigram_samples(batch, ngram_model)

    # data = np.load('bigram-sequences.npy')
    # for row in data:
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
