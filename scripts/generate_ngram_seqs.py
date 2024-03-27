import pickle
import time
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from scipy.stats import entropy
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from tokengrams import MemmapIndex
import time

class NgramModel:
    def __init__(self, bigrams_path: str, token_path: str, token_index_path: str, batch=1, seq_len=2049):
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        self.batch = batch
        self.seq_len = seq_len

        with open(bigrams_path, "rb") as f:
            bigram_counts = pickle.load(f)

        self.unigrams = (
            torch.tensor(bigram_counts.toarray().astype(np.float32)).sum(dim=1).cuda()
        )
        self.bigrams = bigram_counts.toarray().astype(np.float32)
        self.bigrams += np.finfo(self.bigrams.dtype).eps

        self.prob_matrix = self.bigrams / self.bigrams.sum(axis=1)[:, None]
        
        self.mmap_index = MemmapIndex(token_path, token_index_path)

    def generate_bigrams(self, num_samples: int) -> torch.Tensor:
        """Auto-regressively generate bigram model sequence. Initialize each
        sequence by sampling from a unigram model.
        
        Underlying data structure: sparse scipy array of bigram counts over the Pile"""
        data = np.zeros((1024, 2049), dtype=np.int64)
        for i in tqdm(range(0, num_samples, self.batch)):
            result = [
                torch.multinomial(self.unigrams, self.batch, replacement=True).unsqueeze(1)
            ]
            for _ in range(self.seq_len - 1):
                prev = result[-1]
                rows = self.bigrams[prev[:, 0].cpu().numpy()]
                next = torch.multinomial(torch.tensor(rows, device="cuda"), 1)
                result.append(next)
            
            data[i : i + self.batch, :] = torch.cat(result, dim=1).numpy()
        return data

    def generate_ngrams(self, n: int, num_samples: int) -> NDArray:
        """Auto-regressively generate trigram model sequence. Initialize each
        sequence by sampling from a unigram and then a bigram model.
        
        TODO try catch if we hit a sequence that only ever exists at the end of lines.
        The index is currently built on unchunked data so this shouldn't happen.
        """
        samples = self.mmap_index.batch_sample([], n=n, k=self.seq_len, num_samples=num_samples)
        return np.array(samples)


    def perplexity(self, sample: NDArray):
        nll = -np.log(
            self.prob_matrix[sample[:, :-1].flatten(), sample[:, 1:].flatten()]
        )
        return np.exp(nll.mean())


    def cross_entropy(self, sample: NDArray):
        rows = np.log(self.prob_matrix[sample])

        return F.cross_entropy(
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


def bigram_properties(bigrams_path, ngram_model, batch):
    data = np.load('bigram-sequences.npy')
    for row in data:
        ngram_model.cross_entropy(row)

    with open(bigrams_path, "rb") as f:
        arr = pickle.load(f)
    conditional_entropy(arr)

    # perplexities = []
    # for i in tqdm(range(0, 1024 // batch, batch)):
    #     perplexity = ngram_model.perplexity(mmap[i:i + batch])
    #     perplexities.append(perplexity.item())
    # print(np.mean(perplexities)) # entropy = 5.594


# one_billion_tokens_path = "/mnt/ssd-1/nora/pile-head.bin"
# one_billion_tokens_index_path = "/mnt/ssd-1/nora/pile-head.idx"
# ten_billion_tokens_index_path = "/mnt/ssd-1/nora/pile-10G.idx"
# ten_billion_tokens_path = "/mnt/ssd-1/nora/pile-10G.bin"
def main(n: int, k: int, num_samples: int):
    bigrams_path = "/mnt/ssd-1/lucia/pythia-deduped-bigrams.pkl"
    forty_billion_tokens_index_path = "/mnt/ssd-1/nora/pile-40B.idx"
    forty_billion_tokens_path = "/mnt/ssd-1/nora/pile-40B.bin"
    
    ngram_model = NgramModel(
        bigrams_path, 
        forty_billion_tokens_path, 
        forty_billion_tokens_index_path, 
        4, 
        k
    )  
    print(f"Loaded ngram model, generating {num_samples} {n}-gram sequences of {k} tokens...")

    start = time.time()
    data = ngram_model.generate_ngrams(n, num_samples)
    print(time.time() - start)
    np.save(f"{4}-gram-sequences-{num_samples}.npy", data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--n",
        default=3,
        type=int,
        help="N-gram order to generate",
    )
    parser.add_argument(
        "--k",
        default=2049,
        type=int,
        help="Sequence length to generate",
    )
    parser.add_argument(
        "--numsamples",
        default=1024,
        type=int,
        help="Number of sequences to generate",
    )
    args = parser.parse_args()
    main(args.n, args.k, args.numsamples)
