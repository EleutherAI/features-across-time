import argparse
import math
import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, load_from_disk
from numpy.typing import NDArray
from scipy.sparse import coo_matrix
from scipy.stats import entropy
from tokengrams import MemmapIndex
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer


def build_bigrams(tokens_path: Path, bigrams_path: Path):
    # tokens = np.array([1, 2, 3], dtype=np.uint16)
    tokens = np.memmap(tokens_path, dtype=np.uint16, mode="r")
    bigrams = (
        np.lib.stride_tricks.sliding_window_view(tokens, 2).view(np.uint32).squeeze()
    )

    counts = (
        np.bincount(bigrams, minlength=2**32)
        .reshape(2**16, 2**16)[:50277, :50277]
        .T
    )

    es_bigrams = coo_matrix(counts)

    with open(bigrams_path, "wb") as f:
        pickle.dump(es_bigrams, f)


def conditional_entropy(arr: NDArray, bpb_ratio: float):
    """Bigram model entropy"""
    """H(Y|X) = H(X, Y) - H(X)"""
    H = entropy(arr.data) - entropy(arr.sum(1))
    print("Entropy: ", H)
    print("Entropy (bpb):", H * bpb_ratio)


class NgramSeqModel:
    def __init__(
        self,
        bigrams_path: str,
        token_path: str | None,
        token_index_path: str | None,
        batch=1,
        seq_len=2049,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        self.batch = batch
        self.seq_len = seq_len

        with open(bigrams_path, "rb") as f:
            bigram_counts = pickle.load(f).toarray().astype(np.float32)

        self.unigram_probs = torch.tensor(bigram_counts).sum(dim=1).cuda()
        self.unigram_probs /= self.unigram_probs.sum()

        self.bigrams = bigram_counts + np.finfo(bigram_counts.dtype).eps
        self.bigram_probs = self.bigrams / self.bigrams.sum(axis=1)[:, None]

        if token_path and token_index_path:
            self.mmap_index = MemmapIndex(token_path, token_index_path)

    def generate_unigram_seq(self, num_samples: int) -> torch.Tensor:
        return torch.multinomial(
            self.unigram_probs, num_samples * self.seq_len, replacement=True
        ).reshape(num_samples, self.seq_len)

    def generate_bigram_seq(self, num_samples: int) -> np.array:
        """Auto-regressively generate bigram model sequence. Initialize each
        sequence by sampling from a unigram model.

        Underlying data structure: sparse scipy array of bigram counts over the Pile"""
        data = np.zeros((1024, 2049), dtype=np.int64)
        for i in tqdm(range(0, num_samples, self.batch)):
            result = [
                torch.multinomial(
                    self.unigram_probs, self.batch, replacement=True
                ).unsqueeze(1)
            ]
            for _ in range(self.seq_len - 1):
                prev = result[-1]
                rows = self.bigrams[prev[:, 0].cpu().numpy()]
                next = torch.multinomial(torch.tensor(rows, device="cuda"), 1)
                result.append(next)

            data[i : i + self.batch, :] = torch.cat(result, dim=1).cpu().numpy()
        return data

    def generate_ngrams(self, n: int, num_samples: int) -> NDArray:
        """Auto-regressively generate n-gram model sequence. Initialize each
        sequence by sampling from a unigram and then a bigram model.

        TODO try catch if we hit a sequence that only ever exists at the end of lines.
        The index is currently built on unchunked data so this shouldn't happen.
        """
        if n == 1:
            return np.array(self.generate_unigram_seq(num_samples).cpu())
        elif n == 2:
            return self.generate_bigram_seq(num_samples)
        else:
            samples = self.mmap_index.batch_sample(
                [], n=n, k=self.seq_len, num_samples=num_samples
            )
        return np.array(samples)

    def generate_ngram_dists(
        self, n: int, num_samples: int, vocab_size: int = 50_277
    ) -> None:
        batch = 64
        num_iters = math.ceil(num_samples / batch)
        print(num_iters)

        pile_data_loader = DataLoader(
            load_from_disk("/mnt/ssd-1/lucia/val_tokenized.hf"), batch_size=batch
        )
        pile = iter(pile_data_loader)

        mmap = np.memmap(
            f"{n}-gram-pile-dists.npy",
            mode="w+",
            dtype=np.float64,
            shape=(num_iters * batch * (self.seq_len - 1), vocab_size),
        )
        for i in tqdm(range(num_iters)):
            # dists are compared with logits. there are logits for all positions.
            # however there are no logits between chunks
            ngram_prefixes = []
            tokens_batch = next(pile)["input_ids"]
            for row in tokens_batch:
                ngram_prefixes.extend(
                    [row[i : i + (n - 1)].tolist() for i in range(len(row) - (n - 2))]
                )

            counts = torch.tensor(
                self.mmap_index.batch_next_token_counts(ngram_prefixes, vocab_size)
            )[:, :vocab_size]
            probs = counts / (
                counts.sum(dim=1).unsqueeze(1) + torch.finfo(torch.float64).eps
            )
            probs = probs.log()
            # 8192 50304

            # 8192 = 4 * (8192 - 1)
            chunk_len = batch * (self.seq_len - 1)
            print(batch, chunk_len, self.seq_len, batch * (self.seq_len - 1))
            print(((i * chunk_len) + chunk_len) - (i * chunk_len))
            mmap[(i * chunk_len) : ((i * chunk_len) + chunk_len)] = np.array(
                probs, dtype=np.float64
            )

    def get_sample_strs(self, n: int, k: int) -> None:
        seqs = self.generate_ngrams(n, k)
        return [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in seqs]

    def perplexity(self, sample: NDArray):
        nll = -np.log(
            self.bigram_probs[sample[:, :-1].flatten(), sample[:, 1:].flatten()]
        )
        return np.exp(nll.mean())

    def cross_entropy(self, sample: NDArray):
        rows = np.log(self.bigram_probs[sample])

        return F.cross_entropy(
            torch.tensor(rows[:-1]),
            torch.tensor(sample[1:]),
            reduction="mean",
        ).item()


def main(
    n: int,
    k: int,
    num_samples: int,
    tokens_path: Path,
    bigrams_path: Path,
    data_path: Path,
    bpb_ratio: float,
):
    if not os.path.exists(bigrams_path):
        build_bigrams(tokens_path, bigrams_path)

    with open(bigrams_path, "rb") as f:
        arr = pickle.load(f)
        unigram_H = entropy(arr.sum(1))
        print("Unigram entropy: ", unigram_H)
        print("Unigram entropy (bpb):", unigram_H * bpb_ratio)
        conditional_entropy(arr, bpb_ratio)

    ngram_model = NgramSeqModel(
        bigrams_path,
        None,  # one_billion_tokens_path,
        None,  # one_billion_tokens_index_path,
        4,
        k,
    )

    # Check sampled sequences look correct
    print(f"{n}-gram sequence sample:\n" + ngram_model.get_sample_strs(n, 1)[0])

    print(
        f"Loaded ngram model, generating {num_samples} \
            {n}-gram sequences of {k} tokens..."
    )

    data = ngram_model.generate_ngrams(n, num_samples)
    data_dict = {"input_ids": torch.tensor(data)}
    Dataset.from_dict(data_dict).save_to_disk(str(data_path / f"{n}-gram-sequences.hf"))

    # ngram_model.generate_ngram_dists(n, num_samples=1024)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--n",
        default=3,
        type=int,
        help="N-gram order of sequences",
    )
    parser.add_argument(
        "--k",
        default=2049,
        type=int,
        help="Sequence length",
    )
    parser.add_argument(
        "--num_samples",
        default=1024,
        type=int,
        help="Number of sequences",
    )
    parser.add_argument(
        "--data_path",
        default="data/pile",
        type=str,
        help="Path to write data",
    )
    # "/mnt/ssd-1/pile_preshuffled/standard/document.bin"
    parser.add_argument(
        "--tokens_path",
        # "data/es/es_tokenized.bin",
        default="/mnt/ssd-1/pile_preshuffled/standard/document.bin",
        type=str,
        help="Path to u16 tokenized dataset",
    )
    # "/mnt/ssd-1/lucia/pythia-deduped-bigrams.pkl",
    parser.add_argument(
        "--bigrams_path",
        default="data/pile/pythia-deduped-bigrams.pkl",
        type=str,
        help="Path to dataset bigram table",
    )
    args = parser.parse_args()

    bpb_ratio = 0.4157027

    main(
        args.n,
        args.k,
        args.num_samples,
        Path(args.tokens_path),
        Path(args.bigrams_path),
        Path(args.data_path),
        bpb_ratio,
    )
