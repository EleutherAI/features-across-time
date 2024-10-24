import argparse
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
from tokengrams import ShardedMemmapIndex
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer


def conditional_entropy(arr: NDArray):
    """Bigram model entropy"""
    """H(Y|X) = H(X, Y) - H(X)"""
    return entropy(arr.data) - entropy(arr.sum(1))


def build_bigrams(tokens_path: Path, bigrams_path: Path):
    tokens = np.memmap(tokens_path, dtype=np.uint16, mode="r")
    bigrams = (
        np.lib.stride_tricks.sliding_window_view(tokens, 2).view(np.uint32).squeeze()
    )

    counts = (
        np.bincount(bigrams, minlength=2**32)
        .reshape(2**16, 2**16)[:50277, :50277]
        .T
    )

    sparse_bigrams = coo_matrix(counts)
    with open(bigrams_path, "wb") as f:
        pickle.dump(sparse_bigrams, f)


class NgramSeqModel:
    def __init__(
        self,
        bigrams_path: Path,
        tokengrams_paths: list[tuple[str, str]] | None,
        batch: int,
        seq_len: int,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        self.batch = batch
        self.seq_len = seq_len

        with open(bigrams_path, "rb") as f:
            bigram_counts = pickle.load(f).toarray().astype(np.float32)

        self.unigram_probs = torch.tensor(bigram_counts).sum(dim=1).cuda()
        self.unigram_probs /= self.unigram_probs.sum()

        self.bigram_probs = bigram_counts / bigram_counts.sum(axis=1)[:, None]
        self.bigram_probs = np.nan_to_num(self.bigram_probs)
        self.bigram_probs = torch.tensor(self.bigram_probs).cuda()


        print(self.bigram_probs.sum())

        if tokengrams_paths:
            self.tokengrams = ShardedMemmapIndex(tokengrams_paths, self.tokenizer.vocab_size)

    def generate_unigram_seq(self, num_samples: int) -> torch.Tensor:
        return torch.multinomial(
            self.unigram_probs, num_samples * self.seq_len, replacement=True
        ).reshape(num_samples, self.seq_len)

    def generate_bigram_seq(self, num_samples: int) -> np.ndarray:
        """Auto-regressively generate bigram model sequence. Initialize each
        sequence by sampling from a unigram model.

        Underlying data structure: sparse scipy array of bigram counts over the Pile"""
        data = np.zeros((num_samples, self.seq_len), dtype=np.int64)
        for i in tqdm(range(0, num_samples, self.batch)):
            result = [
                torch.multinomial(
                    self.unigram_probs, self.batch, replacement=True
                ).unsqueeze(1)
            ]
            for _ in range(self.seq_len - 1):
                prev = result[-1]
                rows = self.bigram_probs[prev[:, 0].cpu().numpy()]
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
            import time

            start = time.time()
            samples = self.tokengrams.sample_unsmoothed(
                [], n=n, k=self.seq_len, num_samples=num_samples
            )
            print(
                time.time() - start,
                f"seconds to generate {num_samples} * \
{self.seq_len} tokens of order {n}",
            )
        return np.array(samples)


    def generate_ngram_logprobs(
        self,
        data: Dataset,
        data_path: Path,
        n: int,
        vocab_size: int = 50_277,
        batch_size: int = 4,
    ) -> None:
        data_loader = DataLoader(data, batch_size) # type: ignore

        file_path = data_path / f"{n}-gram-pile-logprobs-32bit-{len(data)}.npy"
        mode = "w+" if not file_path.exists() else "r+"
        mmap = np.memmap(
            file_path,
            mode=mode,
            dtype=np.float32,
            shape=(len(data) * self.seq_len, vocab_size),
        )

        chunk_len = batch_size * self.seq_len
        for i, batch in tqdm(enumerate(data_loader)):
            if n == 1:
                logprobs = self.unigram_probs.log().cpu().repeat(chunk_len).reshape(-1, vocab_size)
                mmap[(i * chunk_len) : ((i * chunk_len) + chunk_len)] = np.array(logprobs.cpu(), dtype=np.float32)
            elif n == 2:
                logprobs = self.bigram_probs[batch["input_ids"].flatten()].log()
                mmap[(i * chunk_len) : ((i * chunk_len) + chunk_len)] = np.array(logprobs.cpu(), dtype=np.float32)
            else:
                ngram_prefixes = []
                for row in batch["input_ids"]:
                    # split into sequences of up to n - 1 tokens that end with each token
                    ngram_prefixes.extend(
                        [row[max(0, i - (n - 1)) : i].tolist() for i in range(len(row))]
                    )
                counts = torch.tensor(
                    self.tokengrams.batch_count_next(ngram_prefixes),
                )[:, :-1]
                logprobs = torch.nan_to_num(counts / counts.sum(dim=1, keepdim=True)).log()
                mmap[(i * chunk_len) : ((i * chunk_len) + chunk_len)] = np.array(
                    logprobs, dtype=np.float32
                )

            if i == 0:
                print(logprobs[0].exp().sum())
    
    def generate_smoothed_ngram_dists(
        self,
        data: Dataset,
        data_path: Path,
        n: int,
        vocab_size: int = 50_277,
        batch_size: int = 16,
    ) -> None:
        print(n)
        data_loader = DataLoader(data, batch_size) # type: ignore

        file_path = data_path / f"smoothed-{n}-gram-pile-dists-16bit-{len(data)}.npy"
        mode = "w+" if not file_path.exists() else "r+"
        mmap = np.memmap(
            file_path,
            mode=mode,
            dtype=np.float16,
            shape=(len(data) * self.seq_len, vocab_size),
        )

        chunk_len = batch_size * self.seq_len
        for i, batch in tqdm(enumerate(data_loader)):
            ngram_prefixes = []
            for row in batch["input_ids"]:
                ngram_prefixes.extend(
                    [row[max(0, i - (n - 1)) : i].tolist() for i in range(len(row))]
                )
            res = self.tokengrams.batch_get_smoothed_probs(ngram_prefixes)
            probs = np.log(np.array(res, dtype=np.float32))
            
            mmap[(i * chunk_len) : ((i * chunk_len) + chunk_len)] = probs.astype(np.float16)


    def get_sample_strs(self, n: int, k: int) -> list[str]:
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
    ngram_orders: list[int],
    k: int,
    num_samples: int,
    bpb_coeff: float,
    tokens_path: Path,
    tokengrams_paths: list[tuple[str, str]],
    data_path: Path
):
    if 2 in ngram_orders and not os.path.exists(data_path / "bigrams.pkl"):
        print("Building bigrams...")
        build_bigrams(tokens_path, data_path / "bigrams.pkl")

        with open(data_path / "bigrams.pkl", "rb") as f:
            arr = pickle.load(f)
            H = entropy(arr.sum(1))
            print("Unigram entropy: ", H, "Unigram entropy (bpb):", H * bpb_coeff)
            H = conditional_entropy(arr)
            print("Bigram entropy: ", H, "Bigram entropy (bpb):", H * bpb_coeff)

    for n in ngram_orders:
        print(f"Loading {n}-gram model...")
        ngram_model = NgramSeqModel(
            data_path / "bigrams.pkl", 
            tokengrams_paths if max(ngram_orders) > 2 else None, 
            1, 
            k
        )

        print(f"{n}-gram sequence sample:\n" + ngram_model.get_sample_strs(n, 1)[0])

        print(f"Generating {num_samples} {n}-gram sequences of {k} tokens...")
        data = ngram_model.generate_ngrams(n, num_samples)
        data_dict = {"input_ids": torch.tensor(data)}
        Dataset.from_dict(data_dict).save_to_disk(str(data_path / f"{n}-gram-sequences-full-pile-{num_samples}.hf"))

        print(f"Generating {n}-gram logprobs...")
        pile_val = load_from_disk(str(data_path / "val_tokenized.hf")).select(
            range(num_samples)
        )
        ngram_model.generate_ngram_logprobs(pile_val, data_path, n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--n", default=[3], nargs="+", type=int, 
                        help="N-gram models to sample from")
    parser.add_argument("--k", default=2049, help="Sample length", type=int)
    parser.add_argument("--num_samples", default=1024, type=int)
    parser.add_argument("--bpb_coeff", default=0.3650388, type=float) # pile
    parser.add_argument(
        "--data_path",
        default="data/pile-deduped",
        type=str,
        help="Path to read and write data",
    )
    parser.add_argument(
        "--tokens_path",
        default="/mnt/ssd-1/pile_preshuffled/standard/document.bin",
        type=str,
        help="Path to u16 tokenized dataset",
    )
    args = parser.parse_args()

    tokengrams_paths = [
        (
            f'/mnt/ssd-1/pile-ngrams-tokens/document-{i:05d}-of-00020.bin', 
            f'/mnt/ssd-1/pile-suffix-arrays/suffix_array_{i:02d}.idx'
        )
        for i in range(21)
    ][:4]

    main(
        args.n,
        args.k,
        args.num_samples,
        args.bpb_coeff,
        Path(args.tokens_path),
        tokengrams_paths,
        Path(args.data_path),
    )
