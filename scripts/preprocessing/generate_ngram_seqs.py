import pickle
import time
import argparse
import time
import math

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from tokengrams import MemmapIndex
from torch.utils.data import DataLoader
from datasets import load_from_disk

from scripts.script_utils.bpb import conditional_entropy


class NgramSeqModel:
    def __init__(self, bigrams_path: str, token_path: str | None, token_index_path: str | None, batch=1, seq_len=2049):
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        self.batch = batch
        self.seq_len = seq_len

        with open(bigrams_path, "rb") as f:
            bigram_counts = pickle.load(f).toarray().astype(np.float32)

        self.unigram_probs = (
            torch.tensor(bigram_counts).sum(dim=1).cuda()
        )
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
                torch.multinomial(self.unigram_probs, self.batch, replacement=True).unsqueeze(1)
            ]
            for _ in range(self.seq_len - 1):
                prev = result[-1]
                rows = self.bigrams[prev[:, 0].cpu().numpy()]
                next = torch.multinomial(torch.tensor(rows, device="cuda"), 1)
                result.append(next)
            
            data[i : i + self.batch, :] = torch.cat(result, dim=1).cpu().numpy()
        return data


    def generate_ngrams(self, n: int, num_samples: int) -> NDArray:
        """Auto-regressively generate trigram model sequence. Initialize each
        sequence by sampling from a unigram and then a bigram model.
        
        TODO try catch if we hit a sequence that only ever exists at the end of lines.
        The index is currently built on unchunked data so this shouldn't happen.
        """
        if n == 1:
            return np.array(self.generate_unigram_seq(num_samples).cpu())
        elif n == 2:
            return self.generate_bigram_seq(num_samples)
        else:
            samples = self.mmap_index.batch_sample([], n=n, k=self.seq_len, num_samples=num_samples)
        return np.array(samples)


    def generate_ngram_dists(self, n: int, num_samples: int, vocab_size: int = 50_277) -> None:
        batch = 64
        num_iters = math.ceil(num_samples / batch)
        print(num_iters)

        pile_data_loader = DataLoader(load_from_disk("/mnt/ssd-1/lucia/val_tokenized.hf"), batch_size=batch)
        pile = iter(pile_data_loader)

        mmap = np.memmap(f'{n}-gram-pile-dists.npy', mode='w+', dtype=np.float64, shape=(num_iters * batch * (self.seq_len - 1), vocab_size))
        for i in tqdm(range(num_iters)):
            # dists are compared with logits. there are logits for all positions. however there are no logits between chunks
            ngram_prefixes = []
            tokens_batch = next(pile)["input_ids"]
            for row in tokens_batch:
                ngram_prefixes.extend([row[i:i + (n - 1)].tolist() for i in range(len(row) - (n - 2))])

            counts = torch.tensor(self.mmap_index.batch_next_token_counts(ngram_prefixes, vocab_size))[:, :vocab_size]
            probs = counts / (counts.sum(dim=1).unsqueeze(1) + torch.finfo(torch.float64).eps)
            probs = probs.log()
            # 8192 50304

            # 8192 = 4 * (8192 - 1)
            chunk_len = batch * (self.seq_len - 1)
            print(batch, chunk_len, self.seq_len, batch * (self.seq_len - 1))
            print(((i * chunk_len) + chunk_len) - (i * chunk_len))
            mmap[
                (i * chunk_len):((i * chunk_len) + chunk_len)
            ] = np.array(probs, dtype=np.float64)


    def print_samples(self) -> None:
        # Demonstrate sampling is in working order by sampling and displaying 1- and 2-gram sequences
        for i in [1, 2]:
            seqs = self.generate_ngrams(i, 3)
            decoded_strings = [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in seqs]
            print(f"3 {i}-gram sequence samples:\n" + "\n".join(decoded_strings))



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




def bigram_properties(bigrams_path, ngram_model, batch):
    data = np.load('bigram-sequences.npy')
    entropies = []
    for row in data:
        entropy = ngram_model.cross_entropy(row)
        entropies.append(entropy)
    print(entropies[:5], sum(entropies) / len(entropies))

    with open(bigrams_path, "rb") as f:
        arr = pickle.load(f)
    conditional_entropy(arr)

    # perplexities = []
    # for i in tqdm(range(0, 1024 // batch, batch)):
    #     perplexity = ngram_model.perplexity(mmap[i:i + batch])
    #     perplexities.append(perplexity.item())
    # print(np.mean(perplexities)) # entropy = 5.594


def trigram_properties(bigrams_path, ngram_model, batch):
    data = np.memmap('3-gram-sequences.npy', mode='r+')
    entropies = []
    for row in data:
        entropy = ngram_model.cross_entropy(row)
        entropies.append(entropy)
    print(entropies[:5], sum(entropies) / len(entropies))

    # with open(bigrams_path, "rb") as f:
    #     arr = pickle.load(f)
    # conditional_entropy(arr)

# one_billion_tokens_path = "/mnt/ssd-1/nora/pile-head.bin"
# one_billion_tokens_index_path = "/mnt/ssd-1/nora/pile-head.idx"
# ten_billion_tokens_index_path = "/mnt/ssd-1/nora/pile-10G.idx"
# ten_billion_tokens_path = "/mnt/ssd-1/nora/pile-10G.bin"


def main(n: int, k: int, num_samples: int):
    bigram_paths = {
        "pile": "/mnt/ssd-1/lucia/pythia-deduped-bigrams.pkl",
        "es": "/mnt/ssd-1/lucia/es/es-bigrams.pkl"
    }

    forty_billion_tokens_index_path = "/mnt/ssd-1/nora/pile-40B.idx"
    forty_billion_tokens_path = "/mnt/ssd-1/nora/pile-40B.bin"
    # one_billion_tokens_path = "/mnt/ssd-1/nora/pile-head.bin"
    # one_billion_tokens_index_path = "/mnt/ssd-1/nora/pile-head.idx"
    
    ngram_model = NgramSeqModel(
        bigram_paths['es'], 
        None,
        None,
        # one_billion_tokens_path, 
        # one_billion_tokens_index_path, 
        4, 
        k
    )
    ngram_model.print_samples()
    # print(f"Loaded ngram model, generating {num_samples} {n}-gram sequences of {k} tokens...")

    # print(bigram_properties(bigrams_path, ngram_model, 4))
    # print(trigram_properties(bigrams_path, ngram_model, 4))

    # start = time.time()
    # data = np.memmap('3-gram-sequences.npy', mode='r+')
    

    # data = ngram_model.generate_ngrams(n, num_samples)
    # print(time.time() - start)
    # mmap = np.memmap(f"es-{n}-gram-sequences.npy", mode="w+", dtype=data.dtype, shape=data.shape)
    # mmap[:] = data

    # ngram_model.generate_ngram_dists(n, num_samples=1024)


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
