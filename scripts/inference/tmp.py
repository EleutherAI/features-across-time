import pickle
import torch
import numpy as np
from scripts.script_utils.divergences import kl_divergence_log_space, kl_divergence_linear_space
import os
from pathlib import Path


def get_dense_bigrams(path):
    with open(path, "rb") as f:
        counts = pickle.load(f).toarray().astype(np.float32)
        dense_bigrams = torch.tensor(
            counts / (counts.sum(axis=1) + np.finfo(np.float32).eps),
            device="cpu"
        )
        return dense_bigrams


bigrams_path = "data/es/bigrams.pkl"
es_dense_bigrams = get_dense_bigrams(bigrams_path)

pile_bigrams_path = "/mnt/ssd-1/lucia/pythia-deduped-bigrams.pkl"
pile_dense_bigrams = get_dense_bigrams(pile_bigrams_path)

# Different sizes (sparse COO array)
print(
    os.path.getsize(Path("data/es/bigrams.pkl")), 
    os.path.getsize(Path("/mnt/ssd-1/lucia/pythia-deduped-bigrams.pkl")))

es_dense_bigram_probs = es_dense_bigrams / es_dense_bigrams.sum(dim=-1).unsqueeze(-1) + np.finfo(np.float32).eps
pile_dense_bigram_probs = pile_dense_bigrams / pile_dense_bigrams.sum(dim=-1).unsqueeze(-1) + np.finfo(np.float32).eps

es_dense_unigram_probs = es_dense_bigrams.sum(dim=-1) + np.finfo(np.float32).eps
es_dense_unigram_probs /= es_dense_unigram_probs.sum()
pile_dense_unigram_probs = pile_dense_bigrams.sum(dim=-1) + np.finfo(np.float32).eps
pile_dense_unigram_probs /= pile_dense_unigram_probs.sum()

# All small
divs = kl_divergence_log_space(es_dense_unigram_probs.log(), pile_dense_unigram_probs.log())
print(divs.mean(), divs.max())

divs = kl_divergence_linear_space(es_dense_unigram_probs, pile_dense_unigram_probs)
print(divs.mean(), divs.max())

divs = kl_divergence_linear_space(es_dense_bigram_probs, pile_dense_bigram_probs)
print(divs.mean(), divs.max())

divs = kl_divergence_log_space(es_dense_bigram_probs.log(), pile_dense_bigram_probs.log())
print(divs.mean(), divs.max())

# divs = kl_divergence_linear_space(es_dense_bigrams, pile_dense_bigrams)
# print(divs.mean(), divs.max())
