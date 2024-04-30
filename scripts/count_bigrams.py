import numpy as np
from scipy.sparse import coo_matrix
import pickle
from pathlib import Path


def count_bigrams(tokens_path: Path, counts_path: Path):
    # tokens = np.array([1, 2, 3], dtype=np.uint16)
    tokens = np.memmap(tokens_path, dtype=np.uint16, mode="r")[:1_000_000]
    bigrams = np.lib.stride_tricks.sliding_window_view(tokens, 2).view(np.uint32).squeeze()

    counts = np.bincount(bigrams, minlength=2**32).reshape(2**16, 2**16)[:50277, :50277].T

    es_bigrams = coo_matrix(counts)

    with open(counts_path, 'wb') as f:
        pickle.dump(es_bigrams, f)


if __name__ == "__main__":
    tokens_path = Path("/mnt/ssd-1/lucia/es_1b_full.bin") # "/mnt/ssd-1/pile_preshuffled/standard/document.bin"
    counts_path = Path("/mnt/ssd-1/lucia/es/es-bigrams.pkl")
    count_bigrams(tokens_path, counts_path)