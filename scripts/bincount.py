import numpy as np
from pathlib import Path


def bincount(tokens_path: Path, counts_path: Path):
    tokens = np.memmap(
        tokens_path, dtype=np.uint16, mode="r")[:1_000_000]
    bigrams = np.lib.stride_tricks.sliding_window_view(tokens, 2).view(np.uint32).squeeze()

    counts = np.bincount(bigrams, minlength=2**32).reshape(2**16, 2**16)[:50277, :50277].T
    
    counts_out = np.memmap(counts_path, dtype=np.uint16, mode="w", shape=(counts.shape))
    counts_out[:] = counts


if __name__ == "__main__":
    tokens_path = Path("/mnt/ssd-1/pile_preshuffled/standard/document.bin")
    counts_path = Path("/mnt/ssd-1/lucia/es/es-1b-counts.bin")
    bincount(tokens_path, counts_path)
