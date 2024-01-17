import io

import numpy as np
import zstandard as zstd
from transformers import AutoTokenizer


def count_bigrams(arr: np.ndarray, vocab_size: int) -> np.ndarray:
    assert arr.dtype == np.uint16

    windows = np.lib.stride_tricks.sliding_window_view(arr, 2).view(np.uint32)
    counts = np.bincount(windows.squeeze(), minlength=2**32).reshape(-1, 2**16)
    return counts[:vocab_size, :vocab_size].T


def main():
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    tokens = np.memmap(
        "/mnt/ssd-1/pile_preshuffled/standard/document.bin", dtype=np.uint16, mode="r"
    )
    counts = count_bigrams(tokens, len(tokenizer.vocab))

    cctx = zstd.ZstdCompressor()
    buffer = io.BytesIO()
    np.save(buffer, counts)
    with open("2-grams.zst", "wb") as compressed_file:
        compressed_data = cctx.compress(buffer.getvalue())
        compressed_file.write(compressed_data)


if __name__ == "__main__":
    main()
