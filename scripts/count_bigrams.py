import numpy as np
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
    print(len(tokenizer.vocab))
    count_bigrams(tokens, len(tokenizer.vocab))


if __name__ == "__main__":
    main()
