import os

import numpy as np


def main(seq_len: int = 2049):
    tokens_path = "/mnt/ssd-1/nora/pile-40B.bin"
    output_path = "/mnt/ssd-1/lucia/pile-1GB.bin"

    # Number of bytes in 1 GB
    bytes_in_gb = 1 * 1024 * 1024 * 1024

    # Load the input data
    file_size = os.path.getsize(tokens_path)
    element_size = np.dtype(np.uint16).itemsize
    num_rows = file_size // element_size // seq_len
    token_mmap = np.memmap(
        tokens_path, dtype=np.uint16, mode="r", shape=(num_rows, seq_len)
    )

    # Save the first 1 GB of data to a new binary file
    num_output_rows = bytes_in_gb // element_size // seq_len
    output_mmap = np.memmap(
        output_path, dtype=np.uint16, mode="w+", shape=(num_output_rows, seq_len)
    )
    output_mmap[:num_output_rows] = token_mmap[:num_output_rows]
    output_mmap.flush()


if __name__ == "__main__":
    main()
