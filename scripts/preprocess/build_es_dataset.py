# !pip install mwparserfromhell
import math
import os
from argparse import ArgumentParser
from chunk import chunk_and_tokenize
from pathlib import Path

import numpy as np
from datasets import DownloadConfig, load_dataset, load_from_disk
from transformers import AutoTokenizer


def bpb_ratio(data):
    """
    Compute the ratio used to convert loss from nats per token to bits per
    utf-8 encoded byte.
    Loss in nats to loss in bits: loss = log_2(exp(loss))
    Loss per token to loss per byte: loss *= (total_tokens / total_bytes)

    total_tokens / total_bytes * log2(exp(loss)) =
        loss * (total_tokens / total_bytes / ln(2))
    """
    total_bytes: float = sum(data["bytes"])
    total_tokens: float = sum(data["length"])
    return total_tokens / total_bytes / math.log(2)


def get_num_elements(data_path: Path):
    return os.path.getsize(data_path) // np.dtype(np.uint16).itemsize


def main(dataset_str: str, output_dir: Path):
    # dataset = load_dataset(
    #     dataset_str,
    #     "es",
    #     split='train',
    #     download_config=DownloadConfig(resume_download=True),
    #     cache_dir=Path('/') / 'mnt' / 'ssd-1' / 'hf_cache'))

    dataset = load_dataset(
        dataset_str,
        split="train",
        download_config=DownloadConfig(resume_download=True),
        cache_dir=Path("/") / "mnt" / "ssd-1" / "hf_cache",
    )

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-14m")

    data, bpb_ratio = chunk_and_tokenize(dataset.shuffle(), tokenizer, max_length=2049)
    data.save_to_disk(output_dir / "es_tokenized.hf")

    fp = np.memmap(
        output_dir / "es.bin", dtype=np.uint16, mode="w+", shape=(len(data), 2049)
    )
    for i, item in enumerate(data):
        fp[i] = item["input_ids"].numpy()
    fp.flush()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="spanish_billion_words",
        help="Name of HuggingFace dataset to be processed",
    )
    parser.add_argument(
        "--output", type=str, default="data", help="Path to store processed dataset"
    )
    args = parser.parse_args()

    main(args.dataset, Path(args.output))

    # Stats
    data = load_from_disk(str(args.output / "es_tokenized.hf"))
    print(bpb_ratio(data))  # 0.4157027
    print(get_num_elements(args.output / "es.bin"))  # 2677323801, or 2.6 billion tokens
