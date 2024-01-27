import argparse
import math
import os
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import tqdm.auto as tqdm
from scipy import stats
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


class NgramModel:
    def __init__(
        self, path: str, encode_tokenizer: PreTrainedTokenizer, batch=1, seq_len=2048
    ):
        decode_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        self.decode_d_vocab = len(decode_tokenizer.vocab)
        self.transcoder = Transcoder(decode_tokenizer, encode_tokenizer, seq_len)
        self.batch = batch
        self.seq_len = seq_len

        with open(path, "rb") as f:
            bigram_counts = pickle.load(f)

        self.unigrams = (
            torch.tensor(bigram_counts.toarray().astype(np.float32)).sum(dim=1).cuda()
        )  # small
        self.bigrams = torch.sparse_csr_tensor(
            bigram_counts.indptr.astype(np.int64),
            bigram_counts.indices.astype(np.int64),
            bigram_counts.data.astype(np.float32),
            dtype=torch.float32,
            device="cuda",
        )  # 0.7 GB

    def generate_unigrams(self) -> torch.Tensor:
        tokens = torch.multinomial(
            self.unigrams, self.batch * self.seq_len, replacement=True
        ).reshape(self.batch, self.seq_len)
        return self.transcoder.transcode(tokens)

    # TODO separate slice sparse array function and add tests
    def sample_bigram(self, prev: torch.Tensor) -> torch.Tensor:
        """Given a batch of previous tokens, sample from a bigram model
        using conditional distributions stored in a sparse CSR tensor."""
        starts = self.bigrams.crow_indices()[prev]
        ends = self.bigrams.crow_indices()[prev + 1]

        # Pad rows with variable numbers of non-zero elements (0s are ignored)
        token_probs = torch.zeros(
            (self.batch, self.decode_d_vocab), dtype=torch.float32, device="cuda"
        )
        token_col_indices = torch.zeros(
            (self.batch, self.decode_d_vocab), dtype=torch.int32, device="cuda"
        )
        for i in range(self.batch):
            token_probs[i, : ends[i] - starts[i]] = self.bigrams.values()[
                starts[i] : ends[i]
            ]
            token_col_indices[i, : ends[i] - starts[i]] = self.bigrams.col_indices()[
                starts[i] : ends[i]
            ]

        sampled_value_indices = torch.multinomial(token_probs, 1)
        return torch.gather(token_col_indices, 1, sampled_value_indices)

    def generate_bigrams(self) -> torch.Tensor:
        """Auto-regressively generate bigram model sequence. Initialize each
        sequence by sampling from a unigram model."""
        result = [
            torch.multinomial(self.unigrams, self.batch, replacement=True).unsqueeze(1)
        ]
        for _ in range(self.seq_len - 1):
            prev = result[-1]
            result.append(self.sample_bigram(prev))

        result = torch.cat(result, dim=-1)
        return self.transcoder.transcode(result)


class Transcoder:
    def __init__(
        self, decoder: PreTrainedTokenizer, encoder: PreTrainedTokenizer, seq_len: int
    ):
        self.decoder = decoder
        self.encoder = encoder
        self.seq_len = seq_len

    def transcode(self, tokens: torch.Tensor):
        result = []
        for i in range(len(tokens)):
            token_strs = self.decoder.decode(tokens[i].tolist())
            encoded_tokens = torch.tensor(
                self.encoder.encode(token_strs), device="cuda"
            )[: self.seq_len]
            result.append(encoded_tokens)
            assert (
                len(result[-1]) == self.seq_len
            ), "Transcoded tokens too short; increase input seq_length"
        return torch.stack(result)


def split_by_eod(
    data: torch.Tensor,
    eod_indices: list[torch.Tensor],
) -> list[np.ndarray]:
    result = []
    start_idx = 0
    for i in range(data.shape[0]):
        for zero_idx in eod_indices[i]:
            if zero_idx > start_idx:
                result.append(data[i, start_idx : zero_idx + 1].cpu().numpy())
            start_idx = zero_idx + 1
        if start_idx < len(data):
            result.append(data[i, start_idx:].cpu().numpy())
    return result


def positional_summary_stats(
    vectors: list[np.ndarray], target_length: int
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    padded_vectors = [
        np.pad(v, (0, target_length - len(v)), constant_values=np.nan) for v in vectors
    ]
    stacked_vectors = np.vstack(padded_vectors).astype(np.float64)
    means = np.nanmean(stacked_vectors, axis=0)
    standard_errors = stats.sem(stacked_vectors, axis=0, nan_policy="omit")
    conf_intervals = stats.norm.interval(0.95, loc=means, scale=standard_errors)

    return means, conf_intervals


def get_sequence_losses(
    model: PreTrainedModel,
    sample: torch.Tensor,
    batch: int,
    seq_len: int,
    eod_token_index: int,
) -> list[np.ndarray]:
    """Get sequence losses. Start a new sequence at each EOD token."""
    outputs = model(sample)
    eod_indices = [
        torch.where(sample[i] == eod_token_index)[0] for i in range(len(sample))
    ]
    loss = F.cross_entropy(
        outputs.logits[:, :-1].reshape(batch * (seq_len - 1), -1),
        sample[:, 1:].reshape(batch * (seq_len - 1)),
        reduction="none",
    ).reshape(batch, seq_len - 1)
    return split_by_eod(loss, eod_indices)


@torch.inference_mode()
def multi_step_worker(
    gpu_id: int,
    model_name: str,
    team: str,
    ngram_path: str,
    tmp_cache_path: str,
    num_samples: int,
    batch: int,
    seq_len: int,
    d_vocab: int,
    steps: list[str],
) -> pd.DataFrame:
    hf_model_name = f"{team}/{model_name}"

    tmp_cache_dir = f"{tmp_cache_path}/{gpu_id}"
    shutil.rmtree(tmp_cache_dir, ignore_errors=True)
    os.makedirs(tmp_cache_dir, exist_ok=True)

    tokenizer = LlamaTokenizer.from_pretrained(f"{team}/{model_name}")
    ngram_model = NgramModel(ngram_path, tokenizer, batch=batch, seq_len=seq_len)

    labels = ["unigram_loss", "bigram_loss", "random_loss"]
    data = {f"mean_{label}": [] for label in labels}
    data.update({f"bottom_conf_{label}": [] for label in labels})
    data.update({f"top_conf_{label}": [] for label in labels})
    data["index"] = []
    data["step"] = []

    num_iters = math.ceil(num_samples / batch)
    pbar = tqdm.tqdm(total=len(steps) * num_iters, position=gpu_id)
    for step in steps:
        model = LlamaForCausalLM.from_pretrained(
            hf_model_name,
            revision=f"ckpt_{step}",
            torch_dtype="auto",
            cache_dir=tmp_cache_dir,
        ).cuda()

        step_losses = {label: [] for label in labels}
        generators = [
            ngram_model.generate_unigrams,
            ngram_model.generate_bigrams,
            lambda: torch.randint(0, d_vocab, [batch, seq_len], device="cuda"),
        ]
        for _ in range(num_iters):
            for label, generator in zip(labels, generators):
                sample = generator()
                losses = get_sequence_losses(
                    model, sample, batch, seq_len, eod_token_index=2
                )
                step_losses[label].extend(losses)

            torch.cuda.synchronize()
            pbar.update(1)

        for i in range(seq_len - 1):
            data["index"].append(i)
            data["step"].append(int(step))
        for label in labels:
            mean_loss, conf_intervals = positional_summary_stats(
                step_losses[label], seq_len - 1
            )
            data[f"mean_{label}"].extend(mean_loss)
            data[f"bottom_conf_{label}"].extend(conf_intervals[0])
            data[f"top_conf_{label}"].extend(conf_intervals[1])

        shutil.rmtree(tmp_cache_dir, ignore_errors=True)
    pd.DataFrame(data)


def main(ngram_path: str, tmp_cache_path: str):
    mp.set_start_method("spawn")

    team = "LLM360"
    model_batch_sizes = {"Amber": 1}
    tokenizer = LlamaTokenizer.from_pretrained("LLM360/Amber", revision="ckpt_356")
    vocab_size = tokenizer.vocab_size
    # model_name = "Mistral-7B-v0.1"
    # team = "mistralai"
    # model_batch_sizes = {f"pythia-14m-seed{i}": 8 for i in range(1, 10)}
    # team = 'EleutherAI'

    # Amber steps go from 0 to 359. Assuming linearly spaced (not specified)
    steps = ["000"] + [f"{2**i:03}" for i in range(int(math.log2(359)) + 1)] + ["359"]
    print(steps)

    num_gpus = torch.cuda.device_count()
    max_steps_per_chunk = math.ceil(len(steps) / num_gpus)
    step_indices = [
        steps[i : i + max_steps_per_chunk]
        for i in range(0, len(steps), max_steps_per_chunk)
    ]

    num_samples = 1024
    batch = 1
    seq_len = 2048
    for model_name, batch in model_batch_sizes.items():
        args = [
            (
                i,
                model_name,
                team,
                ngram_path,
                tmp_cache_path,
                num_samples,
                batch,
                seq_len,
                vocab_size,
                step_indices[i],
            )
            for i in range(len(step_indices))
        ]
        print(f"Parallelising over {len(step_indices)} GPUs...")
        with mp.Pool(len(step_indices)) as pool:
            dfs = pool.starmap(multi_step_worker, args)

    df = pd.concat(dfs)
    df.to_csv(
        Path.cwd() / "output" / f"{model_name}_{num_samples}_steps.csv",
        index=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--ngram_path",
        default="pythia-deduped-bigrams.pkl",  # /mnt/ssd-1/lucia/
        help="Path to pickled sparse scipy array of bigram counts over the Pile",
    )
    # parser.add_argument(
    #     "--pile_path",
    #     default="val_tokenized.hf",  # '/mnt/ssd-1/lucia/val_tokenized.hf',
    #     help="Path to Pile validation data",
    # )
    parser.add_argument(
        "--tmp_cache_path",
        default=".cache",
        help="Path to cache which will be manually cleared",
    )
    args = parser.parse_args()
    main(args.ngram_path, args.tmp_cache_path)
