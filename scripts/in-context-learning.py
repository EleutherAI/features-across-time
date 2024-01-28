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


# def kl_divergence(
#     logit_p: torch.Tensor, logit_q: torch.Tensor, dim: int = -1
# ) -> torch.Tensor:
#     """Compute the KL divergence between two sets of logits."""
#     logsumexp_p = logit_p.logsumexp(dim).unsqueeze(dim)
#     logsumexp_q = logit_q.logsumexp(dim).unsqueeze(dim)

#     return torch.nansum(
#         logit_p.sub(logsumexp_p).exp()
#         * (logit_p.sub(logsumexp_p) - logit_q.sub(logsumexp_q)),
#         dim,
#     )


# def js_divergence(
#     logit_p: torch.Tensor, logit_q: torch.Tensor, dim: int = -1
# ) -> torch.Tensor:
#     """Compute the Jensen-Shannon divergence between two sets of logits"""
#     logsumexp_p = logit_p.logsumexp(dim).unsqueeze(dim)
#     logsumexp_q = logit_q.logsumexp(dim).unsqueeze(dim)

#     # Mean of P and Q
#     log_m = (
#         torch.stack([logit_p - logsumexp_p, logit_q - logsumexp_q])
#         .sub(math.log(2))
#         .logsumexp(0)
#     )

#     kl_p = torch.nansum(
#         logit_p.sub(logsumexp_p).exp() * (logit_p.sub(logsumexp_p).sub(log_m)), dim
#     )
#     kl_q = torch.nansum(
#         logit_q.sub(logsumexp_q).exp() * (logit_q.sub(logsumexp_q).sub(log_m)), dim
#     )
#     return 0.5 * (kl_p + kl_q)


# def one_hot_js_divergence(
#     logit_q: torch.Tensor,
#     p_index: torch.Tensor,
#     batch: int,
#     seq_len: int,
#     dim: int = -1
# ) -> torch.Tensor:
#     logsumexp_q = logit_q.logsumexp(-1).unsqueeze(-1)

#     # accumulate log_m (starting in linear space)
#     log_m = logit_q.sub(logsumexp_q).sub(math.log(2)).exp()
#     log_m[torch.arange(batch * seq_len), p_index] += 0.5
#     log_m += torch.finfo(torch.float32).eps
#     log_m = log_m.log()

#     # p * log(p / m) at p = 1 -> log(p) - log(m) = -log(m)
#     kl_p = -log_m[torch.arange(batch * seq_len), p_index]
#     kl_q = torch.nansum(
#         logit_q.sub(logsumexp_q).exp() * (logit_q.sub(logsumexp_q).sub(log_m)), dim
#     )
#     return 0.5 * (kl_p + kl_q)


# def get_sequence_divergences(
#     tokens: torch.Tensor,
#     logits: torch.Tensor,
#     ngram_model: NgramModel,
#     batch: int,
#     d_vocab: int,
#     seq_len: int
# ) -> np.ndarray:
#     divergences = []
#     logits = logits[:, :-1, :d_vocab].flatten(
#         0, 1
#     )
#     sample = tokens[:, 1:].flatten()
#     bigram_dists = (
#         ngram_model.get_bigram_dists(tokens[:, :-1].flatten())
#         + torch.finfo(torch.float32).eps
#     )
#    divergences.append(one_hot_js_divergence(logits, sample, batch, seq_len)
#                       .reshape(batch, -1))
#     divergences.append(
#         one_hot_js_divergence(bigram_dists, sample, batch, seq_len).reshape(batch, -1)
#     )
#     del sample

#     divergences.append(kl_divergence(bigram_dists, logits).reshape(batch, -1))
#     divergences.append(
#         js_divergence(bigram_dists, logits).reshape(batch, -1)
#     )  # uses extra 32-25 GB - mem bottleneck. unigrams might too
#     del bigram_dists

# unigram_dist = ngram_model.unigrams + torch.finfo(torch.float32).eps
# divergences.append(kl_divergence(unigram_dist, logits).reshape(batch, -1))
# divergences.append(
#     js_divergence(unigram_dist.repeat(seq_len * batch, 1), logits
#                   ).reshape(batch, -1)
# )
# labels = [
#     "logit_token_js_div",
#     "bigram_token_js_div",
#     "bigram_logit_kl_div",
#     "bigram_logit_js_div",
#     "unigram_logit_kl_div",
#     "unigram_logit_js_div",
# ]
# return torch.stack(divergences), labels


@torch.inference_mode()
def multi_step_worker(
    gpu_id: int,
    model_name: str,
    team: str,
    ngram_path: str,
    pile_path: str,
    tmp_cache_path: str,
    num_samples: int,
    batch: int,
    seq_len: int,
    d_vocab: int,
    steps: list[str],
) -> pd.DataFrame:
    hf_model_name = f"{team}/{model_name}"

    tmp_cache_dir = f"{tmp_cache_path}/{gpu_id}"
    # shutil.rmtree(tmp_cache_dir, ignore_errors=True)
    os.makedirs(tmp_cache_dir, exist_ok=True)

    tokenizer = LlamaTokenizer.from_pretrained(f"{team}/{model_name}")
    ngram_model = NgramModel(ngram_path, tokenizer, batch=batch, seq_len=seq_len)

    labels = ["unigram_loss", "bigram_loss", "random_loss"]
    data = {f"mean_{label}": [] for label in labels}
    data.update({f"bottom_conf_{label}": [] for label in labels})
    data.update({f"top_conf_{label}": [] for label in labels})
    data["index"] = []
    data["step"] = []

    # div_labels = [
    #     "logit_token_js_div",
    #     "bigram_token_js_div",
    #     "bigram_logit_kl_div",
    #     "bigram_logit_js_div",
    #     "unigram_logit_kl_div",
    #     "unigram_logit_js_div",
    # ]
    # div_means = {label: [] for label in div_labels}
    # div_conf_intervals = {label: [] for label in div_labels}

    num_iters = math.ceil(num_samples / batch)
    pbar = tqdm.tqdm(total=len(steps) * num_iters, position=gpu_id)
    for step in steps:
        # pile = iter(DataLoader(load_from_disk(pile_path), batch_size=batch))
        model = LlamaForCausalLM.from_pretrained(
            hf_model_name,
            revision=f"ckpt_{step}",
            torch_dtype="auto",
            cache_dir=tmp_cache_dir,
        ).cuda()

        step_losses = {label: [] for label in labels}

        # step_div_means = torch.zeros(len(div_labels), seq_len - 1)
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

            # sample = next(pile)["input_ids"].cuda()[:, :-1].to(torch.int32)
            # print(sample.shape)
            # logits = model(sample).logits
            # divergences, _ = get_sequence_divergences(
            #     sample, logits, ngram_model, batch, d_vocab, seq_len - 1
            # )
            # step_div_means += (divergences / num_iters).cpu()
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

        # for i, label in enumerate(div_labels):
        #     mean_div_loss, mean_div_conf_intervals = positional_summary_stats(
        #         step_div_means[i], (seq_len - 1)
        #     )
        #     div_means[label].append(mean_div_loss)
        #     div_conf_intervals[label].append(mean_div_conf_intervals)

        shutil.rmtree(tmp_cache_dir, ignore_errors=True)
    pd.DataFrame(data)


def main(ngram_path: str, pile_path: str, tmp_cache_path: str):
    mp.set_start_method("spawn")

    # model_name = "Mistral-7B-v0.1"
    # team = "mistralai"
    # model_batch_sizes = {f"pythia-14m-seed{i}": 8 for i in range(1, 10)}
    # team = 'EleutherAI'
    team = "LLM360"
    model_batch_sizes = {"Amber": 1}
    tokenizer = LlamaTokenizer.from_pretrained("LLM360/Amber", revision="ckpt_356")
    vocab_size = tokenizer.vocab_size

    num_samples = 1024
    batch = 1
    seq_len = 2048
    # Amber steps go from 0 to 359. Assuming linearly spaced (not specified)
    steps = ["000"] + [f"{2**i:03}" for i in range(int(math.log2(359)) + 1)] + ["359"]
    print(steps)

    num_gpus = torch.cuda.device_count()
    max_steps_per_chunk = math.ceil(len(steps) / num_gpus)
    step_indices = [
        steps[i : i + max_steps_per_chunk]
        for i in range(0, len(steps), max_steps_per_chunk)
    ]

    for model_name, batch in model_batch_sizes.items():
        args = [
            (
                i,
                model_name,
                team,
                ngram_path,
                pile_path,
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
    parser.add_argument(
        "--pile_path",
        default="val_tokenized.hf",  # '/mnt/ssd-1/lucia/val_tokenized.hf',
        help="Path to Pile validation data",
    )
    parser.add_argument(
        "--tmp_cache_path",
        default=".cache",
        help="Path to cache which will be manually cleared",
    )
    args = parser.parse_args()
    main(args.ngram_path, args.pile_path, args.tmp_cache_path)
