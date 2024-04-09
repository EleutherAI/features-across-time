import argparse
import os
from pathlib import Path

import pandas as pd
import torch
import tqdm.auto as tqdm
from transformers import AutoTokenizer
from script_utils.ngram_model import NgramModel
from script_utils.experiment import Experiment
from script_utils.load_model import get_auto_model, get_auto_tokenizer
from ngram_indices_steps import positional_summary_stats, get_sequence_losses, encode


def generate_random(d_vocab, batch, seq_len) -> torch.Tensor:
    return torch.randint(0, d_vocab, [batch, seq_len], device="cuda")


@torch.inference_mode()
def worker(
    experiment: Experiment,
    ngram_path: str,
) -> pd.DataFrame:
    tmp_cache_dir = Path(".cache")
    os.makedirs(tmp_cache_dir, exist_ok=True)

    tokenizer = experiment.get_tokenizer(experiment.team, experiment.model_name)
    ngram_model = NgramModel(
        ngram_path, batch=experiment.batch_size, seq_len=experiment.seq_len
    )

    use_encode = not (
        isinstance(tokenizer, AutoTokenizer)
        and NgramModel.tokenizer.name_or_path == tokenizer.name_or_path
    )
    if not use_encode:
        del tokenizer


    model = experiment.get_model(experiment.team, experiment.model_name, 0, tmp_cache_dir)
    token_indices = []
    labels = ["unigram_loss", "bigram_loss", "random_loss"]
    means = {label: [] for label in labels}
    bottom_conf_intervals = {label: [] for label in labels}
    top_conf_intervals = {label: [] for label in labels}

    unigram_losses = []
    bigram_losses = []
    random_losses = []
    for i in tqdm.tqdm(range(experiment.num_samples // experiment.batch_size)):
        bigram_sample = (
            encode(ngram_model.get_ngram_str(2, i), tokenizer, experiment.seq_len)
            if use_encode
            else ngram_model.get_ngram_seq(2, i)
        )
        bigram_losses.extend(get_sequence_losses(model, bigram_sample, experiment.batch_size, experiment.seq_len, experiment.eod_index))
        
        unigram_sample = (
            encode(ngram_model.get_ngram_str(1, i), tokenizer, experiment.seq_len)
            if use_encode
            else ngram_model.get_ngram_seq(1, i)
        )
        unigram_losses.extend(
            get_sequence_losses(model, unigram_sample, experiment.batch_size, experiment.seq_len, experiment.eod_index)
        )

        random_sample = generate_random(experiment.d_vocab, experiment.batch_size, experiment.seq_len)
        random_losses.extend(get_sequence_losses(model, random_sample, experiment.batch_size, experiment.seq_len, experiment.eod_index))

    mean_unigram_loss, unigram_conf_intervals = positional_summary_stats(
        unigram_losses, target_length=experiment.seq_len
    )
    means["unigram_loss"].extend(mean_unigram_loss)
    bottom_conf_intervals["unigram_loss"].extend(unigram_conf_intervals[0])
    top_conf_intervals["unigram_loss"].extend(unigram_conf_intervals[1])

    mean_bigram_loss, bigram_conf_intervals = positional_summary_stats(
        bigram_losses, target_length=experiment.seq_len
    )
    means["bigram_loss"].extend(mean_bigram_loss)
    bottom_conf_intervals["bigram_loss"].extend(bigram_conf_intervals[0])
    top_conf_intervals["bigram_loss"].extend(bigram_conf_intervals[1])

    mean_random_loss, random_conf_intervals = positional_summary_stats(
        random_losses, target_length=experiment.seq_len
    )
    means["random_loss"].extend(mean_random_loss)
    bottom_conf_intervals["random_loss"].extend(random_conf_intervals[0])
    top_conf_intervals["random_loss"].extend(random_conf_intervals[1])

    token_indices.extend(list(range(experiment.seq_len - 1)))

    index_data = {"index": token_indices}
    mean_data = {f"mean_{label}": means[label] for label in labels}
    bottom_conf_data = {
        f"bottom_conf_{label}": bottom_conf_intervals[label] for label in labels
    }
    top_conf_data = {f"top_conf_{label}": top_conf_intervals[label] for label in labels}
    return pd.DataFrame(
        {**index_data, **mean_data, **bottom_conf_data, **top_conf_data}
    )


def main(ngram_path: str):
    experiment = Experiment(
        num_samples=1024,
        team="mistralai", 
        model_name="Mistral-7B-v0.1",
        batch_size=1,
        seq_len=2049, 
        steps=[0, 1, 2, 4, 8, 16, 256, 1000, 8000, 33_000, 66_000, 131_000, 143_000],
        d_vocab=50_277, # len(get_neo_tokenizer().vocab) 
        get_model=get_auto_model, 
        get_tokenizer=get_auto_tokenizer,
        eod_index=2 # tokenizer.eos_token_id
    )

    df = worker(experiment, ngram_path)
    df.to_csv(
        Path.cwd() / "output" / f"{experiment.model_name}_{experiment.num_samples}.csv",
        index=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--ngram_path",
        default="/mnt/ssd-1/lucia/pythia-deduped-bigrams.pkl",
        help="Path to pickled sparse scipy array of bigram counts over the Pile",
    )
    args = parser.parse_args()
    main(args.ngram_path)
