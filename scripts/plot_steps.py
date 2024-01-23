import argparse
import math
import pickle
from pathlib import Path
import os

import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def base_2_log_ticks(values):
    max_val = np.log2(values.max())
    ticks = 2 ** np.arange(1, np.ceil(max_val) + 1)
    return ticks, ticks.astype(int)


def adjust_confidence_intervals(
    df, mean_col: str, bottom_conf_col: str, top_conf_col: str, sample_size=2048
):
    """Adjust confidence intervals for data averaged over token positions"""
    df[top_conf_col] = df[mean_col] + (df[top_conf_col] - df[mean_col]) / np.sqrt(
        sample_size
    )
    df[bottom_conf_col] = df[mean_col] - (df[mean_col] - df[bottom_conf_col]) / np.sqrt(
        sample_size
    )
    return df


def plot_shuffled_bpb():
    bpb_coefficient = 0.3366084909549386 / math.log(2)

    output_path = Path.cwd() / "output" / "step_shuffled.pkl"
    with open(output_path, "rb") as f:
        shuffled_df = pickle.load(f)

    df = shuffled_df.groupby("step").mean().reset_index()
    df["mean_bow_bpb"] = df["mean_bow_loss"] * bpb_coefficient
    df["mean_bow_bpb_bottom_conf"] = df["mean_bow_bottom_conf"] * bpb_coefficient
    df["mean_bow_bpb_top_conf"] = df["mean_bow_top_conf"] * bpb_coefficient
    df = adjust_confidence_intervals(
        df, "mean_bow_bpb", "mean_bow_bpb_bottom_conf", "mean_bow_bpb_top_conf"
    )

    fig = px.line(df, x="step", y="mean_bow_bpb")
    fig.update_layout(
        {
            "title": "Mean BPB on shuffled sequences over training steps",
            "yaxis_title": "Mean BPB",
            "xaxis_title": "Training step (1 step = 2,097,152 tokens)",
        }
    )
    fig.add_traces(
        go.Scatter(
            x=df["step"],
            y=df["mean_bow_bpb_top_conf"],
            fill=None,
            mode="lines",
            line=dict(color="powderblue"),
            showlegend=False,
        )
    )
    fig.add_traces(
        go.Scatter(
            x=df["step"],
            y=df["mean_bow_bpb_bottom_conf"],
            fill="tonexty",
            mode="lines",
            line=dict(color="powderblue"),
            showlegend=False,
        )
    )
    tick_values, tick_texts = base_2_log_ticks(df["step"])
    fig.update_xaxes(type="log", tickvals=tick_values, ticktext=tick_texts)
    fig.write_image(Path.cwd() / "images" / "shuffled_bpb.png")


def plot_ngram_model_bpb():
    model_names = [
        "pythia-14m",
        "pythia-70m",
        "pythia-160m",
        "pythia-410m",
        "pythia-1b",
        "pythia-1.4b",
        "pythia-2.8b",
        "pythia-6.9b",   
        "pythia-12b",   
    ]
    for model_name in model_names:

    # NN bias towards function of low frequency in the fourier domain

        with open(Path.cwd() / "output" / f"step_ngrams_model_means_{model_name}.pkl", "rb") as f:
            df = pickle.load(f)

        df.to_csv(Path.cwd() / "output" / f"means_ngrams_model_{model_name}.csv", index=False)

        bpb_coefficient = 0.3366084909549386 / math.log(2)
        df["mean_bigram_bpb"] = df["mean_bigram_loss"] * bpb_coefficient
        df["mean_bigram_bpb_bottom_conf"] = df["bottom_conf_bigram_loss"] * bpb_coefficient
        df["mean_bigram_bpb_top_conf"] = df["top_conf_bigram_loss"] * bpb_coefficient

        fig = px.line(df, x="step", y="mean_bigram_bpb")
        fig.update_layout(
            {
                "title": f"Mean BPB on bigram sequences over training steps ({model_name})",
                "yaxis_title": "Bits per byte",
                "xaxis_title": "Training step (1 step = 2,097,152 tokens)",
                "yaxis_range": [3, 12],
            }
        )
        fig.add_traces(
            go.Scatter(
                x=df["step"],
                y=df["mean_bigram_bpb_top_conf"],
                fill=None,
                mode="lines",
                line=dict(color="powderblue"),
                showlegend=False,
            )
        )
        fig.add_traces(
            go.Scatter(
                x=df["step"],
                y=df["mean_bigram_bpb_bottom_conf"],
                fill="tonexty",
                mode="lines",
                line=dict(color="powderblue"),
                showlegend=False,
            )
        )

        tick_values, tick_texts = base_2_log_ticks(df["step"])
        fig.update_xaxes(type="log", tickvals=tick_values, ticktext=tick_texts)
        fig.write_image(Path.cwd() / "images" / f"bigram_data_bpb_{model_name}.png")

        df["mean_unigram_bpb"] = df["mean_unigram_loss"] * bpb_coefficient
        df["mean_unigram_bpb_bottom_conf"] = (
            df["bottom_conf_unigram_loss"] * bpb_coefficient
        )
        df["mean_unigram_bpb_top_conf"] = df["top_conf_unigram_loss"] * bpb_coefficient

        fig = px.line(df, x="step", y="mean_unigram_bpb")
        fig.update_layout(
            {
                "title": f"BPB on unigram sequences over training steps ({model_name})",
                "yaxis_title": "Bits per byte",
                "xaxis_title": "Training step (1 step = 2,097,152 tokens)",
                "yaxis_range": [3, 12],
            }
        )
        fig.add_traces(
            go.Scatter(
                x=df["step"],
                y=df["mean_unigram_bpb_top_conf"],
                fill=None,
                mode="lines",
                line=dict(color="powderblue"),
                showlegend=False,
            )
        )
        fig.add_traces(
            go.Scatter(
                x=df["step"],
                y=df["mean_unigram_bpb_bottom_conf"],
                fill="tonexty",
                mode="lines",
                line=dict(color="powderblue"),
                showlegend=False,
            )
        )

        tick_values, tick_texts = base_2_log_ticks(df["step"])
        fig.update_xaxes(type="log", tickvals=tick_values, ticktext=tick_texts)
        fig.write_image(Path.cwd() / "images" / f"unigram_data_bpb_{model_name}.png")


def plot_divs():
    with open(Path.cwd() / "output" / "step_divergences.pkl", "rb") as f:
        divergences_df = pickle.load(f)

    div_labels = [
        "bigram_logit_kl_div",
        "unigram_logit_kl_div",
        "unigram_logit_js_div",
        "bigram_logit_js_div",
        "bigram_token_js_div",
        "logit_token_js_div",
    ]

    df = divergences_df.groupby("step").mean().reset_index()
    df.to_csv(Path.cwd() / "output" / "mean_divergences.csv", index=False)

    tick_values, tick_texts = base_2_log_ticks(df["step"])

    for label in div_labels:
        df = adjust_confidence_intervals(
            df, f"mean_{label}", f"bottom_conf_{label}", f"top_conf_{label}"
        )

        fig = px.line(df, x="step", y=f"mean_{label}")
        fig.update_layout(
            {
                "title": f"Mean divergence over training steps ({label})",
                "yaxis_title": "Mean divergence",
                "xaxis_title": "Training step (1 step = 2,097,152 tokens)",
            }
        )
        fig.add_traces(
            go.Scatter(
                x=df["step"],
                y=df[f"top_conf_{label}"],
                fill=None,
                mode="lines",
                line=dict(color="powderblue"),
                showlegend=False,
            )
        )
        fig.add_traces(
            go.Scatter(
                x=df["step"],
                y=df[f"bottom_conf_{label}"],
                fill="tonexty",
                mode="lines",
                line=dict(color="powderblue"),
                showlegend=False,
            )
        )
        fig.update_xaxes(type="log", tickvals=tick_values, ticktext=tick_texts)
        fig.write_image(Path.cwd() / "images" / f"{label}.png")


def main(divergences: bool, ngram_samples: bool, shuffled_samples: bool):
    os.makedirs(Path.cwd() / 'images', exist_ok=True)
    if divergences:
        plot_divs()
    if ngram_samples:
        plot_ngram_model_bpb()
    if shuffled_samples:
        plot_shuffled_bpb()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--divergences", action="store_true")
    parser.add_argument("--ngram_samples", action="store_true")
    parser.add_argument("--shuffled_samples", action="store_true")

    args = parser.parse_args()
    main(args.divergences, args.ngram_samples, args.shuffled_samples)
