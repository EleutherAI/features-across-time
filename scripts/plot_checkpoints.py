import math
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats


def summary_stats(
    vectors: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    max_length = max(len(v) for v in vectors)
    padded_vectors = [
        np.pad(v, (0, max_length - len(v)), constant_values=np.nan) for v in vectors
    ]
    stacked_vectors = np.vstack(padded_vectors)

    means = np.nanmean(stacked_vectors, axis=0)
    standard_errors = stats.sem(stacked_vectors, axis=0, nan_policy="omit")
    conf_intervals = stats.norm.interval(0.95, loc=means, scale=standard_errors)

    return means, conf_intervals


def plot_shuffled_loss():
    dfs = []
    for r, _, fs in os.walk("output"):
        dfs.extend(
            [pickle.load(open(os.path.join(r, f), "rb")) for f in fs if "token" in f]
        )
    df = pd.concat(dfs)
    print(df.head())

    fig = go.Figure(
        [
            go.Heatmap(
                {
                    "x": df["step"],
                    "y": df["index"],
                    "z": df["token_bow_mean_losses"] * 0.3366084909549386 / math.log(2),
                }
            )
        ]
    )
    fig.update_layout(
        {
            "title": "Mean BPB on shuffled sequences over training steps",
            "yaxis_title": "Token index",
            "xaxis_title": "Training step (1 step = 2,097,152 tokens)",
        }
    )
    fig.write_image(Path.cwd() / "images" / "token_losses.png")

    # log_steps = [0] + [2**i for i in range(int(math.log2(512)) + 1)]
    linear_steps = [i for i in range(1000, 144000, 1000)]
    steps = linear_steps  # log_steps +

    agg_means = []
    mean_conf_bottoms = []
    mean_conf_tops = []
    for step in steps:
        mean_bow_loss = df[df["step"] == step]["token_bow_mean_losses"]
        mean_conf_bottoms.append(
            df[df["step"] == step]["token_bottom_conf_intervals"].mean()
        )
        mean_conf_tops.append(df[df["step"] == step]["token_top_conf_intervals"].mean())
        agg_means.append(mean_bow_loss.mean())

    agg_df = pd.DataFrame(
        {
            "step": steps,
            "mean_agg_loss": agg_means,
            "mean_conf_bottom": mean_conf_bottoms,
            "mean_conf_top": mean_conf_tops,
        }
    )

    num_sequences = len(df[df["step"] == 1000]["token_bow_mean_losses"].tolist())

    fig = px.line(agg_df, x="step", y="mean_agg_loss")
    fig.update_layout(
        {
            "title": "Mean loss on shuffled sequences over training steps.",
            "yaxis_title": f"Mean loss over {num_sequences} sequences",
            "xaxis_title": "Training step (1 step = 2,097,152 tokens)",
        }
    )
    fig.add_traces(
        go.Scatter(
            x=agg_df["step"],
            y=agg_df["mean_conf_top"],
            fill=None,
            mode="lines",
            line=dict(color="powderblue"),
            showlegend=False,
        )
    )
    fig.add_traces(
        go.Scatter(
            x=agg_df["step"],
            y=agg_df["mean_conf_bottom"],
            fill="tonexty",
            mode="lines",
            line=dict(color="powderblue"),
            showlegend=False,
        )
    )
    fig.write_image(Path.cwd() / "images" / "agg_losses.png")


def plot_bigram_divs():
    bigram_dfs = []
    for r, _, fs in os.walk("output"):
        bigram_dfs.extend(
            [pickle.load(open(os.path.join(r, f), "rb")) for f in fs if "bigram" in f]
        )
    bigram_df = pd.concat(bigram_dfs)
    print(bigram_df.head())

    div_labels = [
        "bigram_logit_kl_div",
        "logit_bigram_kl_div",
        "bigram_logit_js_div",
        "bigram_token_js_div",
        "logit_token_js_div",
    ]
    for div_label in div_labels:
        fig = go.Figure(
            [
                go.Heatmap(
                    {
                        "x": bigram_df["step"],
                        "y": bigram_df["index"],
                        "z": bigram_df[f"mean_{div_label}"],
                    }
                )
            ]
        )
        fig.update_layout(
            {
                "title": f"Mean divergence over training steps ({div_label})",
                "yaxis_title": "Token index",
                "xaxis_title": "Training step (1 step = 2,097,152 tokens)",
            }
        )
        fig.update_xaxes(type="log")
        fig.write_image(Path.cwd() / "images" / f"{div_label}.png")


def main():
    plot_bigram_divs()
    plot_shuffled_loss()


if __name__ == "__main__":
    main()
