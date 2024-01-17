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


def main():
    dfs = []
    for root, dirs, files in os.walk("output"):
        for file in files:
            with open(os.path.join(root, file), "rb") as f:
                dfs.append(pickle.load(f))

    df = pd.concat(dfs)

    heatmap = go.Heatmap(
        {
            "x": df["step"],
            "y": df["index"],
            "z": df["token_bow_mean_losses"] * 0.3366084909549386 / math.log(2),
        }
    )
    # print((df["token_bow_mean_losses"] * 0.3366084909549386 / math.log(2)).min())
    fig = go.Figure(
        data=[heatmap],
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
    agg_conf_intervals_bottom = []
    agg_conf_intervals_top = []

    for step in steps:
        mean_bow_loss = df[df["step"] == step]["token_bow_mean_losses"]
        token_bottom_conf_intervals = df[df["step"] == step][
            "token_bottom_conf_intervals"
        ]
        token_top_conf_intervals = df[df["step"] == step]["token_top_conf_intervals"]

        agg_means.append(mean_bow_loss.mean())
        agg_conf_intervals_bottom.append(token_bottom_conf_intervals.mean())
        agg_conf_intervals_top.append(token_top_conf_intervals.mean())

    agg_df = pd.DataFrame(
        {
            "step": steps,
            "mean_agg_loss": agg_means,
            "conf_bottom": agg_conf_intervals_bottom,
            "conf_top": agg_conf_intervals_top,
        }
    )
    agg_df.to_csv("agg_df.csv")

    # agg_df = pd.read_csv('agg_df.csv')

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
            y=agg_df["conf_top"],
            fill=None,
            mode="lines",
            line=dict(color="powderblue"),
            showlegend=False,
        )
    )
    fig.add_traces(
        go.Scatter(
            x=agg_df["step"],
            y=agg_df["conf_bottom"],
            fill="tonexty",
            mode="lines",
            line=dict(color="powderblue"),
            showlegend=False,
        )
    )
    fig.write_image(Path.cwd() / "images" / "agg_losses.png")


if __name__ == "__main__":
    main()


# def gradient_loss(means):
#     derivatives = np.gradient(means)

#     fig = px.line(derivatives)
#     fig.save_image("Derivatives")
#     # each point is (step, token index)
#     # its value is the derivative of the loss wrt the log(index)
#     # avg loss at each token index -> np.gradient(means)
