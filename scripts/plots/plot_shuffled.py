import math
import os
import pickle
from pathlib import Path

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
            "title": "Mean BPB on shuffled sequences across time",
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
    fig.write_image(Path.cwd() / "images" / "shuffled_bpb.pdf", format="pdf")


def main():
    os.makedirs(Path.cwd() / "images", exist_ok=True)
    plot_shuffled_bpb()


if __name__ == "__main__":
    main()
