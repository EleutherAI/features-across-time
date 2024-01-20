import math
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
    df, mean_col, conf_bottom_col, conf_top_col, num_items=2048
):
    adjustment_factor = 1 / np.sqrt(num_items)
    df[conf_top_col] = (
        df[mean_col] + (df[conf_top_col] - df[mean_col]) * adjustment_factor
    )
    df[conf_bottom_col] = (
        df[mean_col] - (df[mean_col] - df[conf_bottom_col]) * adjustment_factor
    )
    return df


def plot_shuffled_loss():
    output_path = Path.cwd() / "output" / "checkpoint_shuffled.pkl"
    with open(output_path, "rb") as f:
        shuffled_df = pickle.load(f)

    df = shuffled_df.groupby("step").mean().reset_index()
    df.columns = [
        "step",
        "index",
        "mean_loss",
        "mean_conf_bottom",
        "mean_conf_top",
        "mean_bow_loss",
        "mean_bow_conf_bottom",
        "mean_bow_conf_top",
    ]
    df["mean_bow_bpb"] = df["mean_bow_loss"] * 0.3366084909549386 / math.log(2)
    df["mean_bow_bpb_conf_bottom"] = (
        df["mean_bow_conf_bottom"] * 0.3366084909549386 / math.log(2)
    )
    df["mean_bow_bpb_conf_top"] = (
        df["mean_bow_conf_top"] * 0.3366084909549386 / math.log(2)
    )
    df = adjust_confidence_intervals(
        df, "mean_bow_bpb", "mean_bow_bpb_conf_bottom", "mean_bow_bpb_conf_top"
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
            y=df["mean_bow_bpb_conf_top"],
            fill=None,
            mode="lines",
            line=dict(color="powderblue"),
            showlegend=False,
        )
    )
    fig.add_traces(
        go.Scatter(
            x=df["step"],
            y=df["mean_bow_bpb_conf_bottom"],
            fill="tonexty",
            mode="lines",
            line=dict(color="powderblue"),
            showlegend=False,
        )
    )
    tick_values, tick_texts = base_2_log_ticks(df["step"])
    fig.update_xaxes(type="log", tickvals=tick_values, ticktext=tick_texts)
    fig.write_image(Path.cwd() / "images" / "mean_shuffled_bpb.png")


def plot_bigram_model():
    with open(Path.cwd() / "output" / "checkpoint_bigrams_model.pkl", "rb") as f:
        bigram_df = pickle.load(f)

    df = bigram_df.groupby("step").mean().reset_index()
    df.columns = ["step", "index", "mean_loss", "mean_conf_bottom", "mean_conf_top"]
    df["mean_bpb"] = df["mean_loss"] * 0.3366084909549386 / math.log(2)
    df["mean_bpb_conf_bottom"] = (
        df["mean_conf_bottom"] * 0.3366084909549386 / math.log(2)
    )
    df["mean_bpb_conf_top"] = df["mean_conf_top"] * 0.3366084909549386 / math.log(2)

    df = adjust_confidence_intervals(
        df, "mean_bpb", "mean_bpb_conf_bottom", "mean_bpb_conf_top"
    )

    fig = px.line(df, x="step", y="mean_bpb")
    fig.update_layout(
        {
            "title": "Mean loss on bigram sequences over training steps.",
            "yaxis_title": "Mean loss",
            "xaxis_title": "Training step (1 step = 2,097,152 tokens)",
        }
    )
    fig.add_traces(
        go.Scatter(
            x=df["step"],
            y=df["mean_bpb_conf_top"],
            fill=None,
            mode="lines",
            line=dict(color="powderblue"),
            showlegend=False,
        )
    )
    fig.add_traces(
        go.Scatter(
            x=df["step"],
            y=df["mean_bpb_conf_bottom"],
            fill="tonexty",
            mode="lines",
            line=dict(color="powderblue"),
            showlegend=False,
        )
    )

    tick_values, tick_texts = base_2_log_ticks(df["step"])
    fig.update_xaxes(type="log", tickvals=tick_values, ticktext=tick_texts)
    fig.write_image(Path.cwd() / "images" / "bigram_data_bpb.png")


def plot_bigram_divs():
    with open(Path.cwd() / "output" / "checkpoint_bigrams.pkl", "rb") as f:
        bigram_df = pickle.load(f)

    div_labels = [
        "bigram_logit_kl_div",
        "unigram_logit_kl_div",
        "unigram_logit_js_div",
        "bigram_logit_js_div",
        "bigram_token_js_div",
        "logit_token_js_div",
    ]

    df = bigram_df.groupby("step").mean().reset_index()

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


def main():
    plot_bigram_model()
    plot_bigram_divs()
    plot_shuffled_loss()


if __name__ == "__main__":
    main()
