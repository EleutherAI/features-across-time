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
    df[conf_top_col] = df[mean_col] + (df[conf_top_col] - df[mean_col]) / np.sqrt(
        num_items
    )
    df[conf_bottom_col] = df[mean_col] - (df[mean_col] - df[conf_bottom_col]) / np.sqrt(
        num_items
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


def plot_ngram_model():
    with open(Path.cwd() / "output" / "checkpoint_ngrams_model.pkl", "rb") as f:
        ngram_df = pickle.load(f)

    df = ngram_df.groupby("step").mean().reset_index()
    df["mean_bigram_bpb"] = df["mean_bigram_loss"] * 0.3366084909549386 / math.log(2)
    df["mean_bigram_bpb_conf_bottom"] = (
        df["bottom_conf_bigram_loss"] * 0.3366084909549386 / math.log(2)
    )
    df["mean_bigram_bpb_conf_top"] = (
        df["top_conf_bigram_loss"] * 0.3366084909549386 / math.log(2)
    )

    df = adjust_confidence_intervals(
        df, "mean_bigram_bpb", "mean_bigram_bpb_conf_bottom", "mean_bigram_bpb_conf_top"
    )

    fig = px.line(df, x="step", y="mean_bigram_bpb")
    fig.update_layout(
        {
            "title": "Mean BPB on bigram sequences over training steps.",
            "yaxis_title": "Bits per byte",
            "xaxis_title": "Training step (1 step = 2,097,152 tokens)",
            "yaxis_range": [3, 6],
        }
    )
    fig.add_traces(
        go.Scatter(
            x=df["step"],
            y=df["mean_bigram_bpb_conf_top"],
            fill=None,
            mode="lines",
            line=dict(color="powderblue"),
            showlegend=False,
        )
    )
    fig.add_traces(
        go.Scatter(
            x=df["step"],
            y=df["mean_bigram_bpb_conf_bottom"],
            fill="tonexty",
            mode="lines",
            line=dict(color="powderblue"),
            showlegend=False,
        )
    )

    tick_values, tick_texts = base_2_log_ticks(df["step"])
    fig.update_xaxes(type="log", tickvals=tick_values, ticktext=tick_texts)
    fig.write_image(Path.cwd() / "images" / "bigram_data_bpb.png")

    df = ngram_df.groupby("step").mean().reset_index()
    df["mean_unigram_bpb"] = df["mean_unigram_loss"] * 0.3366084909549386 / math.log(2)
    df["mean_unigram_bpb_conf_bottom"] = (
        df["bottom_conf_unigram_loss"] * 0.3366084909549386 / math.log(2)
    )
    df["mean_unigram_bpb_conf_top"] = (
        df["top_conf_unigram_loss"] * 0.3366084909549386 / math.log(2)
    )

    df = adjust_confidence_intervals(
        df,
        "mean_unigram_bpb",
        "mean_unigram_bpb_conf_bottom",
        "mean_unigram_bpb_conf_top",
    )

    fig = px.line(df, x="step", y="mean_unigram_bpb")
    fig.update_layout(
        {
            "title": "BPB on unigram sequences over training steps.",
            "yaxis_title": "Bits per byte",
            "xaxis_title": "Training step (1 step = 2,097,152 tokens)",
            "yaxis_range": [3, 6],
        }
    )
    fig.add_traces(
        go.Scatter(
            x=df["step"],
            y=df["mean_unigram_bpb_conf_top"],
            fill=None,
            mode="lines",
            line=dict(color="powderblue"),
            showlegend=False,
        )
    )
    fig.add_traces(
        go.Scatter(
            x=df["step"],
            y=df["mean_unigram_bpb_conf_bottom"],
            fill="tonexty",
            mode="lines",
            line=dict(color="powderblue"),
            showlegend=False,
        )
    )

    tick_values, tick_texts = base_2_log_ticks(df["step"])
    fig.update_xaxes(type="log", tickvals=tick_values, ticktext=tick_texts)
    fig.write_image(Path.cwd() / "images" / "unigram_data_bpb.png")


def plot_divs():
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
    plot_ngram_model()
    plot_divs()
    plot_shuffled_loss()


if __name__ == "__main__":
    main()
