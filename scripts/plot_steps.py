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
    bpb_coefficient = 0.3366084909549386 / math.log(2)

    with open(Path.cwd() / "output" / "step_ngrams_model.pkl", "rb") as f:
        ngram_df = pickle.load(f)

    df = ngram_df.groupby("step").mean().reset_index()
    df.to_csv(Path.cwd() / "output" / "means_ngrams_model.csv", index=False)

    df["mean_bigram_bpb"] = df["mean_bigram_loss"] * bpb_coefficient
    df["mean_bigram_bpb_bottom_conf"] = df["bottom_conf_bigram_loss"] * bpb_coefficient
    df["mean_bigram_bpb_top_conf"] = df["top_conf_bigram_loss"] * bpb_coefficient

    df = adjust_confidence_intervals(
        df, "mean_bigram_bpb", "mean_bigram_bpb_bottom_conf", "mean_bigram_bpb_top_conf"
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
    fig.write_image(Path.cwd() / "images" / "bigram_data_bpb.png")

    df = ngram_df.groupby("step").mean().reset_index()
    df["mean_unigram_bpb"] = df["mean_unigram_loss"] * bpb_coefficient
    df["mean_unigram_bpb_bottom_conf"] = (
        df["bottom_conf_unigram_loss"] * bpb_coefficient
    )
    df["mean_unigram_bpb_top_conf"] = df["top_conf_unigram_loss"] * bpb_coefficient

    df = adjust_confidence_intervals(
        df,
        "mean_unigram_bpb",
        "mean_unigram_bpb_bottom_conf",
        "mean_unigram_bpb_top_conf",
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
    fig.write_image(Path.cwd() / "images" / "unigram_data_bpb.png")


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


def main():
    plot_ngram_model_bpb()
    plot_divs()
    plot_shuffled_bpb()


if __name__ == "__main__":
    main()
