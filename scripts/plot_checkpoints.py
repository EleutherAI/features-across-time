import math
import pickle
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go


def plot_shuffled_loss():
    output_path = Path.cwd() / "output" / "checkpoint_shuffled.pkl"
    with open(output_path, "rb") as f:
        df = pickle.load(f)

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

    grouped = df.groupby("step")
    agg_df = grouped.mean()[
        [
            "token_bow_mean_losses",
            "token_bottom_conf_intervals",
            "token_top_conf_intervals",
        ]
    ]
    agg_df.reset_index(inplace=True)
    agg_df.columns = ["step", "mean_agg_loss", "mean_conf_bottom", "mean_conf_top"]

    fig = px.line(agg_df, x="step", y="mean_agg_loss")
    fig.update_layout(
        {
            "title": "Mean loss on shuffled sequences over training steps.",
            "yaxis_title": "Mean loss",
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


def plot_bigram_model():
    with open(Path.cwd() / "output" / "checkpoint_bigrams_model.pkl", "rb") as f:
        bigram_df = pickle.load(f)

    grouped = bigram_df.groupby("step")
    agg_df = grouped.mean()[
        [
            "mean_bigram_losses",
            "bigram_bottom_conf_intervals",
            "bigram_top_conf_intervals",
        ]
    ]
    agg_df.reset_index(inplace=True)
    agg_df.columns = ["step", "mean_agg_loss", "mean_conf_bottom", "mean_conf_top"]

    fig = px.line(agg_df, x="step", y="mean_agg_loss")
    fig.update_layout(
        {
            "title": "Mean loss on bigram sequences over training steps.",
            "yaxis_title": "Mean loss",
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
    fig.update_xaxes(type="log")
    fig.write_image(Path.cwd() / "images" / "line_bigram_losses.png")

    fig = go.Figure(
        [
            go.Heatmap(
                {
                    "x": bigram_df["step"],
                    "y": bigram_df["index"],
                    "z": bigram_df["mean_bigram_losses"],
                }
            )
        ]
    )
    fig.update_layout(
        {
            "title": "Mean losses on auto-regressively sampled bigrams \
                over training steps",
            "yaxis_title": "Token index",
            "xaxis_title": "Training step (1 step = 2,097,152 tokens)",
        }
    )
    fig.update_xaxes(type="log")
    fig.write_image(Path.cwd() / "images" / "mean_bigram_losses.png")


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
    plot_bigram_model()
    plot_bigram_divs()
    plot_shuffled_loss()


if __name__ == "__main__":
    main()
