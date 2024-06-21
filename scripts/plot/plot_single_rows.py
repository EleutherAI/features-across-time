import os
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plot_ngram import (
    base_2_log_ticks,
    get_confidence_intervals,
    hex_to_rgba,
    kaleido_workaround,
    marker_series,
)
from plotly.subplots import make_subplots


def plot_loss_and_divergences(
    df: pd.DataFrame,
    loss_image_name: str,
    divergence_image_name: str,
    num_samples: int,
    bpb_coefficient=0.3650388,
    entropies=[2.89, 2.04],
    qualitative=False,
):
    kaleido_workaround()

    fig = make_subplots(
        rows=1,
        cols=4,
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=[
            "1-gram sequence loss across time",
            "2-gram sequence loss across time",
            "3-gram sequence loss across time",
            "4-gram sequence loss across time",
        ],
        horizontal_spacing=0.02,
        vertical_spacing=0.05,
    )
    tick_values, tick_texts = base_2_log_ticks(df["step"], spacing=2)

    for idx, ngram in enumerate(["1_gram", "2_gram", "3_gram", "4_gram"]):
        df[f"bottom_conf_{ngram}_loss"] = df[f"mean_{ngram}_loss"].map(
            lambda x: get_confidence_intervals(x, num_samples)[0]
        )
        df[f"top_conf_{ngram}_loss"] = df[f"mean_{ngram}_loss"].map(
            lambda x: get_confidence_intervals(x, num_samples)[1]
        )

        df[f"mean_{ngram}_bpb"] = df[f"mean_{ngram}_loss"] * bpb_coefficient
        df[f"bottom_conf_{ngram}_bpb"] = (
            df[f"bottom_conf_{ngram}_loss"] * bpb_coefficient
        )
        df[f"top_conf_{ngram}_bpb"] = df[f"top_conf_{ngram}_loss"] * bpb_coefficient

        for i, model in enumerate(df["pretty_model_name"].unique()):
            df_model = df[df["pretty_model_name"] == model]
            color = (
                px.colors.sequential.Plasma_r[i]
                if qualitative is False
                else px.colors.qualitative.Plotly[i]
            )
            transparent_color = hex_to_rgba(color, opacity=0.17)

            fig.add_trace(
                go.Scatter(
                    x=df_model["step"],
                    y=df_model[f"top_conf_{ngram}_bpb"],
                    fill=None,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=idx + 1,
            )
            fig.add_trace(
                go.Scatter(
                    x=df_model["step"],
                    y=df_model[f"bottom_conf_{ngram}_bpb"],
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=transparent_color,
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=idx + 1,
            )
            fig.add_trace(
                go.Scatter(
                    x=df_model["step"],
                    y=df_model[f"mean_{ngram}_bpb"],
                    mode="lines+markers",
                    marker=dict(size=5, symbol=marker_series[i]),
                    name=model,
                    line=dict(color=color),
                    showlegend=idx == 1,
                ),
                row=1,
                col=idx + 1,
            )

        if idx < 2:
            entropy = entropies[idx]
            fig.add_shape(
                type="line",
                x0=1,
                y0=entropy,
                x1=2**17,
                y1=entropy,
                line=dict(color="black", width=2, dash="dot"),
                row=1,
                col=idx + 1,
            )

    fig.update_layout(
        width=1200,
        height=400,
        legend=dict(
            x=0.98,
            y=0.65,
            xanchor="right",
            yanchor="middle",
            font=dict(size=8),
            title="Pythia loss",
            bgcolor="rgba(255, 255, 255, 0.5)",
        ),
        yaxis=dict(range=[1.5, 5.2]),
        margin=dict(l=20, r=20, t=50, b=60),
    )

    fig.update_xaxes(
        title_text="", type="log", tickvals=tick_values, ticktext=tick_texts
    )
    fig.update_yaxes(
        title_text="Loss", title_font=dict(size=12), title_standoff=10, row=1, col=1
    )
    fig.add_annotation(
        dict(
            text="Training step",
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.2,
            showarrow=False,
            font=dict(size=14),
        )
    )

    fig.write_image(loss_image_name, format="pdf")

    fig = make_subplots(
        rows=1,
        cols=2,
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=[
            "D<sub>KL</sub>(unigram model || Pythia) across time",
            "D<sub>KL</sub>(bigram model || Pythia) across time",
        ],
        horizontal_spacing=0.02,
        vertical_spacing=0.05,
    )

    div_metadata = [
        ("1_gram_kl_div", 1, 1),
        ("2_gram_kl_div", 1, 2),
    ]
    for label, row, col in div_metadata:
        df[f"bottom_conf_{label}"] = df[f"mean_{label}"].map(
            lambda x: get_confidence_intervals(x, num_samples)[0]
        )
        df[f"top_conf_{label}"] = df[f"mean_{label}"].map(
            lambda x: get_confidence_intervals(x, num_samples)[1]
        )

        df[f"mean_{label}_bpb"] = df[f"mean_{label}"] * bpb_coefficient
        df[f"bottom_conf_{label}_bpb"] = df[f"bottom_conf_{label}"] * bpb_coefficient
        df[f"top_conf_{label}_bpb"] = df[f"top_conf_{label}"] * bpb_coefficient

        for i, model in enumerate(df["pretty_model_name"].unique()):
            df_model = df[df["pretty_model_name"] == model]
            color = (
                px.colors.sequential.Plasma_r[i]
                if qualitative is False
                else px.colors.qualitative.Plotly[i]
            )
            transparent_color = hex_to_rgba(color, opacity=0.17)
            fig.add_trace(
                go.Scatter(
                    x=df_model["step"],
                    y=df_model[f"top_conf_{label}_bpb"],
                    fill=None,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=df_model["step"],
                    y=df_model[f"bottom_conf_{label}_bpb"],
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=transparent_color,
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=df_model["step"],
                    y=df_model[f"mean_{label}_bpb"],
                    mode="lines+markers",
                    marker=dict(size=5),
                    name=model,
                    line=dict(color=color),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

    fig.update_layout(
        width=1000,
        height=400,
        # legend=dict(
        #     x=0.98,
        #     y=0.6,
        #     xanchor='right',
        #     yanchor='middle',
        #     font=dict(size=8),
        #     title="Pythia loss",
        #     bgcolor='rgba(255, 255, 255, 0.5)'
        # ),
        margin=dict(l=20, r=20, t=50, b=60),
    )

    fig.update_xaxes(
        title_text="", type="log", tickvals=tick_values, ticktext=tick_texts
    )
    fig.update_yaxes(
        title_text="KL divergence",
        title_font=dict(size=12),
        title_standoff=10,
        row=1,
        col=1,
    )
    fig.add_annotation(
        dict(
            text="Training step",
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.2,
            showarrow=False,
            font=dict(size=14),
        )
    )

    fig.write_image(divergence_image_name, format="pdf")


def plot_suite(
    data_path: Path,
    images_path: Path,
    num_samples: int,
    bpb_coefficient: float,
    entropies: list[float],
):
    model_metadata = [
        ("pythia-14m", "14M"),
        ("pythia-70m", "70M"),
        ("pythia-160m", "160M"),
        ("pythia-410m", "410M"),
        ("pythia-1b", "1B"),
        ("pythia-1.4b", "1.4B"),
        ("pythia-2.8b", "2.8B"),
        # ("pythia-6.9b", "6.9B"),
        # ("pythia-12b", "12B"),
    ]
    model_dfs = []
    for model_name, pretty_model_name in model_metadata:
        model_df = pd.read_csv(data_path / f"ngram_{model_name}_{num_samples}.csv")
        model_df["model_name"] = model_name
        model_df["pretty_model_name"] = pretty_model_name
        model_dfs.append(model_df)
    df = pd.concat(model_dfs)

    loss_image_name = images_path / "ngram-loss.pdf"
    divergence_image_name = images_path / "ngram-divergence.pdf"
    plot_loss_and_divergences(
        df,
        loss_image_name,
        divergence_image_name,
        num_samples,
        bpb_coefficient,
        entropies,
    )


def plot_warmups(
    data_path: Path,
    images_path: Path,
    num_samples: int,
    bpb_coefficient: float,
    entropies: list[float],
):
    for model_size in [14, 70]:
        model_metadata = [
            (f"pythia-{model_size}m", f"{model_size}M (fast warmup)"),
            (f"pythia-{model_size}m-warmup01", f"{model_size}M (slow warmup)"),
        ]
        model_dfs = []
        for model_name, pretty_model_name in model_metadata:
            model_df = pd.read_csv(data_path / f"ngram_{model_name}_{num_samples}.csv")
            model_df["model_name"] = model_name
            model_df["pretty_model_name"] = pretty_model_name

            model_dfs.append(model_df)
        df = pd.concat(model_dfs)

        loss_name = images_path / f"warmups-{model_size}m-loss.pdf"
        divergence_name = images_path / f"warmups-{model_size}m-divergence.pdf"
        plot_loss_and_divergences(
            df,
            loss_name,
            divergence_name,
            num_samples,
            bpb_coefficient,
            entropies,
            qualitative=True,
        )


def main(data_path: Path, images_path: Path, num_samples: int):
    bpb_coefficient = 0.3650388
    entropies = [2.89, 2.04]

    os.makedirs(images_path, exist_ok=True)

    plot_suite(data_path, images_path, num_samples, bpb_coefficient, entropies)
    plot_warmups(data_path, images_path, num_samples, bpb_coefficient, entropies)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default="output")
    parser.add_argument("--images_path", type=str, default="images")
    parser.add_argument("--num_samples", type=int, default=1024)
    args = parser.parse_args()

    main(Path(args.data_path), Path(args.images_path), args.num_samples)
