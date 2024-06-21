import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scripts.plots.plot_ngram import (
    base_2_log_ticks,
    get_confidence_intervals,
    hex_to_rgba,
    kaleido_workaround,
    marker_series,
)


def main(
    model_name: str,
    images_path: Path,
    data_path: Path,
    num_samples: int,
    bpb_coefficient=0.3650388,
    entropies=[2.89, 2.04],
):
    os.makedirs(images_path, exist_ok=True)
    kaleido_workaround()

    df = pd.read_csv(data_path / f"{model_name}_{num_samples}_steps.csv")
    tick_values, tick_texts = base_2_log_ticks(df["index"])

    df["step"] = pd.to_numeric(df["step"], errors="coerce").fillna(0).astype(int)

    # Remove several steps because their lines' confidence intervals overlap
    df = df[df["step"] != 0]
    df = df[df["step"] != 2]
    df = df[df["step"] != 8]
    df = df[df["step"] != 32]
    df = df[df["step"] != 128]
    df = df[df["step"] != 512]
    df = df[df["step"] != 2048]
    df = df[df["step"] != 8000]
    df = df[df["step"] != 20_000]
    df = df[df["step"] != 80_000]
    df = df[df["step"] != 320_000]

    # Make index 1-indexed
    df["index"] = df["index"] + 1

    # Filter to log spaced samples
    max_index = df["index"].max()
    log_spaced_indices = np.unique(
        np.logspace(
            0, np.log2(max_index), num=int(np.log2(max_index)) + 1, base=2
        ).astype(int)
    )
    df = df[df["index"].isin(log_spaced_indices)]

    step_order = sorted(df["step"].unique())

    fig = make_subplots(
        rows=1,
        cols=1,
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=[
            "Trigram loss over sequence positions (Pythia 12B)",
            # "Unigram loss over sequence positions",
            # "Bigram loss over sequence positions",
        ],
        horizontal_spacing=0.02,
        vertical_spacing=0.05,
    )

    for idx, ngram in enumerate(["1-gram", "2-gram"]):  # "3-gram"
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

        # For some reason the legend items are listed in reverse order
        for i, step in enumerate(reversed(step_order)):
            group_df = df[df["step"] == step]
            color = px.colors.sequential.Plasma_r[i]
            transparent_color = hex_to_rgba(color, opacity=0.17)

            fig.add_trace(
                go.Scatter(
                    x=group_df["index"],
                    y=group_df[f"top_conf_{ngram}_bpb"],
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
                    x=group_df["index"],
                    y=group_df[f"bottom_conf_{ngram}_bpb"],
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
                    x=group_df["index"],
                    y=group_df[f"mean_{ngram}_bpb"],
                    mode="lines+markers",
                    marker=dict(size=5, symbol=marker_series[i]),
                    name=f"{step:,}",
                    line=dict(color=color),
                    showlegend=idx == 0,
                ),
                row=1,
                col=idx + 1,
            )

        # fig.add_shape(
        #     type="line",
        #     x0=1,
        #     y0=entropies[idx],
        #     x1=2**11,
        #     y1=entropies[idx],
        #     line=dict(color="black", width=2, dash="dot"),
        #     row=1,
        #     col=idx + 1,
        # )

    fig.update_layout(
        width=600,
        height=400,
        legend=dict(
            x=0.98,
            y=0.6,
            xanchor="right",
            yanchor="middle",
            font=dict(size=8),
            title="Step",
            bgcolor="rgba(255, 255, 255, 0.5)",
        ),
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
            text="Token index",
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.2,
            showarrow=False,
            font=dict(size=14),
        )
    )
    fig.write_image(
        images_path / f"in-context-EleutherAI--{model_name}.pdf", format="pdf"
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default="output")
    parser.add_argument("--images_path", type=str, default="images")
    parser.add_argument("--num_samples", type=int, default=1024)
    args = parser.parse_args()

    for model_name, batch_size in [
        # ("pythia-14m", 4),
        # ("pythia-70m", 4),
        # ("pythia-160m", 4),
        # ("pythia-410m", 4),
        # ("pythia-1b", 4),
        # ("pythia-1.4b", 4),
        # ("pythia-2.8b", 4),
        # ("pythia-6.9b", 1),
        ("pythia-12b", 1),
    ]:
        main(model_name, Path(args.images_path), Path(args.data_path), args.num_samples)
