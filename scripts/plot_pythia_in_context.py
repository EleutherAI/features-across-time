import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plot_ngram import base_2_log_ticks, hex_to_rgba, write_garbage
from plotly.subplots import make_subplots


def add_steps(df, supplementary_path):
    if os.path.exists(supplementary_path):
        print("supplementary data detected, merging...")
        supplementary_df = pd.read_csv(supplementary_path)
        df = pd.concat([df, supplementary_df])
    return df


def main(debug: bool):
    if not debug:
        write_garbage()

    os.makedirs(Path.cwd() / "images", exist_ok=True)
    image_name = Path.cwd() / "images" / "in-context.pdf"

    num_samples = 1024
    bpb_coefficient = 0.3650388
    entropies = [2.89, 2.04]
    marker_series = [
        "circle",
        "square",
        "diamond",
        "cross",
        "x",
        "triangle-up",
        "triangle-down",
        "triangle-left",
        "triangle-right",
        "pentagon",
        "hexagon",
        "octagon",
        "star",
        "hexagram",
    ]

    df = pd.read_csv(Path.cwd() / "output" / f"pythia-12b_{num_samples}_steps.csv")
    supplementary_path = (
        Path.cwd() / "output" / f"pythia-12b_{num_samples}_steps_additional.csv"
    )
    df = add_steps(df, supplementary_path)
    tick_values, tick_texts = base_2_log_ticks(df["index"])

    # Add missing step column
    steps = [16, 256, 1000, 8000, 33_000, 66_000, 131_000, 143_000]
    group_numbers = df["index"].eq(0).cumsum() - 1
    df["step"] = group_numbers.apply(
        lambda x: steps[x] if x < len(steps) else steps[-1]
    )

    # Remove a step or two because the confidence intervals overlap too much
    df = df[df["step"] != 131_000]
    df = df[df["step"] != 33_000]

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

    # Order data
    step_order = sorted(df["step"].unique())

    fig = make_subplots(
        rows=1,
        cols=2,
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=[
            "Unigram loss over sequence positions",
            "Bigram loss over sequence positions",
        ],
        horizontal_spacing=0.02,
        vertical_spacing=0.05,
    )

    for idx, ngram in enumerate(["unigram", "bigram"]):
        df[f"mean_{ngram}_bpb"] = df[f"mean_{ngram}_loss"] * bpb_coefficient
        df[f"bottom_conf_{ngram}_bpb"] = (
            df[f"bottom_conf_{ngram}_loss"] * bpb_coefficient
        )
        df[f"top_conf_{ngram}_bpb"] = df[f"top_conf_{ngram}_loss"] * bpb_coefficient

        # For some reason the legend items are listed in reverse order
        for i, step in enumerate(reversed(step_order)):
            group_df = df[df["step"] == step]
            color = px.colors.sequential.Plasma_r[i + 1]
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
                    showlegend=idx == 1,
                ),
                row=1,
                col=idx + 1,
            )

        fig.add_shape(
            type="line",
            x0=1,
            y0=entropies[idx],
            x1=2**11,
            y1=entropies[idx],
            line=dict(color="black", width=2, dash="dot"),
            row=1,
            col=idx + 1,
        )

    fig.update_layout(
        width=1000,
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
    fig.write_image(image_name, format="pdf")


if __name__ == "__main__":
    main(debug=False)
