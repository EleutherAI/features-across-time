from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scripts.plot.plot_ngram import (
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
    bpb_coefficient: float,
    ngram_orders = [2, 3, 4],
):
    kaleido_workaround()
    images_path.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(data_path / f"context_{model_name}_{num_samples}.csv")

    # Remove several steps because their lines' confidence intervals overlap
    steps_to_remove = [0, 1, 2, 4, 8, 32, 64, 128, 512, 2000, 2048, 5_000, 16_000, 20_000, 33_000, 80_000, 131_000]
    df = df[~df["step"].isin(steps_to_remove)]

    # Filter to log spaced samples
    max_index = df["token"].max()
    log_spaced_indices = np.unique(
        np.logspace(
            0, np.log2(max_index), num=int(np.log2(max_index)) + 1, base=2
        ).astype(int)
    )
    df = df[df["token"].isin(log_spaced_indices)]

    step_order = sorted(df["step"].unique())

    fig = make_subplots(
        rows=1,
        cols=len(ngram_orders),
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=[
            f"{n}-gram"
            for n in ngram_orders
        ],
        horizontal_spacing=0.02,
        vertical_spacing=0.05,
    )

    for idx, ngram in enumerate([str(order) for order in ngram_orders], start = 1):
        df[f"bottom_conf_{ngram}_loss"] = df[f"mean_{ngram}_per_token_loss"].map(
            lambda x: get_confidence_intervals(x, num_samples)[0] # type: ignore
        )
        df[f"top_conf_{ngram}_loss"] = df[f"mean_{ngram}_per_token_loss"].map(
            lambda x: get_confidence_intervals(x, num_samples)[1] # type: ignore
        ) 

        df[f"mean_{ngram}_per_token_loss_bpb"] = df[f"mean_{ngram}_per_token_loss"] * bpb_coefficient
        df[f"bottom_conf_{ngram}_bpb"] = (
            df[f"bottom_conf_{ngram}_loss"] * bpb_coefficient
        )
        df[f"top_conf_{ngram}_bpb"] = df[f"top_conf_{ngram}_loss"] * bpb_coefficient

        for i, step in enumerate(reversed(step_order)):
            group_df = df[df["step"] == step]
            color = px.colors.sequential.Plasma_r[(i + 1) % len(px.colors.sequential.Plasma_r)]
            transparent_color = hex_to_rgba(color, opacity=0.17)

            fig.add_trace(
                go.Scatter(
                    x=group_df["token"],
                    y=group_df[f"top_conf_{ngram}_bpb"],
                    fill=None,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=idx,
            )
            fig.add_trace(
                go.Scatter(
                    x=group_df["token"],
                    y=group_df[f"bottom_conf_{ngram}_bpb"],
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=transparent_color,
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=idx,
            )
            fig.add_trace(
                go.Scatter(
                    x=group_df["token"],
                    y=group_df[f"mean_{ngram}_per_token_loss_bpb"],
                    mode="lines+markers",
                    marker=dict(size=5, symbol=marker_series[i]),
                    name=f"{step:,}",
                    line=dict(color=color),
                    showlegend=idx == 1,
                ),
                row=1,
                col=idx,
            )

    fig.update_layout(
        width=1200,
        height=400,
        legend=dict(
            x=0.98,
            y=0.65,
            xanchor="right",
            yanchor="middle",
            font=dict(size=11),
            title="Step",
            bgcolor="rgba(255, 255, 255, 0.5)",
        ),
        margin=dict(l=20, r=20, t=50, b=60),
    )

    tick_values, tick_texts = base_2_log_ticks(df["token"], spacing=2)
    fig.update_xaxes(
        title_text="", type="log", tickvals=tick_values, ticktext=tick_texts
    )
    fig.update_yaxes(
        title_text="Loss", title_font=dict(size=16), title_standoff=10, row=1, col=1
    )
    fig.add_annotation(
        dict(
            text="Token index",
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.2,
            showarrow=False,
            font=dict(size=18),
        )
    )
    fig.write_image(
        images_path / f"in-context-{model_name}.pdf", format="pdf"
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default="output")
    parser.add_argument("--images_path", type=str, default="images")
    parser.add_argument("--num_samples", type=int, default=4096)
    parser.add_argument("--n", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--model_name", type=str, default="pythia-12b")
    args = parser.parse_args()

    main(
        args.model_name, 
        Path(args.images_path), 
        Path(args.data_path), 
        args.num_samples,
        bpb_coefficient=0.3650388,
        ngram_orders=args.n
    )
