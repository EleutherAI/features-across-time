import tempfile
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats


def kaleido_workaround():
    # Write data to work around Kaleido bug: https://github.com/plotly/plotly.py/issues/3469
    with tempfile.NamedTemporaryFile() as temp_file:
        fig = px.scatter(x=[0], y=[0])
        fig.write_image(temp_file.name, format="pdf")
    time.sleep(2)


def base_2_log_ticks(values, spacing=1):
    max_val = np.ceil(np.log2(values.max()))
    ticks = 2 ** np.arange(0, max_val + 1, spacing)
    labels = [f"2<sup>{int(i)}</sup>" for i in np.arange(0, max_val + 1, spacing)]
    return ticks, labels


def hex_to_rgba(hex_color: str, opacity=0.5):
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f"rgba({r}, {g}, {b}, {opacity})"


def get_confidence_intervals(
    mean: float, num_items: int, confidence=0.95
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    sem = np.sqrt(mean / num_items)
    return stats.norm.interval(confidence, loc=mean, scale=sem)


# Ideally all trend lines will use a different marker for accessibility
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


def main(
    data_path: Path,
    images_path: Path,
    bpb_coefficient: float,
    ngram_bpb_entropies: list[float],
    num_samples: int,
    ngram_orders: list[int],
    name: str,
    model_metadata: list[tuple[str, str]],
    model_series = "Pythia",
):
    images_path.mkdir(exist_ok=True, parents=True)
    dfs = []
    for model_name, pretty_model_name in model_metadata:
        model_df = pd.read_csv(data_path / f"ngram_{model_name}_{num_samples}.csv")
        model_df["model_name"] = model_name
        model_df["pretty_model_name"] = pretty_model_name
        dfs.append(model_df)
    df = pd.concat(dfs)

    kaleido_workaround()
    tick_values, tick_texts = base_2_log_ticks(df["step"], spacing=2)
    fig = make_subplots(
        rows=2,
        cols=len(ngram_orders),
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=[
            f"{i}-gram sequence loss across time" for i in ngram_orders
        ] + [
            f"D<sub>KL</sub>({i}-gram model || {model_series}) across time"
            for i in ngram_orders
        ],
        horizontal_spacing=0.02,
        vertical_spacing=0.1,
    )

    for idx, ngram in enumerate([f"{i}_gram" for i in ngram_orders]):
        df[f"{ngram}_loss_bottom_conf"] = df[f"mean_{ngram}_loss"].map(
            lambda x: get_confidence_intervals(x, num_samples)[0]
        )
        df[f"{ngram}_loss_top_conf"] = df[f"mean_{ngram}_loss"].map(
            lambda x: get_confidence_intervals(x, num_samples)[1]
        )

        df[f"{ngram}_bpb"] = df[f"mean_{ngram}_loss"] * bpb_coefficient
        df[f"{ngram}_bpb_bottom_conf"] = (
            df[f"{ngram}_loss_bottom_conf"] * bpb_coefficient
        )
        df[f"{ngram}_bpb_top_conf"] = df[f"{ngram}_loss_top_conf"] * bpb_coefficient

        for i, model in enumerate(df["pretty_model_name"].unique()):
            df_model = df[df["pretty_model_name"] == model]
            color = px.colors.sequential.Plasma_r[i]
            transparent_color = hex_to_rgba(color, opacity=0.17)

            fig.add_trace(
                go.Scatter(
                    x=df_model["step"],
                    y=df_model[f"{ngram}_bpb_top_conf"],
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
                    y=df_model[f"{ngram}_bpb_bottom_conf"],
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
                    y=df_model[f"{ngram}_bpb"],
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
            fig.add_shape(
                type="line",
                x0=1,
                y0=ngram_bpb_entropies[idx],
                x1=143000,
                y1=ngram_bpb_entropies[idx],
                line=dict(color="black", width=2, dash="dot"),
                row=1,
                col=idx + 1,
            )

    # df column name, y range, row, col
    div_metadata = [
        f"{i}_gram_kl_div"
        for i in ngram_orders
    ]
    row = 2

    for col_idx, label in enumerate(div_metadata):
        df[f"top_conf_{label}"] = df[f"mean_{label}"].map(
            lambda x: get_confidence_intervals(x, num_samples)[0]
        )
        df[f"bottom_conf_{label}"] = df[f"mean_{label}"].map(
            lambda x: get_confidence_intervals(x, num_samples)[1]
        )

        df[f"top_conf_{label}_bpb"] = df[f"top_conf_{label}"] * bpb_coefficient
        df[f"bottom_conf_{label}_bpb"] = df[f"bottom_conf_{label}"] * bpb_coefficient
        df[f"mean_{label}_bpb"] = df[f"mean_{label}"] * bpb_coefficient

        for i, model in enumerate(df["pretty_model_name"].unique()):
            df_model = df[df["pretty_model_name"] == model]
            color = px.colors.sequential.Plasma_r[i]
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
                col=col_idx + 1,
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
                col=col_idx + 1,
            )
            fig.add_trace(
                go.Scatter(
                    x=df_model["step"],
                    y=df_model[f"mean_{label}_bpb"],
                    mode="lines+markers",
                    marker=dict(size=5, symbol=marker_series[i]),
                    name=model,
                    line=dict(color=color),
                    showlegend=False,
                ),
                row=row,
                col=col_idx + 1,
            )
    fig.update_layout(
        width=1800,
        height=800,
        legend=dict(
            title="Pythia",
            x=0.98,
            y=0.98,
            xanchor="right",
            yanchor="top",
            font=dict(size=8),
            bgcolor="rgba(255, 255, 255, 0.85)",
        ),
        margin=dict(l=20, r=20, t=50, b=80),
    )

    fig.update_xaxes(
        title_text="", type="log", tickvals=tick_values, ticktext=tick_texts
    )

    fig.update_yaxes(
        title_text="Loss", title_font=dict(size=12), title_standoff=10, row=1, col=1
    )
    # fig.update_yaxes(range=[1, 5], row=1, col=1)
    # fig.update_yaxes(range=[0, 4.5], row=2, col=1)
    fig.update_yaxes(
        title_text="KL divergence",
        title_font=dict(size=12),
        title_standoff=10,
        row=2,
        col=1,
    )
    # Add a shared, centered x-axis label
    fig.add_annotation(
        dict(
            text="Training step",
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.1,
            showarrow=False,
            font=dict(size=14),
        )
    )

    fig.write_image(images_path / f"{name}.pdf", format="pdf")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default="output")
    parser.add_argument("--name", type=str, default="ngram-combined")
    parser.add_argument("--images_path", type=str, default="images")
    parser.add_argument("--num_samples", type=int, default=1024)
    parser.add_argument("--ngram_orders", "-n", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--data", "-d", type=str, default="pile")
    args = parser.parse_args()

    # model_metadata = [
    #     (f'es_pythia-160m', "160M")
    # ]
    model_metadata = [
        ("pythia-14m", "14M"),
        ("pythia-70m", "70M"),
        ("pythia-160m", "160M"),
        ("pythia-410m", "410M"),
        ("pythia-1b", "1B"),
        # ("pythia-1.4b", "1.4B"),
        # ("pythia-2.8b", "2.8B"),
        # ("pythia-6.9b", "6.9B"),
        # ("pythia-12b", "12B"),
    ]
    # model_metadata = [
    #     (f'es_{model_name}', pretty_name)
    #     for model_name, pretty_name in model_metadata
    # ]

    data_args = {
        "pile": dict(
            bpb_coefficient=0.3650388,
            bpb_entropies=[2.89, 2.04],
        ),
        "es": dict(
            bpb_coefficient=0.4157027,
            bpb_entropies=[2.72, 1.50],
        ),
    }

    main(
        Path(args.data_path),
        Path(args.images_path),
        data_args[args.data]["bpb_coefficient"],
        data_args[args.data]["bpb_entropies"],
        args.num_samples,
        args.ngram_orders,
        args.name,
        model_metadata
    )
