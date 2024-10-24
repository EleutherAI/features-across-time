import tempfile
import time
from argparse import ArgumentParser
from pathlib import Path
import glob
import os

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


def hex_to_rgba(hex_color: str, opacity: float = 0.5):
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f"rgba({r}, {g}, {b}, {opacity})"


def get_confidence_intervals(
    mean: float, num_items: int, confidence=0.95
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    sem = np.sqrt(mean / num_items)
    return stats.norm.interval(confidence, loc=mean, scale=sem)


def get_model_size(model_name: str) -> int:
    """Convert a size string to a float value in millions."""
    if 'warmup' in model_name:
        size_str = model_name.split('-')[-2].lower()
    else:
        size_str = model_name.split('-')[-1].lower()
    numeric_part = ''.join(c for c in size_str if c.isdigit() or c == '.')

    if size_str.endswith('m'):
        return float(numeric_part) * 1e6
    elif size_str.endswith('b'):
        return float(numeric_part) * 1e9
    else:
        raise ValueError(f"Invalid model name format: {model_name}")


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


def get_legend_name(filename):
    return os.path.basename(filename).split('_')[-2]


def load_dfs(data_path: Path):
    csv_files = [
        x for x in glob.glob(str(data_path) + '/*.csv')
        if not 'seed' in x and not 'warmup' in x and not 'es' in x
    ]

    dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        legend_name = get_legend_name(file)
        model_size = int(get_model_size(legend_name))
        df['legend'] = legend_name
        df['model_size'] = model_size
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    return df.sort_values(['model_size', 'step'], ascending=True)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default="old-bigrams")
    parser.add_argument("--name", type=str, default="ngram-combined")
    parser.add_argument("--images_path", type=str, default="images")
    parser.add_argument("--num_samples", type=int, default=1024)
    parser.add_argument("--ngram_orders", "-n", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--data", "-d", type=str, default="pile")
    return parser.parse_args()


def main(
    data_path: Path,
    images_path: Path,
    bpb_coefficient: float,
    bpb_entropies: list[float],
    num_samples: int,
    ngram_orders: list[int],
    name: str,
):
    kaleido_workaround()
    images_path.mkdir(exist_ok=True, parents=True)

    df = load_dfs(data_path)

    fig = make_subplots(
        rows=2,
        cols=len(ngram_orders),
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=[
            f"{i}-gram loss" for i in ngram_orders
        ] + [
            f"D<sub>KL</sub>({i}-gram model || Pythia)"
            for i in ngram_orders
        ],
        horizontal_spacing=0.02,
        vertical_spacing=0.1,
    )
    fig.update_layout(
        title_text="Sequences across time",
        title_x=0.5,
        title_y=1.,
    )

    for col, order in enumerate([f"{n}_gram" for n in ngram_orders], start=1):
        df[f"{order}_loss_bottom_conf"] = df[f"mean_{order}_loss"].map(
            lambda x: get_confidence_intervals(x, num_samples)[0] # type: ignore
        )
        df[f"{order}_loss_top_conf"] = df[f"mean_{order}_loss"].map(
            lambda x: get_confidence_intervals(x, num_samples)[1] # type: ignore
        )

        df[f"{order}_bpb"] = df[f"mean_{order}_loss"] * bpb_coefficient
        df[f"{order}_bpb_bottom_conf"] = (
            df[f"{order}_loss_bottom_conf"] * bpb_coefficient
        )
        df[f"{order}_bpb_top_conf"] = df[f"{order}_loss_top_conf"] * bpb_coefficient

        for i, model in enumerate(df["legend"].unique()):
            df_model = df[df["legend"] == model]
            color = px.colors.sequential.Plasma_r[i % len(px.colors.sequential.Plasma_r)]
            transparent_color = hex_to_rgba(color, opacity=0.17)

            fig.add_trace(
                go.Scatter(
                    x=df_model["step"],
                    y=df_model[f"{order}_bpb_top_conf"],
                    fill=None,
                    mode="lines",
                    line=dict(width=0),
                    legendgroup=f"{i}",
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=df_model["step"],
                    y=df_model[f"{order}_bpb_bottom_conf"],
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=transparent_color,
                    legendgroup=f"{i}",
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=df_model["step"],
                    y=df_model[f"{order}_bpb"],
                    mode="lines+markers",
                    marker=dict(size=5, symbol=marker_series[i]),
                    name=model,
                    line=dict(color=color),
                    legendgroup=f"{i}",
                    showlegend=col==1,
                ),
                row=1,
                col=col,
            )

        if col < 3:
            fig.add_shape(
                type="line",
                x0=1,
                y0=bpb_entropies[col - 1],
                x1=143_000,
                y1=bpb_entropies[col- 1],
                line=dict(color="black", width=2, dash="dot"),
                row=1,
                col=col,
            )

    # df column name, y range, row, col
    div_metadata = [
        f"{i}_gram_kl_div"
        for i in ngram_orders
    ]
    row = 2

    for col, label in enumerate(div_metadata, start=1):
        df[f"top_conf_{label}"] = df[f"mean_{label}"].map(
            lambda x: get_confidence_intervals(x, num_samples)[0]
        )
        df[f"bottom_conf_{label}"] = df[f"mean_{label}"].map(
            lambda x: get_confidence_intervals(x, num_samples)[1]
        )

        df[f"top_conf_{label}_bpb"] = df[f"top_conf_{label}"] * bpb_coefficient
        df[f"bottom_conf_{label}_bpb"] = df[f"bottom_conf_{label}"] * bpb_coefficient
        df[f"mean_{label}_bpb"] = df[f"mean_{label}"] * bpb_coefficient

        for i, model in enumerate(df["legend"].unique()):
            df_model = df[df["legend"] == model]
            color = px.colors.sequential.Plasma_r[i % len(px.colors.sequential.Plasma_r)]
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
                    marker=dict(size=5, symbol=marker_series[i]),
                    name=model,
                    line=dict(color=color),
                    showlegend=False,
                ),
                row=row,
                col=col,
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
            font=dict(size=11),
            bgcolor="rgba(255, 255, 255, 0.85)",
        ),
        margin=dict(l=20, r=20, t=50, b=80),
    )

    tick_values, tick_texts = base_2_log_ticks(df["step"], spacing=2)

    fig.update_xaxes(
        title_text="", type="log", tickvals=tick_values, ticktext=tick_texts
    )

    fig.update_yaxes(
        title_text="Loss", title_font=dict(size=16), title_standoff=10, row=1, col=1
    )
    fig.update_yaxes(
        title_text="KL divergence",
        title_font=dict(size=16),
        title_standoff=10,
        row=2,
        col=1,
    )
    fig.add_annotation(
        dict(
            text="Training steps",
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.1,
            showarrow=False,
            font=dict(size=18),
        )
    )

    fig.show()
    fig.write_image(images_path / f"{name}.pdf", format="pdf")


if __name__ == "__main__":
    args = parse_args()
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
        args.name
    )