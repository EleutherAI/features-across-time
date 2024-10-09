from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objs import Figure
from plotly.subplots import make_subplots
from scripts.plot.plot_ngram import (
    base_2_log_ticks,
    get_confidence_intervals,
    hex_to_rgba,
    kaleido_workaround,
    marker_series,
    get_model_size,
    get_legend_name
)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default="output")
    parser.add_argument("--images_path", type=str, default="images")
    parser.add_argument("--num_samples", type=int, default=1024)
    return parser.parse_args()


def plot_loss(
    df: pd.DataFrame,
    num_samples: int,
    bpb_coefficient: float,
    entropies: list[float],
    qualitative=False,
    ngram_orders=[1, 2, 3, 4],
    title=True,
) -> Figure:
    kaleido_workaround()

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
    if title:
        fig.update_layout(
            title_text="N-gram sequence loss across time",
            title_x=0.5,
            title_y=1.,
            title_font=dict(size=20),
        )
    fig.update_layout(
        annotations=[
            dict(
                font=dict(size=18)
            ) for _ in fig['layout']['annotations']
        ]
    )

    for col, (order, entropy) in enumerate(zip(ngram_orders, entropies), start=1):
        df[f"bottom_conf_{order}_gram_loss"] = df[f"mean_{order}_gram_loss"].map(
            lambda x: get_confidence_intervals(x, num_samples)[0]
        )
        df[f"top_conf_{order}_gram_loss"] = df[f"mean_{order}_gram_loss"].map(
            lambda x: get_confidence_intervals(x, num_samples)[1]
        )

        df[f"mean_{order}_gram_bpb"] = df[f"mean_{order}_gram_loss"] * bpb_coefficient
        df[f"bottom_conf_{order}_gram_bpb"] = (
            df[f"bottom_conf_{order}_gram_loss"] * bpb_coefficient
        )
        df[f"top_conf_{order}_gram_bpb"] = df[f"top_conf_{order}_gram_loss"] * bpb_coefficient

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
                    y=df_model[f"top_conf_{order}_gram_bpb"],
                    fill=None,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=df_model["step"],
                    y=df_model[f"bottom_conf_{order}_gram_bpb"],
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=transparent_color,
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=df_model["step"],
                    y=df_model[f"mean_{order}_gram_bpb"],
                    mode="lines+markers",
                    marker=dict(size=5, symbol=marker_series[i]),
                    name=model,
                    line=dict(color=color),
                    showlegend=col == 1,
                ),
                row=1,
                col=col,
            )

        if col < 3:
            fig.add_shape(
                type="line",
                x0=1,
                y0=entropy,
                x1=2**17,
                y1=entropy,
                line=dict(color="black", width=2, dash="dot"),
                row=1,
                col=col,
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
            title="Pythia loss",
            bgcolor="rgba(255, 255, 255, 0.5)",
        ),
        yaxis=dict(range=[1.5, 5.2]),
        margin=dict(l=20, r=20, t=50, b=60),
    )

    tick_values, tick_texts = base_2_log_ticks(df["step"], spacing=2)
    fig.update_xaxes(
        title_text="", type="log", tickvals=tick_values, ticktext=tick_texts
    )
    fig.update_yaxes(
        title_text="Loss", title_font=dict(size=16), title_standoff=10, row=1, col=1
    )
    fig.add_annotation(
        dict(
            text="Training steps",
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.2,
            showarrow=False,
            font=dict(size=18),
        )
    )

    return fig

def plot_divergences(
    df: pd.DataFrame,
    num_samples: int,
    bpb_coefficient: float,
    ngram_orders: list[int],
    qualitative=False,
    
) -> Figure:
    kaleido_workaround()

    fig = make_subplots(
        rows=1,
        cols=len(ngram_orders),
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=[
            f"D<sub>KL</sub>({order}-gram model || Pythia) across time"
            for order in ngram_orders
        ],
        horizontal_spacing=0.02,
        vertical_spacing=0.05,
    )
    fig.update_layout(
        annotations=[
            dict(
                font=dict(size=18)
            ) for _ in fig['layout']['annotations']
        ]
    )

    for col, order in enumerate(ngram_orders, start=1):
        df[f"bottom_conf_{order}_gram_kl_div"] = df[f"mean_{order}_gram_kl_div"].map(
            lambda x: get_confidence_intervals(x, num_samples)[0]
        )
        df[f"top_conf_{order}_gram_kl_div"] = df[f"mean_{order}_gram_kl_div"].map(
            lambda x: get_confidence_intervals(x, num_samples)[1]
        )

        df[f"mean_{order}_gram_kl_div_bpb"] = df[f"mean_{order}_gram_kl_div"] * bpb_coefficient
        df[f"bottom_conf_{order}_gram_kl_div_bpb"] = df[f"bottom_conf_{order}_gram_kl_div"] * bpb_coefficient
        df[f"top_conf_{order}_gram_kl_div_bpb"] = df[f"top_conf_{order}_gram_kl_div"] * bpb_coefficient

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
                    y=df_model[f"top_conf_{order}_gram_kl_div_bpb"],
                    fill=None,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=df_model["step"],
                    y=df_model[f"bottom_conf_{order}_gram_kl_div_bpb"],
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=transparent_color,
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=df_model["step"],
                    y=df_model[f"mean_{order}_gram_kl_div_bpb"],
                    mode="lines+markers",
                    marker=dict(size=5),
                    name=model,
                    line=dict(color=color),
                    showlegend=False,
                ),
                row=1,
                col=col,
            )

    fig.update_layout(
        width=1200,
        height=400,
        margin=dict(l=20, r=20, t=50, b=60),
    )

    tick_values, tick_texts = base_2_log_ticks(df["step"], spacing=2)
    fig.update_xaxes(
        title_text="", type="log", tickvals=tick_values, ticktext=tick_texts
    )
    fig.update_yaxes(
        title_text="KL divergence",
        title_font=dict(size=16),
        title_standoff=10,
        row=1,
        col=1,
    )
    fig.add_annotation(
        dict(
            text="Training steps",
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.3,
            showarrow=False,
            font=dict(size=18),
        )
    )
    return fig


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
        ("pythia-6.9b", "6.9B"),
        ("pythia-12b", "12B"),
    ]
    model_dfs = []
    for model_name, pretty_model_name in model_metadata:
        model_df = pd.read_csv(data_path / f"ngram_{model_name}_{num_samples}.csv")
        model_df["model_name"] = model_name
        model_df["pretty_model_name"] = pretty_model_name
        model_df["size"] = get_model_size(model_name)
        model_dfs.append(model_df)
    df = pd.concat(model_dfs)
    df.sort_values(by=["step", "size"], inplace=True)

    fig = plot_loss(df, num_samples, bpb_coefficient, entropies, ngram_orders=[1, 2, 3, 4])
    fig.write_image(images_path / "ngram-loss.pdf", format="pdf")
    fig = plot_loss(df, num_samples, bpb_coefficient, entropies, ngram_orders=[1, 2, 3, 4], title=False)
    fig.write_image(images_path / "ngram-loss-no-title.pdf", format="pdf")
    fig = plot_divergences(df, num_samples, bpb_coefficient, ngram_orders=[1, 2])
    fig.write_image(images_path / "ngram-divergence.pdf", format="pdf")

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
            model_df["size"] = get_model_size(model_name)
            model_dfs.append(model_df)
        df = pd.concat(model_dfs)
        df.sort_values(by=["step", "size"], inplace=True)

        fig = plot_loss(df, num_samples, bpb_coefficient, entropies, ngram_orders=[1, 2, 3, 4], qualitative=True)
        fig.update_layout(legend=dict(y=0.8))
        fig.write_image(images_path / f"warmups-{model_size}m-loss.pdf", format="pdf")
        
        fig = plot_divergences(df, num_samples, bpb_coefficient, ngram_orders=[1, 2], qualitative=True)
        fig.update_layout(height=300)
        fig.write_image(images_path / f"warmups-{model_size}m-divergence.pdf", format="pdf")


def main():
    args = parse_args()
    bpb_coefficient = 0.3650388
    entropies = [2.89, 2.04, -1, -1] # expensive to compute above 2-gram

    data_path, images_path = Path(args.data_path), Path(args.images_path)
    images_path.mkdir(exist_ok=True, parents=True)

    plot_suite(data_path, images_path, args.num_samples, bpb_coefficient, entropies)
    plot_warmups(data_path, images_path, args.num_samples, bpb_coefficient, entropies)


if __name__ == "__main__":
    main()
