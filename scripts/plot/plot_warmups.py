from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scripts.plot.plot_ngram import (
    base_2_log_ticks,
    get_confidence_intervals,
    hex_to_rgba,
    kaleido_workaround,
    marker_series,
    get_model_size,
)
from plotly.subplots import make_subplots


def plot_loss_and_divergence(
    df: pd.DataFrame,
    images_path: Path,
    model_size: int,
    bpb_coefficient: float,
    entropies: list[float],
    num_samples: int,
):
    kaleido_workaround()
    fig = make_subplots(
        rows=2,
        cols=2,
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=[
            "Unigram sequence loss across time",
            "Bigram sequence loss across time",
            "D<sub>KL</sub>(unigram model || Pythia) across time",
            "D<sub>KL</sub>(bigram model || Pythia) across time",
        ],
        horizontal_spacing=0.02,
        vertical_spacing=0.1,
    )
    div_metadata = [("1_gram_kl_div", 2, 1), ("2_gram_kl_div", 2, 2)]

    tick_values, tick_texts = base_2_log_ticks(df["step"], spacing=2)
    for idx, ngram in enumerate(["1_gram", "2_gram"]):
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
            color = px.colors.qualitative.Plotly[i]
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

        fig.add_shape(
            type="line",
            x0=1,
            y0=entropies[idx],
            x1=143_000,
            y1=entropies[idx],
            line=dict(color="black", width=2, dash="dot"),
            row=1,
            col=idx + 1,
        )

    for label, row, col in div_metadata:
        for i, model in enumerate(df["pretty_model_name"].unique()):
            color = px.colors.qualitative.Plotly[i]
            transparent_color = hex_to_rgba(color, opacity=0.17)

            df_model = df[df["pretty_model_name"] == model]
            df_model[f"bottom_conf_{label}"] = df_model[f"mean_{label}"].map(
                lambda x: get_confidence_intervals(x, num_samples)[0]
            )
            df_model[f"top_conf_{label}"] = df_model[f"mean_{label}"].map(
                lambda x: get_confidence_intervals(x, num_samples)[1]
            )
            df_model[f"top_conf_{label}_bpb"] = (
                df_model[f"top_conf_{label}"] * bpb_coefficient
            )
            df_model[f"bottom_conf_{label}_bpb"] = (
                df_model[f"bottom_conf_{label}"] * bpb_coefficient
            )
            df_model[f"mean_{label}_bpb"] = df_model[f"mean_{label}"] * bpb_coefficient

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
        width=1200,
        height=600,
        legend=dict(
            x=0.98,
            y=0.98,
            xanchor="right",
            yanchor="top",
            font=dict(size=11),
            bgcolor="rgba(255, 255, 255, 0.85)",
        ),
        margin=dict(l=20, r=20, t=50, b=60),
    )

    fig.update_xaxes(
        title_text="", type="log", tickvals=tick_values, ticktext=tick_texts
    )

    fig.update_yaxes(
        title_text="Loss", title_font=dict(size=16), title_standoff=10, row=1, col=1
    )
    fig.update_yaxes(range=[1.8, 4.5], row=1, col=1)
    fig.update_yaxes(range=[0, 4.5], row=2, col=1)
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

    fig.write_image(images_path / f"warmups-{model_size}m.pdf", format="pdf")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default="output")
    parser.add_argument("--images_path", type=str, default="images")
    parser.add_argument("--num_samples", "-s", type=int, default=4096)
    return parser.parse_args()


def main(
        data_path: Path, 
        images_path: Path, 
        num_samples: int,
        bpb_coefficient: float,
        entropies: list[float]
    ):
    images_path.mkdir(exist_ok=True, parents=True)
    
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
            model_df["model_size"] = get_model_size(model_name)
            model_dfs.append(model_df)

        df = pd.concat(model_dfs)
        df = df.sort_values(['model_size', 'step'], ascending=True)
        plot_loss_and_divergence(
            df, 
            images_path, 
            model_size,
            bpb_coefficient,
            entropies,
            num_samples,
        )


if __name__ == "__main__":
    args = parse_args()

    main(
        Path(args.data_path), 
        Path(args.images_path), 
        args.num_samples,
        bpb_coefficient=0.3650388,
        entropies=[2.89, 2.04]
    )
