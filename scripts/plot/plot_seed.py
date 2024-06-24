import os
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plot_ngram import base_2_log_ticks, hex_to_rgba, kaleido_workaround, marker_series
from plotly.subplots import make_subplots


def plot_seed_loss(df: pd.DataFrame, bpb_coefficient: float, entropies: list[float]):
    kaleido_workaround()

    tick_values, tick_texts = base_2_log_ticks(df["step"], spacing=2)

    def create_row(
        df,
        title: str,
        name: str,
        ytitle,
        show_xaxis=False,
        show_legend=False,
        entropy=None,
    ):
        fig = make_subplots(
            rows=1,
            cols=4,
            shared_xaxes=True,
            shared_yaxes=True,
            subplot_titles="",
            horizontal_spacing=0.02,
            vertical_spacing=0.05,
        )
        for model_index, model in enumerate(df["pretty_model_name"].unique()):
            df_model = df[df["pretty_model_name"] == model]
            color = px.colors.qualitative.Plotly[model_index]
            transparent_color = hex_to_rgba(color, opacity=0.2)

            for seed in df_model["seed"].unique():
                seed_df = df_model[df_model["seed"] == seed]
                fig.add_trace(
                    go.Scatter(
                        x=seed_df["step"],
                        y=seed_df[name],
                        mode="lines",
                        name=model,
                        line=dict(color=transparent_color),
                        showlegend=False,
                    ),
                    row=1,
                    col=model_index + 1,
                )

            seed_mean = df_model.groupby("step")[name].mean()
            fig.add_trace(
                go.Scatter(
                    x=df_model["step"],
                    y=seed_mean,
                    mode="lines+markers",
                    name=model,
                    line=dict(color=color),
                    showlegend=show_legend,
                    marker=dict(size=5, symbol=marker_series[model_index]),
                ),
                row=1,
                col=model_index + 1,
            )

        if entropy:
            for col in [1, 2, 3, 4]:
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
            width=1000,
            height=300,
            legend=dict(
                x=0.98,
                y=0.94,
                xanchor="right",
                yanchor="top",
                font=dict(size=8),
                bgcolor="rgba(255, 255, 255, 0.85)",
            ),
            legend_title="Pythia model",
            autosize=True,
            margin=dict(l=20, r=20, t=35, b=0),
        )

        fig.add_annotation(
            dict(
                text=title,
                xref="paper",
                yref="paper",
                x=0.5,
                y=1.15,
                showarrow=False,
                font=dict(size=16),
            )
        )

        if show_xaxis:
            fig.add_annotation(
                dict(
                    text="Training step",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=-0.2,
                    showarrow=False,
                    font=dict(size=16),
                )
            )
            fig.update_layout(margin=dict(l=20, r=20, t=30, b=30))

        fig.update_xaxes(
            title_text="", type="log", tickvals=tick_values, ticktext=tick_texts
        )

        fig.update_yaxes(
            title_text=ytitle, title_font=dict(size=12), title_standoff=10, col=1
        )
        fig.update_yaxes(range=[0.3, 0.8], row=3)
        return fig

    for idx, ngram in enumerate(["1_gram", "2_gram"]):
        title = "Unigram" if ngram == "1_gram" else "Bigram"

        df[f"mean_{ngram}_bpb"] = df[f"mean_{ngram}_loss"] * bpb_coefficient
        fig = create_row(
            df,
            f"{title} sequence loss across time",
            f"mean_{ngram}_bpb",
            ytitle="Loss",
            show_legend=ngram == "1_gram",
            entropy=entropies[idx],
        )

        image_name = Path.cwd() / "images" / f"seed_{ngram}.pdf"
        fig.write_image(image_name, format="pdf")

    div_metadata = [
        (
            "1_gram_kl_div",
            "D<sub>KL</sub>(unigram model || Pythia) across time",
        ),
        (
            "2_gram_kl_div",
            "D<sub>KL</sub>(bigram model || Pythia) across time",
        ),
    ]
    for label, pretty_label in div_metadata:
        df[f"mean_{label}_bpb"] = df[f"mean_{label}"] * bpb_coefficient
        fig = create_row(
            df,
            pretty_label,
            f"mean_{label}_bpb",
            ytitle="KL divergence",
            show_xaxis=label == "2_gram_kl_div",
        )

        image_name = Path.cwd() / "images" / f"seed_{label}.pdf"
        fig.write_image(image_name, format="pdf")


def main(data_path: Path, images_path: Path, num_samples=1024):
    os.makedirs(images_path, exist_ok=True)

    model_metadata = [
        ("pythia-14m", "14M", 9),
        ("pythia-70m", "70M", 9),
        ("pythia-160m", "160M", 9),
        ("pythia-410m", "410M", 4),
    ]

    seed_dfs = []
    for model_name, pretty_model_name, num_seeds in model_metadata:
        for i in range(1, num_seeds + 1):
            seed_df = pd.read_csv(
                data_path / f"ngram_{model_name}-seed{i}_{num_samples}.csv"
            )
            seed_df["seed"] = i
            seed_df["model_name"] = model_name
            seed_df["pretty_model_name"] = pretty_model_name
            seed_dfs.append(seed_df)

    plot_seed_loss(
        pd.concat(seed_dfs), bpb_coefficient=0.3650388, entropies=[2.89, 2.04]
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default="output")
    parser.add_argument("--images_path", type=str, default="images")
    args = parser.parse_args()

    main(Path(args.data_path), Path(args.images_path))