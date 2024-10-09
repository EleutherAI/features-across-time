from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scripts.plot.plot_ngram import (
    base_2_log_ticks, 
    hex_to_rgba, 
    kaleido_workaround, 
    marker_series,
    get_model_size
)
from plotly.subplots import make_subplots


def plot_seed(
        df: pd.DataFrame, 
        bpb_coefficient: float, 
        entropies: list[float],
        ngram_orders=[1, 2, 3, 4]
    ):
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
                    x=df_model["step"].unique(),
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

        if entropy is not None:
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
            width=1200,
            height=300,
            legend=dict(
                x=0.98,
                y=0.94,
                xanchor="right",
                yanchor="top",
                font=dict(size=11),
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
                    text="Training steps",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=-0.2,
                    showarrow=False,
                    font=dict(size=18),
                )
            )
            fig.update_layout(margin=dict(l=20, r=20, t=30, b=30))

        fig.update_xaxes(
            title_text="", type="log", tickvals=tick_values, ticktext=tick_texts
        )

        fig.update_yaxes(
            title_text=ytitle, title_font=dict(size=16), title_standoff=10, col=1
        )
        fig.update_yaxes(range=[0.3, 0.8], row=3)
        return fig

    for n, entropy in zip(ngram_orders, entropies):
        df[f"mean_{n}_gram_bpb"] = df[f"mean_{n}_gram_loss"] * bpb_coefficient
        fig = create_row(
            df,
            f"{n}-gram sequence loss across time",
            f"mean_{n}_gram_bpb",
            ytitle="Loss",
            show_legend=n==ngram_orders[0],
            show_xaxis=n==ngram_orders[-1],
            entropy=entropy if entropy != -1 else None,
        )

        image_name = Path.cwd() / "images" / f"seed_{n}_gram.pdf"
        fig.write_image(image_name, format="pdf")


    for n in [1, 2]:
        df[f"mean_{n}_gram_kl_div_bpb"] = df[f"mean_{n}_gram_kl_div"] * bpb_coefficient
        fig = create_row(
            df,
            f"D<sub>KL</sub>({n}-gram model || Pythia) across time",
            f"mean_{n}_gram_kl_div_bpb",
            ytitle="KL divergence",
            show_xaxis=n==2,
        )

        image_name = Path.cwd() / "images" / f"seed_{n}_gram_kl_div.pdf"
        fig.write_image(image_name, format="pdf")


def main(
        data_path: Path, 
        images_path: Path, 
        num_samples: int,
        bpb_coefficient: float,
        entropies: list[float]
    ):
    images_path.mkdir(exist_ok=True, parents=True)

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
            seed_df["model_size"] = get_model_size(model_name)
            seed_dfs.append(seed_df)

    df = pd.concat(seed_dfs)
    df.sort_values(by=["step", "model_size"], inplace=True)

    plot_seed(df, bpb_coefficient, entropies)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default="final_4096")
    parser.add_argument("--images_path", type=str, default="images")
    parser.add_argument("--num_samples", type=int, default=4096)
    args = parser.parse_args()

    main(
        Path(args.data_path), 
        Path(args.images_path),
        num_samples=args.num_samples,
        bpb_coefficient=0.3650388,
        entropies=[2.89, 2.04, -1, -1]
    )
