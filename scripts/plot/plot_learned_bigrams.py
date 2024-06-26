from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from scripts.plot.plot_ngram import base_2_log_ticks, kaleido_workaround, marker_series


def main(data_path: Path, images_path: Path, bpb_coefficient: float, num_samples: int):
    images_path.mkdirs(exist_ok=True, parents=True)
    kaleido_workaround()

    model_metadata = [
        # ("pythia-14m", "14M"),
        # ("pythia-70m", "70M"),
        ("pythia-160m", "160M"),
        # ("pythia-410m", "410M"),
        # ("pythia-1b", "1B"),
        # ("pythia-1.4b", "1.4B"),
        # ("pythia-2.8b", "2.8B"),
        # ("pythia-6.9b", "6.9B"),
        # ("pythia-12b", "12B"),
    ]
    model_dfs = []
    for model_name, pretty_model_name in model_metadata:
        model_df = pd.read_csv(
            data_path / f"finetune_bigram_{model_name}_{num_samples}.csv"
        )
        model_df["pretty_model_name"] = pretty_model_name
        model_dfs.append(model_df)
    df = pd.concat(model_dfs)
    df["kl_mean_bpb"] = df["kl_mean"] * bpb_coefficient
    df["kl_std_bpb"] = df["kl_std"] * bpb_coefficient

    fig = go.Figure()
    for i, model in enumerate(df["pretty_model_name"].unique()):
        model_df = df[df["pretty_model_name"] == model]

        color = px.colors.sequential.Plasma_r[i + 1]
        y_err = 1.96 * model_df["kl_std_bpb"] / (num_samples ** 0.5)

        fig.add_trace(
            go.Scatter(
                x=model_df["step"],
                y=model_df["kl_mean_bpb"],
                mode="lines+markers",
                marker=dict(size=5, symbol=marker_series[i + 1]),
                name=model,
                line=dict(color=color),
                error_y=dict(type="data", array=y_err, visible=True),
            )
        )

    fig.update_layout(
        width=1000,
        height=600,
        legend=dict(
            x=0.98,
            y=0.98,
            xanchor="right",
            yanchor="top",
            font=dict(size=8),
            bgcolor="rgba(255, 255, 255, 0.85)",
        ),
        margin=dict(l=20, r=20, t=50, b=60),
        xaxis_title="Training step",
        yaxis_title="KL Divergence (BPB)",
        title="KL Divergence(Learned | Dataset Bigrams) over Training Steps",
    )
    tick_values, tick_texts = base_2_log_ticks(df["step"], spacing=2)
    fig.update_xaxes(type="log", tickvals=tick_values, ticktext=tick_texts)
    fig.update_yaxes(range=[0, 6])

    fig.write_image(images_path / "learned-bigram-kl-div.pdf", format="pdf")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default="output")
    parser.add_argument("--images_path", type=str, default="images")
    parser.add_argument("--num_samples", type=int, default=8192)
    args = parser.parse_args()

    # es 1 billion tokens
    bpb_coefficient = 0.4157027
    # bpb_entropies = [2.72, 1.50]

    # pile
    # bpb_coefficient = 0.3650388
    # bpb_entropies = [2.89, 2.04]

    main(
        Path(args.data_path), 
        Path(args.images_path), 
        bpb_coefficient, 
        args.num_samples
    )
