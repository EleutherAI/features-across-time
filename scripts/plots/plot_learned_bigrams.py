import os
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from scripts.plots.plot_ngram import base_2_log_ticks, kaleido_workaround


def plot(df: pd.DataFrame, image_name: str, model_series: str, num_samples: int):
    kaleido_workaround()

    # es 1 billion words
    bpb_coefficient = 0.4157027

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

    fig = go.Figure()
    for i, model in enumerate(df["pretty_model_name"].unique()):
        df_model = df[df["pretty_model_name"] == model]
        df_model["kl_mean_bpb"] = df_model["kl_mean"].map(
            lambda x: float(re.findall(r"tensor\(([^)]+)\)", x)[0]) * bpb_coefficient
        )
        df_model["kl_std_bpb"] = df_model["kl_std"].map(
            lambda x: float(re.findall(r"tensor\(([^)]+)\)", x)[0]) * bpb_coefficient
        )
        color = px.colors.sequential.Plasma_r[i + 1]
        y_err = 1.96 * df_model["kl_std_bpb"] / (num_samples**0.5)

        fig.add_trace(
            go.Scatter(
                x=df_model["step"],
                y=df_model["kl_mean_bpb"],
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
    tick_values, tick_texts = base_2_log_ticks(df["step"], step=2)
    fig.update_xaxes(type="log", tickvals=tick_values, ticktext=tick_texts)
    fig.update_yaxes(range=[0, 6])

    fig.write_image(str(image_name), format="pdf")


def main():
    os.makedirs(Path.cwd() / "images", exist_ok=True)

    model_series = "Pythia"
    num_samples = 8192  # 1024
    image_name = Path.cwd() / "images" / "learned-bigram-kl-div.pdf"
    model_metadata = [
        # ("pythia-14m", "14M", 8),
        # ("pythia-70m", "70M", 8),
        ("pythia-160m", "160M", 8),
        # ("pythia-410m", "410M", 8),
        # ("pythia-1b", "1B", 8),
        # ("pythia-1.4b", "1.4B", 8),
        # ("pythia-2.8b", "2.8B", 8),
        # ("pythia-6.9b", "6.9B", 8),
        # ("pythia-12b", "12B", 8),
    ]

    model_dfs = []
    for model_name, pretty_model_name, num_chunks in model_metadata:
        model_df = pd.read_csv(
            Path.cwd() / "output" / f"finetune_bigram_{model_name}_{num_samples}-1.csv"
        )
        model_df["pretty_model_name"] = pretty_model_name
        model_dfs.append(model_df)

    df = pd.concat(model_dfs)
    plot(df, image_name, model_series, num_samples)


if __name__ == "__main__":
    main()
