import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats


def kaleido_workaround():
    # Garbage data to work around Kaleido bug: https://github.com/plotly/plotly.py/issues/3469
    with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_file:
        fig = px.scatter(x=[0], y=[0])
        fig.write_image(temp_file.name, format="pdf")
    time.sleep(2)


def base_2_log_ticks(values, step=1):
    max_val = np.ceil(np.log2(values.max()))
    ticks = 2 ** np.arange(0, max_val + 1, step)
    labels = [f"2<sup>{int(i)}</sup>" for i in np.arange(0, max_val + 1, step)]
    return ticks, labels


def hex_to_rgba(hex_color, opacity=0.5):
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    return f"rgba({r}, {g}, {b}, {opacity})"


def get_confidence_intervals(
    mean: float, num_items: int, confidence=0.95
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    sem = np.sqrt(mean / num_items)
    return stats.norm.interval(confidence, loc=mean, scale=sem)


def plot_bpb_and_divergences(
    df: pd.DataFrame,
    image_name: str,
    bpb_coefficient: int,
    model_series: str,
    ngram_entropies_bpb: list[int],
    qualitative=False,
):
    kaleido_workaround()
    tick_values, tick_texts = base_2_log_ticks(df["step"], step=2)

    div_metadata = [
        (
            "1-gram_logit_kl_div",
            f"D<sub>KL</sub>(unigram model || {model_series}) across time",
            [0, 7],
            2,
            2,
        ),
        (
            "2-gram_logit_kl_div",
            f"D<sub>KL</sub>(bigram model || {model_series}) across time",
            [0, 7],
            2,
            1,
        ),
    ]
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

    fig = make_subplots(
        rows=2,
        cols=2,
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=[
            "Unigram sequence loss across time",
            "Bigram sequence loss across time",
        ]
        + [label[1] for label in div_metadata],
        horizontal_spacing=0.02,
        vertical_spacing=0.1,
    )

    for idx, ngram in enumerate(["1_gram", "2_gram"]):  # "3_gram"
        df[f"mean_{ngram}_loss_bottom_conf"] = df[f'mean_{ngram}_loss'].map(lambda x : get_confidence_intervals(x, 1024)[0])
        df[f"mean_{ngram}_loss_top_conf"] = df[f'mean_{ngram}_loss'].map(lambda x : get_confidence_intervals(x, 1024)[1])

        df[f"mean_{ngram}_bpb"] = df[f"mean_{ngram}_loss"] * bpb_coefficient
        df[f"mean_{ngram}_bpb_bottom_conf"] = df[f"mean_{ngram}_loss_bottom_conf"] * bpb_coefficient
        df[f"mean_{ngram}_bpb_top_conf"] = df[f"mean_{ngram}_loss_top_conf"] * bpb_coefficient

        for i, model in enumerate(df["pretty_model_name"].unique()):
            df_model = df[df["pretty_model_name"] == model]
            color = (
                px.colors.qualitative.Plotly[i]
                if qualitative
                else px.colors.sequential.Plasma_r[i]
            )
            transparent_color = hex_to_rgba(color, opacity=0.17)

            fig.add_trace(
                go.Scatter(
                    x=df_model["step"],
                    y=df_model[f"mean_{ngram}_bpb_top_conf"],
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
                    y=df_model[f"mean_{ngram}_bpb_bottom_conf"],
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

        if idx != 2:
            fig.add_shape(
                type="line",
                x0=1,
                y0=ngram_entropies_bpb[idx],
                x1=143000,
                y1=ngram_entropies_bpb[idx],
                line=dict(color="black", width=2, dash="dot"),
                row=1,
                col=idx + 1,
            )

    for label, pretty_label, y_range, row, col in div_metadata:
        df[f"top_conf_{label}"] = df[f'mean_{label}'].map(lambda x : get_confidence_intervals(x, 1024)[0])
        df[f"bottom_conf_{label}"] = df[f'mean_{label}'].map(lambda x : get_confidence_intervals(x, 1024)[1])

        df[f"top_conf_{label}_bpb"] = df[f"top_conf_{label}"] * bpb_coefficient
        df[f"bottom_conf_{label}_bpb"] = df[f"bottom_conf_{label}"] * bpb_coefficient
        df[f"mean_{label}_bpb"] = df[f"mean_{label}"] * bpb_coefficient

        for i, model in enumerate(df["pretty_model_name"].unique()):
            df_model = df[df["pretty_model_name"] == model]
            color = (
                px.colors.qualitative.Plotly[i]
                if qualitative
                else px.colors.sequential.Plasma_r[i]
            )
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
        width=1000,
        height=800,
        legend=dict(
            title="Pythia loss",
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
    fig.update_yaxes(range=[1.8, 5], row=1, col=1)
    fig.update_yaxes(range=[0, 4.5], row=2, col=1)
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

    fig.write_image(image_name, format="pdf")


def plot_model_sizes(bpb_coefficient: float, entropies_bpb: list[float]):
    num_samples = 1024
    model_series = "Pythia"
    os.makedirs(Path.cwd() / "images", exist_ok=True)

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
        # ("es_pythia-14m", "14M"),
        # ("es_pythia-70m", "70M"),
        # ("es_pythia-160m", "160M"),
        # ("es_pythia-410m", "410M"),
        # ("es_pythia-1b", "1B"),
        # ("es_pythia-1.4b", "1.4B"),
    ]
    model_dfs = []
    for model_name, pretty_model_name in model_metadata:
        dfs = []
        model_df = pd.read_csv(
            Path.cwd()
            / "output"
            / "24-06-05"
            / f"means_ngrams_model_{model_name}_{num_samples}.csv"
        )

        dfs.append(model_df)

        model_df["model_name"] = model_name
        model_df["pretty_model_name"] = pretty_model_name

        model_dfs.append(model_df)
    df = pd.concat(model_dfs)

    image_name = Path.cwd() / "images" / "combined-ngram-data-bpb.pdf"

    plot_bpb_and_divergences(
        df, image_name, bpb_coefficient, model_series, entropies_bpb
    )


def main():
    # es_bpb_coefficient = 0.4157027
    # es_entropies_bpb = [2.72, 1.50]

    pile_bpb_coefficient = 0.3650388
    pile_entropies_bpb = [2.89, 2.04]

    plot_model_sizes(pile_bpb_coefficient, pile_entropies_bpb)


if __name__ == "__main__":
    main()
