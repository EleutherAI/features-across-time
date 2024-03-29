import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plot_ngram import add_kl_data, base_2_log_ticks, hex_to_rgba, write_garbage
from plotly.subplots import make_subplots


def plot_loss_and_divergences(
    df: pd.DataFrame,
    loss_image_name: str,
    divergence_image_name: str,
    debug: bool,
    qualitative=False,
):
    if not debug:
        write_garbage()

    tick_values, tick_texts = base_2_log_ticks(df["step"], step=2)
    bpb_coefficient = 0.3650388

    fig = make_subplots(
        rows=1,
        cols=2,
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=[
            "Unigram sequence loss across time",
            "Bigram sequence loss across time",
        ],
        horizontal_spacing=0.02,
        vertical_spacing=0.05,
    )

    entropies = [2.89, 2.04]
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

    for idx, ngram in enumerate(["unigram", "bigram"]):
        df[f"mean_{ngram}_bpb"] = df[f"mean_{ngram}_loss"] * bpb_coefficient
        df[f"bottom_conf_{ngram}_bpb"] = (
            df[f"bottom_conf_{ngram}_loss"] * bpb_coefficient
        )
        df[f"top_conf_{ngram}_bpb"] = df[f"top_conf_{ngram}_loss"] * bpb_coefficient

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

        entropy = entropies[idx]
        fig.add_shape(
            type="line",
            x0=1,
            y0=entropy,
            x1=2**17,
            y1=entropy,
            line=dict(color="black", width=2, dash="dot"),
            row=1,
            col=idx + 1,
        )

    fig.update_layout(
        width=1000,
        height=400,
        legend=dict(
            x=0.98,
            y=0.65,
            xanchor="right",
            yanchor="middle",
            font=dict(size=8),
            title="Pythia loss",
            bgcolor="rgba(255, 255, 255, 0.5)",
        ),
        yaxis=dict(range=[1.5, 5.2]),
        margin=dict(l=20, r=20, t=50, b=60),
    )

    fig.update_xaxes(
        title_text="", type="log", tickvals=tick_values, ticktext=tick_texts
    )
    fig.update_yaxes(
        title_text="Loss", title_font=dict(size=12), title_standoff=10, row=1, col=1
    )
    fig.add_annotation(
        dict(
            text="Training step",
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.2,
            showarrow=False,
            font=dict(size=14),
        )
    )

    fig.write_image(loss_image_name, format="pdf")

    div_metadata = [
        (
            "unigram_logit_kl_div",
            "D<sub>KL</sub>(unigram model || Pythia) across time",
            [0, 7],
            1,
            2,
        ),
        (
            "bigram_logit_kl_div",
            "D<sub>KL</sub>(bigram model || Pythia) across time",
            [0, 7],
            1,
            1,
        ),
    ]
    fig = make_subplots(
        rows=1,
        cols=2,
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=[label[1] for label in div_metadata],
        horizontal_spacing=0.02,
        vertical_spacing=0.05,
    )

    for label, pretty_label, y_range, row, col in div_metadata:
        for i, model in enumerate(df["pretty_model_name"].unique()):
            df_model = df[df["pretty_model_name"] == model]
            df_model[f"mean_{label}_bpb"] = df_model[f"mean_{label}"] * bpb_coefficient
            df_model[f"top_conf_{label}_bpb"] = (
                df_model[f"top_conf_{label}"] * bpb_coefficient
            )
            df_model[f"bottom_conf_{label}_bpb"] = (
                df_model[f"bottom_conf_{label}"] * bpb_coefficient
            )

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
                    marker=dict(size=5),
                    name=model,
                    line=dict(color=color),
                    showlegend=False,
                ),  # col==2
                row=row,
                col=col,
            )

    fig.update_layout(
        width=1000,
        height=400,
        # legend=dict(
        #     x=0.98,
        #     y=0.6,
        #     xanchor='right',
        #     yanchor='middle',
        #     font=dict(size=8),
        #     title="Pythia loss",
        #     bgcolor='rgba(255, 255, 255, 0.5)'
        # ),
        margin=dict(l=20, r=20, t=50, b=60),
    )

    fig.update_xaxes(
        title_text="", type="log", tickvals=tick_values, ticktext=tick_texts
    )
    fig.update_yaxes(
        title_text="KL divergence",
        title_font=dict(size=12),
        title_standoff=10,
        row=1,
        col=1,
    )
    fig.add_annotation(
        dict(
            text="Training step",
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.2,
            showarrow=False,
            font=dict(size=14),
        )
    )

    fig.write_image(divergence_image_name, format="pdf")


def plot_model_sizes(debug: bool):
    bpb_num_samples = 1024
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
    ]
    model_dfs = []
    for model_name, pretty_model_name in model_metadata:
        model_df = pd.read_csv(
            Path.cwd()
            / "output"
            / f"means_ngrams_model_{model_name}_{bpb_num_samples}.csv"
        )
        supplementary_kl_div_path = (
            Path.cwd()
            / "output"
            / f"means_ngrams_model_{model_name}_{bpb_num_samples}_kl_div.csv"
        )
        model_df = add_kl_data(model_df, supplementary_kl_div_path)

        model_df["step"] = model_df["step"] + 1
        model_df["model_name"] = model_name
        model_df["pretty_model_name"] = pretty_model_name

        model_dfs.append(model_df)
    df = pd.concat(model_dfs)

    loss_image_name = Path.cwd() / "images" / "ngram-loss.pdf"
    divergence_image_name = Path.cwd() / "images" / "ngram-divergence.pdf"
    plot_loss_and_divergences(df, loss_image_name, divergence_image_name, debug)


def plot_warmups(debug: bool):
    num_samples = 1024

    model_metadata = [
        ("pythia-14m", "14M (fast warmup)"),
        ("pythia-14m-warmup01", "14M (slow warmup)"),
    ]
    model_dfs = []
    for model_name, pretty_model_name in model_metadata:
        model_df = pd.read_csv(
            Path.cwd() / "output" / f"means_ngrams_model_{model_name}_{num_samples}.csv"
        )
        supplementary_kl_div_path = (
            Path.cwd()
            / "output"
            / f"means_ngrams_model_{model_name}_{num_samples}_kl_div.csv"
        )
        if os.path.exists(supplementary_kl_div_path):
            print("supplementary data detected, merging...")
            supplementary_kl_div_df = pd.read_csv(supplementary_kl_div_path)
            model_df["mean_unigram_logit_kl_div"] = supplementary_kl_div_df[
                "mean_unigram_logit_kl_div"
            ]
            model_df["mean_bigram_logit_kl_div"] = supplementary_kl_div_df[
                "mean_bigram_logit_kl_div"
            ]

        model_df["step"] = model_df["step"] + 1
        model_df["model_name"] = model_name
        model_df["pretty_model_name"] = pretty_model_name

        model_dfs.append(model_df)
    df = pd.concat(model_dfs)

    divergence_name = Path.cwd() / "images" / "warmups-14m-divergence.pdf"
    loss_name = Path.cwd() / "images" / "warmups-14m-loss.pdf"
    plot_loss_and_divergences(df, loss_name, divergence_name, debug, qualitative=True)

    model_metadata = [
        ("pythia-70m", "70M (fast warmup)"),
        ("pythia-70m-warmup01", "70M (slow warmup)"),
    ]
    model_dfs = []
    for model_name, pretty_model_name in model_metadata:
        model_df = pd.read_csv(
            Path.cwd() / "output" / f"means_ngrams_model_{model_name}_{num_samples}.csv"
        )
        supplementary_kl_div_path = (
            Path.cwd()
            / "output"
            / f"means_ngrams_model_{model_name}_{num_samples}_kl_div.csv"
        )
        if os.path.exists(supplementary_kl_div_path):
            print("supplementary data detected, merging...")
            supplementary_kl_div_df = pd.read_csv(supplementary_kl_div_path)
            model_df["mean_unigram_logit_kl_div"] = supplementary_kl_div_df[
                "mean_unigram_logit_kl_div"
            ]
            model_df["mean_bigram_logit_kl_div"] = supplementary_kl_div_df[
                "mean_bigram_logit_kl_div"
            ]

        model_df["step"] = model_df["step"] + 1
        model_df["model_name"] = model_name
        model_df["pretty_model_name"] = pretty_model_name

        model_dfs.append(model_df)
    df = pd.concat(model_dfs)

    loss_name = Path.cwd() / "images" / "warmups-70m-loss.pdf"
    divergence_name = Path.cwd() / "images" / "warmups-70m-divergence.pdf"
    plot_loss_and_divergences(df, loss_name, divergence_name, debug, qualitative=True)


def plot_warmups(debug: bool):
    num_samples = 1024

    model_metadata = [
        ("pythia-14m", "14M (fast warmup)"),
        ("pythia-14m-warmup01", "14M (slow warmup)"),
    ]
    model_dfs = []
    for model_name, pretty_model_name in model_metadata:
        model_df = pd.read_csv(
            Path.cwd() / "output" / f"means_ngrams_model_{model_name}_{num_samples}.csv"
        )
        supplementary_kl_div_path = (
            Path.cwd()
            / "output"
            / f"means_ngrams_model_{model_name}_{num_samples}_kl_div.csv"
        )
        model_df = add_kl_data(model_df, supplementary_kl_div_path)

        model_df["step"] = model_df["step"] + 1
        model_df["model_name"] = model_name
        model_df["pretty_model_name"] = pretty_model_name

        model_dfs.append(model_df)
    df = pd.concat(model_dfs)

    divergence_name = Path.cwd() / "images" / "warmups-14m-divergence.pdf"
    loss_name = Path.cwd() / "images" / "warmups-14m-loss.pdf"
    plot_loss_and_divergences(df, loss_name, divergence_name, debug, qualitative=True)

    model_metadata = [
        ("pythia-70m", "70M (fast warmup)"),
        ("pythia-70m-warmup01", "70M (slow warmup)"),
    ]
    model_dfs = []
    for model_name, pretty_model_name in model_metadata:
        model_df = pd.read_csv(
            Path.cwd() / "output" / f"means_ngrams_model_{model_name}_{num_samples}.csv"
        )
        supplementary_kl_div_path = (
            Path.cwd()
            / "output"
            / f"means_ngrams_model_{model_name}_{num_samples}_kl_div.csv"
        )
        model_df = add_kl_data(model_df, supplementary_kl_div_path)

        model_df["step"] = model_df["step"] + 1
        model_df["model_name"] = model_name
        model_df["pretty_model_name"] = pretty_model_name

        model_dfs.append(model_df)
    df = pd.concat(model_dfs)

    loss_name = Path.cwd() / "images" / "warmups-70m-loss.pdf"
    divergence_name = Path.cwd() / "images" / "warmups-70m-divergence.pdf"
    plot_loss_and_divergences(df, loss_name, divergence_name, debug, qualitative=True)


def main():
    debug = False

    plot_model_sizes(debug)
    plot_warmups(debug)


if __name__ == "__main__":
    main()
