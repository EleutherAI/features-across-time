import argparse
import math
import os
import pickle
from pathlib import Path
import colorsys
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def base_2_log_ticks(values):
    max_val = np.log2(values.max())
    ticks = 2 ** np.arange(1, np.ceil(max_val) + 1)
    return ticks,  [f'2<sup>{int(i)}</sup>' for i in np.arange(0, np.ceil(max_val) + 1)]


def hex_to_rgba(hex_color, opacity=0.5):
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    return f'rgba({r}, {g}, {b}, {opacity})'


def plot_bpb_and_divergences(df: pd.DataFrame, image_name: str, debug: bool, qualitative=False):
    if not debug:
        # Garbage data to work around Kaleido bug: https://github.com/plotly/plotly.py/issues/3469
        fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
        fig.write_image(image_name, format="pdf")
        time.sleep(2)

    tick_values, tick_texts = base_2_log_ticks(df["step"])
    bpb_coefficient = 0.3650388

    div_metadata = [
        ("unigram_logit_kl_div", f"$D_{{KL}}(\\text{{{'unigram model || Pythia'}}})$", [0, 7], 2, 2),
        ("bigram_logit_kl_div", f"$D_{{KL}}(\\text{{{'bigram model || Pythia'}}})$", [0, 7], 2, 1),
    ]
    fig = make_subplots(
        rows=2, cols=2, shared_xaxes=True, shared_yaxes=True, 
        subplot_titles=["Unigram sequence loss across time", "Bigram sequence loss across time"] + [label[1] for label in div_metadata], 
        horizontal_spacing=0.02,
        vertical_spacing=0.05)
    for idx, ngram in enumerate(["unigram", "bigram"]):
        df[f"mean_{ngram}_bpb"] = df[f"mean_{ngram}_loss"] * bpb_coefficient
        df[f"mean_{ngram}_bpb_bottom_conf"] = df[f"bottom_conf_{ngram}_loss"] * bpb_coefficient
        df[f"mean_{ngram}_bpb_top_conf"] = df[f"top_conf_{ngram}_loss"] * bpb_coefficient

        for i, model in enumerate(df['pretty_model_name'].unique()):
            df_model = df[df['pretty_model_name'] == model]
            color = px.colors.qualitative.Plotly[i]  if qualitative else px.colors.sequential.Plasma_r[i + 1] 
            transparent_color = hex_to_rgba(color, opacity=0.2)

            fig.add_trace(go.Scatter(x=df_model['step'], y=df_model[f'mean_{ngram}_bpb_top_conf'], fill=None, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1, col=idx + 1)
            fig.add_trace(go.Scatter(x=df_model['step'], y=df_model[f'mean_{ngram}_bpb_bottom_conf'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor=transparent_color, showlegend=False, hoverinfo='skip'), row=1, col=idx + 1)
            fig.add_trace(go.Scatter(x=df_model['step'], y=df_model[f'mean_{ngram}_bpb'], mode='lines', name=model, line=dict(color=color), showlegend=idx==1), row=1, col=idx + 1)

    for label, pretty_label, y_range, row, col in div_metadata:
        for i, model in enumerate(df['pretty_model_name'].unique()):
            df_model = df[df['pretty_model_name'] == model]
            color = px.colors.qualitative.Plotly[i]  if qualitative else px.colors.sequential.Plasma_r[i + 1] 
            transparent_color = hex_to_rgba(color, opacity=0.2)
            fig.add_trace(
                go.Scatter(x=df_model['step'], y=df_model[f'top_conf_{label}'], fill=None, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'),
                row=row,
                col=col)
            fig.add_trace(
                go.Scatter(x=df_model['step'], y=df_model[f'bottom_conf_{label}'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor=transparent_color, showlegend=False, hoverinfo='skip'), 
                row=row,
                col=col)
            fig.add_trace(
                go.Scatter(x=df_model['step'], y=df_model[f'mean_{label}'], mode='lines', name=model, line=dict(color=color), showlegend=False), # row==1 and col==2
                row=row,
                col=col)

    fig.update_layout(
        width=1000, 
        height=1000, 
        legend=dict(x=0.98, y=0.98, xanchor='right', yanchor='top', font=dict(size=8), bgcolor='rgba(255, 255, 255, 0.85)')
    )

    fig.update_xaxes(title_text="", type="log", tickvals=tick_values, ticktext=tick_texts)
    
    fig.update_yaxes(title_text="Loss (bits per byte)", title_font=dict(size=12), title_standoff=10, row=1, col=1)
    fig.update_yaxes(title_text="Divergence (bits per byte)", title_font=dict(size=12), title_standoff=10, row=2, col=1)
    fig.update_yaxes(range=[0.3, 0.8], row=3)
    # Add a shared, centered x-axis label
    fig.add_annotation(
        dict(
            text="training step (1 step = 2<sup>21</sup> tokens)",
            xref="paper", yref="paper",
            x=0.5, y=-0.05,
            showarrow=False,
            font=dict(size=12)
        )
    )

    fig.write_image(image_name, format="pdf")


def plot_model_sizes(debug: bool):
    bpb_num_samples = 1024
    js_num_samples = 4096
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
            Path.cwd() / "output" / f"means_ngrams_model_{model_name}_{bpb_num_samples}.csv"
        )
        model_df['model_name'] = model_name
        model_df['pretty_model_name'] = pretty_model_name

        model_dfs.append(model_df)
    df = pd.concat(model_dfs)
    
    image_name = Path.cwd() / "images" / "combined-ngram-data-bpb.pdf"
    plot_bpb_and_divergences(df, image_name, debug)


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

        model_df['model_name'] = model_name
        model_df['pretty_model_name'] = pretty_model_name

        model_dfs.append(model_df)
    df = pd.concat(model_dfs)

    image_name = Path.cwd() / "images" / "warmups-14m.pdf"
    plot_bpb_and_divergences(df, image_name, debug, qualitative=True)

    model_metadata = [
        ("pythia-70m", "70M (fast warmup)"),
        ("pythia-70m-warmup01", "70M (slow warmup)"),
    ]
    model_dfs = []
    for model_name, pretty_model_name in model_metadata:
        model_df = pd.read_csv(
            Path.cwd() / "output" / f"means_ngrams_model_{model_name}_{num_samples}.csv"
        )

        model_df['model_name'] = model_name
        model_df['pretty_model_name'] = pretty_model_name

        model_dfs.append(model_df)
    df = pd.concat(model_dfs)

    image_name = Path.cwd() / "images" / "warmups-70m.pdf"
    plot_bpb_and_divergences(df, image_name, debug, qualitative=True)


def main():
    debug = True

    plot_model_sizes(debug)
    plot_warmups(debug)


if __name__ == "__main__":
    main()
