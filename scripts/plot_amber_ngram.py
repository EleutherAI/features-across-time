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

from plot_ngram import base_2_log_ticks, hex_to_rgba, write_garbage


def plot_loss(df: pd.DataFrame, image_name: str, debug: bool):
    if not debug:
        write_garbage()

    tick_values, tick_texts = base_2_log_ticks(np.array([int(i) for i in df["step"]]))

    fig = make_subplots(
        rows=1, cols=2, shared_xaxes=True, shared_yaxes=True, 
       subplot_titles=["Unigram sequence loss across time", "Bigram sequence loss across time"],
        horizontal_spacing=0.02,
        vertical_spacing=0.05)
    for idx, ngram in enumerate(["unigram", "bigram"]):
        for i, model in enumerate(df['pretty_model_name'].unique()):
            df_model = df[df['pretty_model_name'] == model]
            color = px.colors.sequential.Plasma_r[i + 1]
            transparent_color = hex_to_rgba(color, opacity=0.2)

            fig.add_trace(go.Scatter(x=df_model['step'], y=df_model[f'bottom_conf_{ngram}_bpb'], fill=None, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1, col=idx + 1)
            fig.add_trace(go.Scatter(x=df_model['step'], y=df_model[f'top_conf_{ngram}_bpb'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor=transparent_color, showlegend=False, hoverinfo='skip'), row=1, col=idx + 1)
            fig.add_trace(go.Scatter(x=df_model['step'], y=df_model[f'mean_{ngram}_bpb'], mode='lines', name=model, line=dict(color=color), showlegend=idx==2), row=1, col=idx + 1)

    fig.update_layout(
        width=1000, 
        height=400, 
        legend=dict(x=0.98, y=0.98, xanchor='right', yanchor='top', font=dict(size=8), bgcolor='rgba(255, 255, 255, 0.85)'),
    )

    fig.update_xaxes(title_text="", type="log", tickvals=tick_values, ticktext=tick_texts)
    
    fig.update_yaxes(title_text="Loss", title_font=dict(size=12), title_standoff=10, row=1, col=1)
    fig.update_yaxes(title_text="Divergence", title_font=dict(size=12), title_standoff=10, row=2, col=1)
    fig.update_yaxes(title_text="Divergence", title_font=dict(size=12), title_standoff=10, row=3, col=1)
    fig.update_yaxes(range=[0.3, 0.8], row=3)
    # Add a shared, centered x-axis label
    fig.add_annotation(
        dict(
            text="Training step (1 step = 2<sup>21</sup> tokens)",
            xref="paper", yref="paper",
            x=0.5, y=-0.15,
            showarrow=False,
            font=dict(size=12)
        )
    )

    fig.write_image(image_name, format="pdf")


def main():
    debug = True
    image_name = Path.cwd() / "images" / "amber-ngram.pdf"
    bpb_coefficient = 0.3650388
    num_samples = 1024

    os.makedirs(Path.cwd() / "images", exist_ok=True)

    df = pd.read_csv(
        Path.cwd() / "output" / f"Amber_{num_samples}_steps_indices.csv"
    )

    # Add missing step column
    n = 2047
    steps = ["000"] + [f"{2**i:03}" for i in range(int(math.log2(359)) + 1)] + ['359']
    group_numbers = df['index'].eq(0).cumsum() - 1
    df['step'] = group_numbers.apply(lambda x: steps[x] if x < len(steps) else steps[-1])

    def decrease_confidence_intervals(
        df, mean_col: str, bottom_conf_col: str, top_conf_col: str, sample_size=9
    ):
        """Adjust confidence intervals upwards for data with n token positions passed in"""
        df[top_conf_col] = df[mean_col] + ((df[top_conf_col] - df[mean_col]) / np.sqrt(
            sample_size
        ))
        df[bottom_conf_col] = df[mean_col] - ((df[mean_col] - df[bottom_conf_col]) / np.sqrt(
            sample_size
        ))
        return df

    labels = ["random", "unigram", "bigram"]
    for label in labels:
        df[f'mean_{label}_bpb'] = df[f'mean_{label}_loss'] * bpb_coefficient
        df[f'top_conf_{label}_bpb'] = df[f'top_conf_{label}_loss'] * bpb_coefficient
        df[f'bottom_conf_{label}_bpb'] = df[f'bottom_conf_{label}_loss'] * bpb_coefficient

        df = decrease_confidence_intervals(
            df, 
            f'mean_{label}_bpb',
            f'top_conf_{label}_loss', 
            f'bottom_conf_{label}_loss',
            sample_size=n
        )

    df_agg = df.groupby('step').agg({
        **{f'mean_{label}_bpb': 'mean' for label in labels},
        **{f'top_conf_{label}_bpb': 'mean' for label in labels},
        **{f'bottom_conf_{label}_bpb': 'mean' for label in labels}
    }).reset_index()

    df_agg['pretty_model_name'] = "Amber"

    plot_loss(df_agg, image_name, debug)


if __name__ == "__main__":
    main()
