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

from plot_ngram import base_2_log_ticks, hex_to_rgba
import tempfile

def plot_seed_loss(df: pd.DataFrame, debug: bool):
    if not debug:
        # Garbage data to work around Kaleido bug: https://github.com/plotly/plotly.py/issues/3469
        with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_file:
            fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
            fig.write_image(temp_file.name, format="pdf")
        time.sleep(2)

    tick_values, tick_texts = base_2_log_ticks(df["step"])
    bpb_coefficient = 0.3650388

    def create_row(df, title: str, name: str, ytitle, show_xaxis=False, show_legend=False):
        save_name = name.replace("_mean", "").replace("_bpb", "")
        image_name = Path.cwd() / "images" / f"seed_{save_name}.pdf"

        fig = make_subplots(
            rows=1, cols=4, shared_xaxes=True, shared_yaxes=True, 
            subplot_titles="", 
            horizontal_spacing=0.02,
            vertical_spacing=0.05)
        for model_index, model in enumerate(df['pretty_model_name'].unique()):
            df_model = df[df['pretty_model_name'] == model]
            color = px.colors.qualitative.Plotly[model_index]
            transparent_color = hex_to_rgba(color, opacity=0.2)

            for seed in df_model['seed'].unique():
                seed_df = df_model[df_model['seed'] == seed]
                fig.add_trace(go.Scatter(x=seed_df['step'], y=seed_df[name], mode='lines', name=model, line=dict(color=transparent_color), showlegend=False), row=1, col=model_index + 1)

            seed_mean = df_model.groupby('step')[name].mean()
            fig.add_trace(go.Scatter(x=df_model['step'], y=seed_mean, mode='lines+markers', name=model, line=dict(color=color), showlegend=show_legend, marker=dict(size=5)), row=1, col=model_index + 1)

            fig.update_layout(
                width=1000, 
                height=300, 
                legend=dict(x=0.98, y=0.98, xanchor='right', yanchor='top', font=dict(size=8), bgcolor='rgba(255, 255, 255, 0.85)'),
                legend_title="Pythia model",
                autosize=True,
                margin=dict(l=20, r=20, t=30, b=30)
            )

            # Title
            fig.add_annotation(
                dict(
                    text=title,
                    xref="paper", yref="paper",
                    x=0.5, y=1.15,
                    showarrow=False,
                    font=dict(size=16)
                )
            )

            # X axis title
            if show_xaxis:
                fig.add_annotation(
                    dict(
                        text="Training step", # (1 step = 2<sup>21</sup> tokens)
                        xref="paper", yref="paper",
                        x=0.5, y=-0.2,
                        showarrow=False,
                        font=dict(size=16)
                    )
                )
            fig.update_xaxes(title_text="", type="log", tickvals=tick_values[::2], ticktext=tick_texts[::2])

            fig.update_yaxes(title_text=ytitle, title_font=dict(size=12), title_standoff=10, col=1)
            fig.update_yaxes(range=[0.3, 0.8], row=3)
            fig.write_image(image_name, format="pdf")
            

    for idx, ngram in enumerate(["unigram", "bigram"]):
        df[f"mean_{ngram}_bpb"] = df[f"mean_{ngram}_loss"] * bpb_coefficient
        create_row(
            df, f"{ngram.title()} sequence loss over training", f"mean_{ngram}_bpb", ytitle="Loss", show_legend=ngram=="unigram")

    div_metadata = [
        ("unigram_logit_kl_div", "D<sub>KL</sub>(unigram model || Pythia) over training", [0, 7]),
        ("bigram_logit_kl_div", "D<sub>KL</sub>(bigram model || Pythia) over training", [0, 7]),
    ]
    for label, pretty_label, y_range in div_metadata:
        df[f'mean_{label}_bpb'] = df[f'mean_{label}'] * bpb_coefficient
        create_row(
            df, pretty_label, f'mean_{label}_bpb', ytitle="KL divergence", show_xaxis=label=="bigram_logit_kl_div")


def main():
    debug = False
    bpb_num_samples = 1024
    os.makedirs(Path.cwd() / "images", exist_ok=True)

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
                Path.cwd() / "output" / f"means_ngrams_model_{model_name}-seed{i}_{bpb_num_samples}.csv"
            )
            seed_df['seed'] = i
            seed_df['model_name'] = model_name
            seed_df['pretty_model_name'] = pretty_model_name
            seed_dfs.append(seed_df)
    df = pd.concat(seed_dfs)
    plot_seed_loss(df, debug)


if __name__ == "__main__":
    main()
