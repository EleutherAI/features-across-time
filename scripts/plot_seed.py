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


def plot_loss_and_divergence(df: pd.DataFrame, debug: bool):
    image_name = Path.cwd() / "images" / "seed.pdf"
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
        ("unigram_logit_js_div", f"$D_{{JS}}(\\text{{{'unigram model || Pythia'}}})$", [0, 1], 3, 1),
        ("bigram_logit_js_div", f"$D_{{JS}}(\\text{{{'bigram model || Pythia'}}})$", [0, 1], 3, 2),
    ]
    fig = make_subplots(
        rows=3, cols=2, shared_xaxes=True, shared_yaxes=True, 
        subplot_titles=["Unigram model sequences over training", "Bigram model sequences over training"] + [label[1] for label in div_metadata], 
        horizontal_spacing=0.02,
        vertical_spacing=0.05)
    for idx, ngram in enumerate(["unigram", "bigram"]):
        df[f"mean_{ngram}_bpb"] = df[f"mean_{ngram}_loss"] * bpb_coefficient
        df[f"mean_{ngram}_bpb_bottom_conf"] = df[f"bottom_conf_{ngram}_loss"] * bpb_coefficient
        df[f"mean_{ngram}_bpb_top_conf"] = df[f"top_conf_{ngram}_loss"] * bpb_coefficient

        for i, model in enumerate(df['pretty_model_name'].unique()):
            df_model = df[df['pretty_model_name'] == model]
            color = px.colors.sequential.Plasma_r[i + 1]
            transparent_color = hex_to_rgba(color, opacity=0.2)

            for i in df_model['seed'].unique():
                seed_df = df_model[df_model['seed'] == i]
                fig.add_trace(go.Scatter(x=seed_df['step'], y=seed_df[f'mean_{ngram}_bpb'], mode='lines', name=model, line=dict(color=transparent_color), showlegend=False), row=1, col=idx + 1)

            seed_mean = df_model.groupby('step')[f'mean_{ngram}_bpb'].mean()
            print(idx)
            fig.add_trace(go.Scatter(x=df_model['step'], y=seed_mean, mode='lines', name=model, line=dict(color=color), showlegend=idx==1), row=1, col=idx + 1)

    for label, pretty_label, y_range, row, col in div_metadata:
        for i, model in enumerate(df['pretty_model_name'].unique()):
            df_model = df[df['pretty_model_name'] == model]
            color = px.colors.sequential.Plasma_r[i + 1]
            transparent_color = hex_to_rgba(color, opacity=0.2)


            for seed in df_model['seed'].unique():
                seed_df = df_model[df_model['seed'] == seed]
                fig.add_trace(go.Scatter(x=seed_df['step'], y=seed_df[f'mean_{label}'], mode='lines', name=model, line=dict(color=transparent_color), showlegend=False), row=row, col=col)

            seed_mean = df_model.groupby('step')[f'mean_{label}'].mean()
            fig.add_trace(go.Scatter(x=df_model['step'], y=seed_mean, mode='lines', name=model, line=dict(color=color), showlegend=False), row=row, col=col)

    fig.update_layout(
        width=1000, 
        height=1000, 
        legend=dict(x=0.98, y=0.98, xanchor='right', yanchor='top', font=dict(size=8), bgcolor='rgba(255, 255, 255, 0.85)')
    )


    fig.update_xaxes(title_text="", type="log", tickvals=tick_values, ticktext=tick_texts)
    
    fig.update_yaxes(title_text="bits per byte", title_font=dict(size=12), title_standoff=10, row=1, col=1)
    fig.update_yaxes(title_text="divergence", title_font=dict(size=12), title_standoff=10, row=2, col=1)
    fig.update_yaxes(title_text="divergence", title_font=dict(size=12), title_standoff=10, row=3, col=1)
    fig.update_yaxes(range=[0.3, 0.8], row=3)
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


def main():
    debug = True
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
    plot_loss_and_divergence(df, debug)


if __name__ == "__main__":
    main()
