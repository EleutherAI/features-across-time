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

def adjust_confidence_intervals(
    df, mean_col: str, bottom_conf_col: str, top_conf_col: str, sample_size=2048
):
    """Adjust confidence intervals upwards for data with n token positions passed in"""
    df[top_conf_col] = df[mean_col] + ((df[top_conf_col] - df[mean_col]) * np.sqrt(
        sample_size
    ))
    df[bottom_conf_col] = df[mean_col] - ((df[mean_col] - df[bottom_conf_col]) * np.sqrt(
        sample_size
    ))
    return df


def base_2_log_ticks(values):
    max_val = np.log2(values.max())
    ticks = 2 ** np.arange(1, np.ceil(max_val) + 1)
    return ticks,  [f'2<sup>{int(i)}</sup>' for i in np.arange(1, np.ceil(max_val) + 1)]


def lighten_hex_color(hex_color, lighten_percent=0.2):
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    h, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)
    # Increase lightness
    l = min(1, l + lighten_percent)
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    # Convert RGB back to hex
    return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'


def hex_to_rgba(hex_color, opacity=0.5):
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    return f'rgba({r}, {g}, {b}, {opacity})'


def plot_bpb_subplots(df, num_samples=1024):
    # Garbage data to work around Kaleido bug: https://github.com/plotly/plotly.py/issues/3469
    image_name = Path.cwd() / "images" / "combined-ngram-data-bpb.pdf"
    fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
    fig.write_image(image_name, format="pdf")
    time.sleep(2)

    tick_values, tick_texts = base_2_log_ticks(df["step"])
    bpb_coefficient = 0.3366084909549386 / math.log(2)

    fig = make_subplots(rows=1, cols=2, shared_xaxes=True, shared_yaxes=True, subplot_titles=("Unigram model sequences over training", "Bigram model sequences over training"), horizontal_spacing=0.02)
    for idx, ngram in enumerate(["unigram", "bigram"]):
        df[f"mean_{ngram}_bpb"] = df[f"mean_{ngram}_loss"] * bpb_coefficient
        df[f"mean_{ngram}_bpb_bottom_conf"] = df[f"bottom_conf_{ngram}_loss"] * bpb_coefficient
        df[f"mean_{ngram}_bpb_top_conf"] = df[f"top_conf_{ngram}_loss"] * bpb_coefficient
        if ngram == "unigram":
            adjust_models = ["pythia-14m", "pythia-70m", "pythia-160m", "pythia-410m", "pythia-1b", "pythia-1.4b"]
            mask = df['model_name'].isin(adjust_models)
            df.loc[mask] = adjust_confidence_intervals(df[mask], f"mean_{ngram}_bpb", f"mean_{ngram}_bpb_bottom_conf", f"mean_{ngram}_bpb_top_conf")

        for i, model in enumerate(df['pretty_model_name'].unique()):
            df_model = df[df['pretty_model_name'] == model]
            color = px.colors.sequential.Plasma_r[i + 1]
            transparent_color = hex_to_rgba(color, opacity=0.2)

            fig.add_trace(go.Scatter(x=df_model['step'], y=df_model[f'mean_{ngram}_bpb_top_conf'], fill=None, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1, col=idx+1)
            fig.add_trace(go.Scatter(x=df_model['step'], y=df_model[f'mean_{ngram}_bpb_bottom_conf'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor=transparent_color, showlegend=False, hoverinfo='skip'), row=1, col=idx+1)
            fig.add_trace(go.Scatter(x=df_model['step'], y=df_model[f'mean_{ngram}_bpb'], mode='lines', name=model, line=dict(color=color), showlegend=idx==0), row=1, col=idx+1)

    fig.update_layout(
        # yaxis_title="bits per byte",
        xaxis_title="training step (1 step = 2<sup>21</sup> tokens)",
        yaxis_range=[2, 7.5],
        legend=dict(x=0.95, y=0.95, xanchor='right', yanchor='top', font=dict(size=10)),
        height=450,
        width=1000
    )
    # fig.update_yaxes(title_text="bits per byte", row=1, col=1)
    fig.update_xaxes(title_text="", row=1, col=1)
    fig.update_xaxes(title_text="", row=1, col=2)
    fig.update_yaxes(title_text="bits per byte", row=1, col=1)

    # Add a shared, centered x-axis label using an annotation
    fig.add_annotation(
        dict(
            text="training step (1 step = 2<sup>21</sup> tokens)",
            xref="paper", yref="paper",
            x=0.5, y=-0.4,
            showarrow=False,
            font=dict(size=12)
        )
    )

    fig.update_yaxes(showticklabels=False, row=1, col=2)
    fig.update_xaxes(type="log", tickvals=tick_values, ticktext=tick_texts)

    fig.write_image(image_name, format="pdf")


def plot_divergence_subplots(df, num_samples=1024):
    # Garbage data to work around Kaleido bug: https://github.com/plotly/plotly.py/issues/3469
    image_name = Path.cwd() / "images" / "combined-divergences.pdf"
    fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
    fig.write_image(image_name, format="pdf")
    time.sleep(2)

    tick_values, tick_texts = base_2_log_ticks(df["step"])
    bpb_coefficient = 0.3366084909549386 / math.log(2)

    div_metadata = [
        ("bigram_logit_kl_div", f"$D_{{KL}}(\\text{{{'bigram model || Pythia'}}})$", [0, 7], 1, 1),
        ("unigram_logit_kl_div", f"$D_{{KL}}(\\text{{{'unigram model || Pythia'}}})$", [0, 7], 1, 2),
        ("unigram_logit_js_div", f"$D_{{JS}}(\\text{{{'unigram model || Pythia'}}})$", [0, 0.65], 2, 1),
        ("bigram_logit_js_div", f"$D_{{JS}}(\\text{{{'bigram model || Pythia'}}})$", [0, 0.65], 2, 2),
        ("bigram_token_js_div", f"$D_{{JS}}(\\text{{{'bigram model || tokens'}}})$", [0, 1.2], 3, 1),
        ("logit_token_js_div", f"$D_{{JS}}(\\text{{{'Pythia || token'}}})$", [0, 1.2], 3, 2)
    ]
    fig = make_subplots(
        rows=3, 
        cols=2, 
        shared_xaxes=True, 
        shared_yaxes=True, 
        subplot_titles=(
            [label[1] for label in div_metadata]
        ), 
        horizontal_spacing=0.02,
        vertical_spacing=0.1
    )
    for idx, (label, pretty_label, y_range, row, col) in enumerate(div_metadata):
        for i, model in enumerate(df['pretty_model_name'].unique()):
            df_model = df[df['pretty_model_name'] == model]
            color = px.colors.sequential.Plasma_r[i + 1]
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
        dict(
            # legend=dict(x=0.95, y=0.95, xanchor='right', yanchor='top'),
            height=600
        ),
    )
    fig.update_xaxes(title_text="", row=1, col=1)
    fig.update_xaxes(title_text="", row=1, col=2)
    fig.update_yaxes(title_text="divergence", row=1, col=1)
    fig.update_yaxes(title_text="divergence", row=2, col=1)
    fig.update_yaxes(title_text="divergence", row=3, col=1)

    fig.update_xaxes(type="log", tickvals=tick_values, ticktext=tick_texts)

    # Add a shared, centered x-axis label using an annotation
    fig.add_annotation(
        dict(
            text="training step (1 step = 2<sup>21</sup> tokens)",
            xref="paper", yref="paper",
            x=0.5, y=-0.12,
            showarrow=False,
            font=dict(size=12)
        )
    )
    fig.update_annotations(font=dict(size=12))


    fig.write_image(image_name, format="pdf")



def plot_bpb(df, num_samples=1024):
    tick_values, tick_texts = base_2_log_ticks(df["step"])
    bpb_coefficient = 0.3366084909549386 / math.log(2)

    for ngram in ["unigram", "bigram"]:
        # Garbage data to work around Kaleido bug: https://github.com/plotly/plotly.py/issues/3469
        image_name = Path.cwd() / "images" / f"{ngram}-data-bpb.pdf"
        fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
        fig.write_image(image_name, format="pdf")
        time.sleep(2)

        # Draw plot
        df[f"mean_{ngram}_bpb"] = df[f"mean_{ngram}_loss"] * bpb_coefficient
        df[f"mean_{ngram}_bpb_bottom_conf"] = (
            df[f"bottom_conf_{ngram}_loss"] * bpb_coefficient
        )
        df[f"mean_{ngram}_bpb_top_conf"] = df[f"top_conf_{ngram}_loss"] * bpb_coefficient
        if ngram == "unigram":
            adjust_models = ["pythia-14m", "pythia-70m", "pythia-160m", "pythia-410m", "pythia-1b"]
            mask = df['model_name'].isin(adjust_models)
            df.loc[mask] = adjust_confidence_intervals(df[mask], f"mean_{ngram}_bpb", f"mean_{ngram}_bpb_bottom_conf", f"mean_{ngram}_bpb_top_conf")

        fig = go.Figure()
        for i, model in enumerate(df['pretty_model_name'].unique()):
            df_model = df[df['pretty_model_name'] == model]
            color = px.colors.sequential.Plasma_r[i + 1] # Replace with dynamic color assignment if needed
            transparent_color = hex_to_rgba(color, opacity=0.2)

            fig.add_trace(go.Scatter(x=df_model['step'], y=df_model[f'mean_{ngram}_bpb_top_conf'], fill=None, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=df_model['step'], y=df_model[f'mean_{ngram}_bpb_bottom_conf'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor=transparent_color, showlegend=False, hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=df_model['step'], y=df_model[f'mean_{ngram}_bpb'], mode='lines', name=model, line=dict(color=color)))

        fig.update_layout(
            {   
                "title": {
                    'text': f"BPB over training on {ngram} model sequences",
                    'x': 0.5,
                    'y': 0.84,
                    'xanchor': 'center'
                },
                "yaxis_title": "bits per byte",
                "xaxis_title": f"training step (1 step = 2<sup>21</sup> tokens)",
                "yaxis_range": [2, 8],
                "legend": dict(x=0.95, y=0.95, xanchor='right', yanchor='top'),
            }
        )
        fig.update_xaxes(type="log", tickvals=tick_values, ticktext=tick_texts)
        fig.write_image(image_name, format="pdf")
        

def plot_ngram_model(df, num_samples=1024):
    tick_values, tick_texts = base_2_log_ticks(df["step"])
    bpb_coefficient = 0.3366084909549386 / math.log(2)

    div_labels = [
        ("bigram_logit_kl_div", f"$D_{{KL}}(\\text{{{'bigram model || Pythia'}}})$", [0, 7]),
        ("unigram_logit_kl_div", f"$D_{{KL}}(\\text{{{'unigram model || Pythia'}}})$", [0, 7]),
        ("unigram_logit_js_div", f"$D_{{JS}}(\\text{{{'unigram model || Pythia'}}})$", [0, 0.65]),
        ("bigram_logit_js_div", f"$D_{{JS}}(\\text{{{'bigram model || Pythia'}}})$", [0, 0.65]),
        ("bigram_token_js_div", f"$D_{{JS}}(\\text{{{'bigram model || token'}}})$", [0, 1.2]),
        ("logit_token_js_div", f"$D_{{JS}}(\\text{{{'Pythia || token'}}})$", [0, 1.2])
    ]

    for (label, pretty_label, y_range) in div_labels:
        # Garbage data to work around Kaleido bug: https://github.com/plotly/plotly.py/issues/3469
        image_name = Path.cwd() / "images" / f"{label.replace('_', '-')}.pdf"
        fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
        fig.write_image(image_name, format="pdf")
        time.sleep(2)

        # Draw plot
        fig = go.Figure()
        for i, model in enumerate(df['pretty_model_name'].unique()):
            df_model = df[df['pretty_model_name'] == model]
            color = px.colors.sequential.Plasma_r[i + 1] # Replace with dynamic color assignment if needed
            transparent_color = hex_to_rgba(color, opacity=0.2)

            fig.add_trace(go.Scatter(x=df_model['step'], y=df_model[f'top_conf_{label}'], fill=None, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=df_model['step'], y=df_model[f'bottom_conf_{label}'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor=transparent_color, showlegend=False, hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=df_model['step'], y=df_model[f'mean_{label}'], mode='lines', name=model, line=dict(color=color)))

        fig.update_layout(
            {
                "title": {
                    'text': pretty_label, # f"$D_{{KL}}$({bigrams} || {logits})" # f"$D_{{KL}}(\t{{{bigrams} \| {logits}}})$", # f"D<sub>KL</sub>({pretty_label}) over training", 
                    'x': 0.5,
                    'y': 0.84,
                    'xanchor': 'center'
                },
                "yaxis_title": "Mean divergence",
                "xaxis_title": "Training step (1 step = 2<sup>21</sup> tokens)",
                "yaxis_range": y_range,
                "legend": dict(x=0.95, y=0.95, xanchor='right', yanchor='top'),
            }
        )
        fig.update_xaxes(type="log", tickvals=tick_values, ticktext=tick_texts)
        fig.write_image(image_name, format="pdf")

def main():
    num_samples = 1024
    os.makedirs(Path.cwd() / "images", exist_ok=True)

    model_metadata = [
        ("pythia-14m", "14M"),
        ("pythia-70m", "70M"),
        ("pythia-160m", "160M"),
        ("pythia-410m", "410M"),
        ("pythia-1b", "1B"),
        ("pythia-1.4b", "1.4B"),
        ("pythia-2.8b", "2.8B"),
        # ("pythia-6.9b", "6.9B"),
        # ("pythia-12b", "12B"),
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

    # plot_ngram_model(df)
    plot_bpb(df)
    # plot_bpb_subplots(df)
    plot_divergence_subplots(df)


if __name__ == "__main__":
    main()
