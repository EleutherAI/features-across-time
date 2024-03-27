import os
from pathlib import Path
import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plot_ngram import base_2_log_ticks, hex_to_rgba, write_garbage
from plotly.subplots import make_subplots

from scriptutils.experiment import Experiment
from scriptutils.load_model import get_neo_tokenizer, get_zyphra_mamba, get_hails_mamba


def add_steps(df, supplementary_path):
    if os.path.exists(supplementary_path):
        print("supplementary data detected, merging...")
        supplementary_df = pd.read_csv(supplementary_path)
        df = pd.concat([df, supplementary_df])
    return df


def main(debug: bool, experiment: Experiment):
    if not debug:
        write_garbage()

    os.makedirs(Path.cwd() / "images", exist_ok=True)
    image_name = Path.cwd() / "images" / f"in-context-{experiment.team}--{experiment.model_name}.pdf"

    bpb_coefficient = 0.3650388
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

    df = pd.read_csv(Path.cwd() / "output" / f"{experiment.model_name}_{experiment.num_samples}_steps.csv")
    supplementary_path = (
        Path.cwd() / "output" / f"{experiment.model_name}_{experiment.num_samples}_steps_additional.csv"
    )
    df = add_steps(df, supplementary_path)
    tick_values, tick_texts = base_2_log_ticks(df["index"])

    # Add missing step column
    # steps = [16, 256, 1000, 8000, 33_000, 66_000, 131_000, 143_000]
    # group_numbers = df["index"].eq(0).cumsum() - 1
    # df["step"] = group_numbers.apply(
    #     lambda x: experiment.steps[x] if x < len(experiment.steps) else experiment.steps[-1]
    # )

    # Remove a step or two because the confidence intervals overlap too much
    df = df[df["step"] != 0]
    df = df[df["step"] != 2]
    df = df[df["step"] != 8]
    df = df[df["step"] != 32]
    df = df[df["step"] != 128]
    df = df[df["step"] != 512]
    df = df[df["step"] != 2048]
    df = df[df["step"] != 20_000]
    df = df[df["step"] != 80_000]
    df = df[df["step"] != 320_000]


    # Make index 1-indexed
    df["index"] = df["index"] + 1

    # Filter to log spaced samples
    max_index = df["index"].max()
    log_spaced_indices = np.unique(
        np.logspace(
            0, np.log2(max_index), num=int(np.log2(max_index)) + 1, base=2
        ).astype(int)
    )
    df = df[df["index"].isin(log_spaced_indices)]

    # Order data
    step_order = sorted(df["step"].unique())

    fig = make_subplots(
        rows=1,
        cols=2,
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=[
            "Unigram loss over sequence positions",
            "Bigram loss over sequence positions",
        ],
        horizontal_spacing=0.02,
        vertical_spacing=0.05,
    )

    for idx, ngram in enumerate(["unigram", "bigram"]):
        df[f"mean_{ngram}_bpb"] = df[f"mean_{ngram}_loss"] * bpb_coefficient
        df[f"bottom_conf_{ngram}_bpb"] = (
            df[f"bottom_conf_{ngram}_loss"] * bpb_coefficient
        )
        df[f"top_conf_{ngram}_bpb"] = df[f"top_conf_{ngram}_loss"] * bpb_coefficient

        # For some reason the legend items are listed in reverse order
        for i, step in enumerate(reversed(step_order)):
            group_df = df[df["step"] == step]
            # (px.colors.qualitative.Plotly + px.colors.qualitative.D3)
            color = px.colors.sequential.Plasma_r[i]
            transparent_color = hex_to_rgba(color, opacity=0.17)

            fig.add_trace(
                go.Scatter(
                    x=group_df["index"],
                    y=group_df[f"top_conf_{ngram}_bpb"],
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
                    x=group_df["index"],
                    y=group_df[f"bottom_conf_{ngram}_bpb"],
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
                    x=group_df["index"],
                    y=group_df[f"mean_{ngram}_bpb"],
                    mode="lines+markers",
                    marker=dict(size=5, symbol=marker_series[i]),
                    name=f"{step:,}",
                    line=dict(color=color),
                    showlegend=idx == 1,
                ),
                row=1,
                col=idx + 1,
            )

        fig.add_shape(
            type="line",
            x0=1,
            y0=entropies[idx],
            x1=2**11,
            y1=entropies[idx],
            line=dict(color="black", width=2, dash="dot"),
            row=1,
            col=idx + 1,
        )

    fig.update_layout(
        width=1000,
        height=400,
        legend=dict(
            x=0.98,
            y=0.6,
            xanchor="right",
            yanchor="middle",
            font=dict(size=8),
            title="Step",
            bgcolor="rgba(255, 255, 255, 0.5)",
        ),
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
            text="Token index",
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.2,
            showarrow=False,
            font=dict(size=14),
        )
    )
    fig.write_image(image_name, format="pdf")

   
if __name__ == "__main__":
    experiment = Experiment(
        num_samples=1024,
        team="hails", 
        model_name="mamba-160m-hf", 
        batch_size=1,
        seq_len=2049, 
        steps=[0, 1, 2, 4, 8, 16, 256, 1000, 8000, 33_000, 66_000, 131_000, 143_000],
        d_vocab=50_277,
        get_model=get_hails_mamba, 
        get_tokenizer=get_neo_tokenizer,
        eod_index=0
    )
    # experiment = Experiment(
    #     num_samples=1024,
    #     batch_size=2, 
    #     seq_len=2049, 
    #     team="Zyphra", 
    #     model_name="Mamba-370M", 
    #     get_model=get_zyphra_mamba, 
    #     get_tokenizer=get_neo_tokenizer,
    #     d_vocab=50_277, # len(get_neo_tokenizer().vocab) 
    #     # roughly log spaced steps + final step
    #     steps=[2**i for i in range(int(math.log2(2048)) + 1)] + [10_000, 20_000, 40_000, 80_000, 160_000, 320_000, 610_000],
    #     eod_index=get_neo_tokenizer().eos_token_id
    # )
    main(debug=False, experiment=experiment)


