import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scripts.plots.plot_ngram import write_garbage, base_2_log_ticks, hex_to_rgba

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

def main(debug: bool):
    os.makedirs(Path.cwd() / "images", exist_ok=True)

    model_metadata = [
        "convnext", "ConvNext",
        "regnet", "RegNet",
        "swin", "Swin Transformer",
    ]
    
    df = pd.read_csv(Path('/') / 'mnt' / 'ssd-1' / 'lucia' / 'vision-natural.csv')
    maxent = pd.read_csv(Path('/') / 'mnt' / 'ssd-1' / 'lucia' / 'vision-maxent.csv')
    df['maxent_shifted_loss'] = maxent['maxent_shifted_loss']
    df["pretty_model_name"] = df["net"].map(dict(zip(model_metadata[::2], model_metadata[1::2])))

    image_prefix = Path.cwd() / 'images'
    plot(df, image_prefix, debug)


def plot(
    df: pd.DataFrame, image_prefix: str, debug: bool, qualitative=False
):
    if not debug:
        write_garbage()
    tick_values, tick_texts = base_2_log_ticks(df["step"], step=2)
    bpb_coefficient = 0.3650388
    entropies = [2.89, 2.04, 2]

    for idx, loss in enumerate(["maxent_shifted", "ds_shifted"]):
        df[f"{loss}_bpb"] = df[f"{loss}_loss"] * bpb_coefficient

        for dataset in df["ds"].unique():
            fig = make_subplots(
                rows=1,
                cols=df["net"].nunique(),
                shared_xaxes=True,
                shared_yaxes=True,
                subplot_titles=df["pretty_model_name"].unique(),
                horizontal_spacing=0.02,
                vertical_spacing=0.1,
            )
            for i, model in enumerate(df["net"].unique()):
                model_df = df[(df["ds"] == dataset) & (df["net"] == model)]
                
                color = px.colors.qualitative.Plotly[4]

                mean_loss_per_step = model_df.groupby("step")[f"{loss}_bpb"].mean()
                print(mean_loss_per_step)
                fig.add_trace(
                    go.Scatter(
                        x=model_df[model_df["arch"] == model_df["arch"].unique()[0]]["step"],
                        y=mean_loss_per_step,
                        mode="lines+markers",
                        marker=dict(size=5, symbol=marker_series[0]),
                        name="Mean",
                        line=dict(color=color),
                        showlegend=False,
                    ),
                    row=1,
                    col=i + 1,
                )
                
                for j, arch in enumerate(model_df["arch"].unique()):
                    arch_df = model_df[model_df["arch"] == arch]
                    print(len(arch_df))

                    transparent_color = hex_to_rgba(color, opacity=0.17)

                    print(arch_df[f"{loss}_bpb"])
                    print(arch_df["step"])
                    fig.add_trace(
                        go.Scatter(
                            x=arch_df["step"],
                            y=arch_df[f"{loss}_bpb"],
                            mode="lines+markers",
                            marker=dict(size=5, symbol=marker_series[j + 1]),
                            name=arch,
                            line=dict(color=transparent_color),
                            showlegend=False,
                        ),
                        row=1,
                        col=i + 1,
                    )

            if loss == "ds_shifted":
                title = "Mean shifted data across time"
            else:
                title = "Mean shifted maximum entropy data across time"
            
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
                margin=dict(l=20, r=20, t=50, b=60)
            )
            fig.update_layout(title=title)
            fig.update_xaxes(
                title_text="", type="log", tickvals=tick_values, ticktext=tick_texts
            )
            fig.update_yaxes(
                title_text="Loss", title_font=dict(size=12), title_standoff=10, row=1, col=1
            )
            max_y = 3 if loss == "maxent_shifted" else 0.01
            fig.update_yaxes(range=[0, max_y], row=1)
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
            fig.write_image(image_prefix / f"vision-{dataset}-{model}-{loss}.pdf", format="pdf")


if __name__ == "__main__":
    main(debug = False)
