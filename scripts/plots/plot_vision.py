import os
from pathlib import Path
import math
import torch

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datasets import load_from_disk

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

def num_classes(dataset_str: str):
    ds = load_from_disk(f'/mnt/ssd-1/lucia/vision-data/max-entropy-{dataset_str}.hf')
    ds.set_format('torch', columns=['pixel_values','label'])
    return len(torch.unique(ds['label']))


def main(debug: bool):
    os.makedirs(Path.cwd() / "images", exist_ok=True)
    

    for dataset_str, pretty_dataset_str in [
        ("cifar10", "CIFAR10"), 
        # ("cifarnet", "CIFAR-Net"), 
        ("mnist", "MNIST"),
        ("fashion_mnist", "Fashion MNIST"),
        ("svhn:cropped_digits", "SVHN"),
    ]:
        model_metadata = [
            "convnext", "ConvNeXt",
            "regnet", "RegNet-Y",
            "swin", "Swin Transformer",
        ]
        data_path = Path('/') / 'mnt' / 'ssd-1' / 'lucia' / 'features-across-time' / '24-05-22'

        grayscale = "-grayscale" if (dataset_str == "mnist" or dataset_str == "fashion_mnist") else ""
        df = pd.read_csv(data_path / f'vision-{dataset_str}{grayscale}.csv')

        if dataset_str == "svhn:cropped_digits":
            df_svhn_convnext = pd.read_csv(data_path / f'vision-convnext-{dataset_str}.csv')
            df_svhn_swin = pd.read_csv(data_path / f'vision-swin-{dataset_str}.csv')
            condition = (df['ds'] == "svhn:cropped_digits") & (df['net'] == "convnext")
            condition_svhn = (df_svhn_convnext['ds'] == "svhn:cropped_digits") & (df_svhn_convnext['net'] == "convnext")
            df.loc[condition, df.columns] = df_svhn_convnext.loc[condition_svhn, df.columns].values
            condition_svhn_swin = (df_svhn_swin['ds'] == "svhn:cropped_digits") & (df_svhn_swin['net'] == "swin")
            condition = (df['ds'] == "svhn:cropped_digits") & (df['net'] == "swin")
            df.loc[condition, df.columns] = df_svhn_swin.loc[condition_svhn_swin, df.columns].values
            df.to_csv('tmp.csv')

        df["pretty_model_name"] = df["net"].map(dict(zip(model_metadata[::2], model_metadata[1::2])))
        df = df[df['ds'] == dataset_str]
        image_prefix = Path.cwd() / 'images'
        plot(dataset_str, pretty_dataset_str, df, image_prefix, debug)


def plot(
    dataset_str: str, pretty_dataset_str: str, df: pd.DataFrame, image_prefix: str, debug: bool
):
    if not debug:
        write_garbage()

    tick_values, tick_texts = base_2_log_ticks(df["step"], step=2)
    uniform_random_loss = math.log(num_classes(dataset_str))
    uniform_random_accuracy = (1 / num_classes(dataset_str))

    ot_col_names = {
        "cqn": "1st order (CQN)",
        "shifted": "1st order (Bounded shift)",
        "got": "2nd order (Gaussian OT)",
        "real": "Val. set",
    }
    maxent_col_names = {
        "independent": "1st order (ICS)",
        "maxent": "1st order (Dury)",
        "gaussian": "2nd order (Gaussian)",
        # "truncated_normal": "2nd order (Truncated normal)",
        "real": "Val. set",
    }

    for col_dict in [ot_col_names, maxent_col_names]:
        if "maxent" in col_dict:
            title = f"Max-entropy sampling across time ({pretty_dataset_str})"
        else:
            title = f"Optimal transport across time ({pretty_dataset_str})"
    
        fig = make_subplots(
            rows=2,
            cols=df["net"].nunique(),
            shared_xaxes=True,
            shared_yaxes=True,
            subplot_titles=df["pretty_model_name"].unique(),
            horizontal_spacing=0.02,
            vertical_spacing=0.04,
        )
        for i, model in enumerate(df["net"].unique()):
            model_df = df[df["net"] == model]
            
            for j, (col_name, legend_name) in enumerate(col_dict.items()):
                opaque_color = px.colors.qualitative.Plotly[j] if col_name != "real" else "#000000"
                transparent_color = hex_to_rgba(opaque_color, opacity=0.17)

                # Add mean line over model architectures
                mean_acc_per_step = model_df.groupby("step")[f"{col_name}_accuracy"].mean()
                fig.add_trace(
                    go.Scatter(
                        x=model_df[model_df["arch"] == model_df["arch"].unique()[0]]["step"],
                        y=mean_acc_per_step,
                        mode="lines+markers",
                        marker=dict(size=5, symbol=marker_series[0]),
                        name=legend_name,
                        line=dict(color=opaque_color),
                        showlegend=i==df["net"].nunique() - 1,
                    ),
                    row=1,
                    col=i + 1,
                )

                mean_loss_per_step = model_df.groupby("step")[f"{col_name}_loss"].mean()
                fig.add_trace(
                    go.Scatter(
                        x=model_df[model_df["arch"] == model_df["arch"].unique()[0]]["step"],
                        y=mean_loss_per_step,
                        mode="lines+markers",
                        marker=dict(size=5, symbol=marker_series[0]),
                        name=legend_name,
                        line=dict(color=opaque_color),
                        showlegend=False,
                    ),
                    row=2,
                    col=i + 1,
                )
            
                # Add lines for individual model architectures
                for arch in model_df["arch"].unique():
                    arch_df = model_df[model_df["arch"] == arch]

                    fig.add_trace(
                        go.Scatter(
                            x=arch_df["step"],
                            y=arch_df[f"{col_name}_accuracy"],
                            mode="lines",
                            name=arch,
                            line=dict(color=transparent_color),
                            showlegend=False,
                        ),
                        row=1,
                        col=i + 1,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=arch_df["step"],
                            y=arch_df[f"{col_name}_loss"],
                            mode="lines",
                            name=arch,
                            line=dict(color=transparent_color),
                            showlegend=False,
                        ),
                        row=2,
                        col=i + 1,
                    )
                
            # Indicate baseline loss and accuracies
            fig.add_shape(
                row=1,
                col=i + 1,
                type="line",
                x0=1,
                y0=uniform_random_accuracy,
                x1=2**16,
                y1=uniform_random_accuracy,
                line=dict(color="black", width=2, dash="dash"),
            )
            fig.add_shape(
                row=2,
                col=i + 1,
                type="line",
                x0=1,
                y0=uniform_random_loss,
                x1=2**16,
                y1=uniform_random_loss,
                line=dict(color="black", width=2, dash="dash"),
            )

        fig.update_layout(
            title=title, 
            width=1150,
            height=600,
            legend=dict(
                x=0.43,
                y=0.95,
                xanchor="center",
                yanchor="top",
                font=dict(size=8),
                bgcolor="rgba(255, 255, 255, 0.85)",
            ),
            margin=dict(l=20, r=20, t=50, b=60),
        )
        fig.update_xaxes(
            title_text="", type="log", tickvals=tick_values, ticktext=tick_texts
        )
        
        y_ticks = np.linspace(0, 1, num=6)
        ticktext = ["{:.0%}".format(tick) for tick in y_ticks]
        ticktext[0], ticktext[-1] = "", ""
        fig.update_yaxes(row=1, range=[0, 1])
        fig.update_yaxes(
            row=1, col=1, tickvals=y_ticks, ticktext=ticktext, title_text="Accuracy", tickformat=".0%", title_font=dict(size=10), title_standoff=10
        )

        fig.update_yaxes(row=2, range=[0, 10])
        fig.update_yaxes(
            row=2, col=1, title_text="Loss", title_font=dict(size=10), title_standoff=10
        )
        
        fig.add_annotation(
            dict(
                text="Training steps",
                xref="paper",
                yref="paper",
                x=0.5,
                y=-0.1,
                showarrow=False,
                font=dict(size=14),
            )
        )
        fig.write_image(image_prefix / f"vision-{dataset_str}-{'maxent' if 'maxent' in col_dict else 'ot'}.pdf", format="pdf")


if __name__ == "__main__":
    main(debug = False)
