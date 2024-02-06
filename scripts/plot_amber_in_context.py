import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def plot_step_index_heatmap(df: pd.DataFrame, z_label: str):
    return go.Figure(
        [
            # TODO: Fix the obvious undefined name here; I'm pretty sure this was
            # originally run in a Jupyter notebook, so that the function could capture
            # filtered_df from the global scope.
            go.Heatmap(
                {
                    "x": filtered_df["step"],
                    "y": filtered_df["index"],
                    "z": filtered_df[z_label],
                }
            )
        ],
        layout=dict(title=f"{label}", xaxis_title="step", yaxis_title="token index"),
    )


def main():
    num_samples = 1024
    bpb_coefficient = 0.3650388

    os.makedirs(Path.cwd() / "images", exist_ok=True)

    df = pd.read_csv(Path.cwd() / "output" / f"Amber_{num_samples}_steps_indices.csv")

    # Add missing step column
    steps = ["000"] + [f"{2**i:03}" for i in range(int(math.log2(359)) + 1)] + ["359"]
    group_numbers = df["index"].eq(0).cumsum() - 1
    df["step"] = group_numbers.apply(
        lambda x: steps[x] if x < len(steps) else steps[-1]
    )

    # Plot
    filtered_df = df[df["index"] <= 10]
    labels = ["random", "unigram", "bigram"]
    for label in labels:
        filtered_df[f"mean_{label}_bpb"] = (
            filtered_df[f"mean_{label}_loss"] * bpb_coefficient
        )
        filtered_df[f"{label}_grad"] = filtered_df.groupby("step")[
            f"mean_{label}_bpb"
        ].transform(lambda x: np.gradient(x, x.index))

        fig = plot_step_index_heatmap(df, f"mean_{label}_bpb")
        fig.write_image(Path.cwd() / "images" / f"{label}_heatmap.png")

        fig = plot_step_index_heatmap(df, f"{label}_grad")
        fig.write_image(Path.cwd() / "images" / f"{label}_gradient_heatmap.png")

        # Plot Anthropic's in-context learning metric
        ratio_df = df.groupby("step").apply(
            lambda x: x[x["index"] == 49][f"mean_{label}_loss"].values[0]
            / x[x["index"] == 499][f"mean_{label}_loss"].values[0]
            if len(x[x["index"] == 49]) > 0 and len(x[x["index"] == 499]) > 0
            else np.nan
        )
        fig = go.Figure(
            [go.Scatter(x=ratio_df.index, y=ratio_df, mode="lines")],
            layout=dict(
                title=f"Loss Ratio at 50th vs. 500th Token Index for {label}",
                xaxis_title="Step",
                yaxis_title="Loss Ratio (50th Index / 500th Index)",
            ),
        )
        fig.write_image(Path.cwd() / "images" / f"{label}_loss_ratio.png")


if __name__ == "__main__":
    main()
