import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import torch
from scipy import stats


def mean_vector(
    losses: list[torch.tensor],
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    vectors = [loss.detach().numpy() for loss in losses]
    max_length = max(len(v) for v in vectors)

    confidence_level = 0.95
    padded_vectors = [
        np.pad(v, (0, max_length - len(v)), constant_values=np.nan) for v in vectors
    ]
    stacked_vectors = np.vstack(padded_vectors)

    means = np.nanmean(stacked_vectors, axis=0)
    standard_errors = stats.sem(stacked_vectors, axis=0, nan_policy="omit")
    confidence_level = 0.95
    conf_intervals = stats.norm.interval(
        confidence_level, loc=means, scale=standard_errors
    )

    return means, standard_errors, conf_intervals


def main():
    steps = []
    mean_per_token_losses = []
    standard_error_per_token = []
    mean_aggregated_losses = []
    for root, dirs, files in os.walk("output"):
        for file in files:
            (step, token_losses, token_bow_losses) = torch.load(
                os.path.join(root, file)
            )
            mean_bow_loss, standard_errors, conf_intervals = mean_vector(
                token_bow_losses
            )

            standard_error_per_token.append(standard_errors)
            steps.append(step)
            mean_per_token_losses.append(mean_bow_loss.detach().numpy())
            mean_aggregated_losses.append(mean_per_token_losses[-1].mean())

            mean_token_loss, standard_errors, conf_intervals = mean_vector(token_losses)

    df = pd.DataFrame(
        {
            "step": steps,
            "mean_per_token_loss": mean_per_token_losses,
            "mean_aggregated_loss": mean_aggregated_losses,
        }
    )

    fig = px.line(df, x="step", y="mean_aggregated_loss")
    fig.write_image(Path.cwd() / "output" / "image.png")


if __name__ == "__main__":
    main()
