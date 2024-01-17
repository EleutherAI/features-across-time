import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from scipy import stats


def mean_vector(
    vectors: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
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


# def gradient_loss(means):
#     derivatives = np.gradient(means)

#     fig = px.line(derivatives)
#     fig.save_image("Derivatives")
#     # each point is (step, token index)
#     # its value is the derivative of the loss wrt the log(index)
#     # avg loss at each token index -> np.gradient(means)


def main():
    dfs = []
    for root, dirs, files in os.walk("output"):
        for file in files:
            with open(os.path.join(root, file), "rb") as f:
                dfs.append(pickle.load(f))

            # dfs.append(pd.read_csv(
            #     os.path.join(root, file)
            # ))

    df = pd.concat(dfs)
    print(df.head())
    # step
    # token_loss
    # token_bow_losses
    # log_steps = [0] + [2**i for i in range(int(math.log2(512)) + 1)]
    linear_steps = [i for i in range(1000, 144000, 1000)]
    steps = linear_steps  # log_steps +

    per_token_means = []
    per_token_conf_intervals_left = []
    per_token_conf_intervals_right = []
    for step in steps:
        token_bow_losses = df[df["step"] == step]["token_bow_losses"].tolist()
        if len(token_bow_losses) == 0:
            print(step)
        vectors = [np.array(item) for item in token_bow_losses]
        mean_bow_loss, standard_errors, conf_intervals = mean_vector(vectors)
        per_token_means.append(mean_bow_loss.mean())
        per_token_conf_intervals_left.append(conf_intervals[0].mean(-1))
        per_token_conf_intervals_right.append(conf_intervals[1].mean(-1))

    df = pd.DataFrame(
        {
            "step": steps,
            "mean_aggregated_loss": per_token_means,
            "conf_left": per_token_conf_intervals_left,
            "conf_right": per_token_conf_intervals_right,
        }
    )

    fig = px.line(df, x="step", y="mean_aggregated_loss")
    fig.write_image(Path.cwd() / "images" / "image.png")


if __name__ == "__main__":
    main()
