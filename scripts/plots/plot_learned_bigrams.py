import os
from pathlib import Path
import pickle

import torch
from torch import tensor
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scripts.script_utils.divergences import kl_divergence
from scripts.plots.plot_ngram import kaleido_workaround, base_2_log_ticks, hex_to_rgba

def plot(
    df: pd.DataFrame, image_name: str, debug: bool, model_series: str, qualitative=False
):
    print("plot")
    # df = df[df['step'] != 0] #  Step = 0 gives final step, which we collect elsewhere
    if not debug:
        kaleido_workaround()

    bpb_coefficient = 0.3650388

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
    entropies = [2.89, 2.04, 2]

    # log_spaced_indices = np.unique(np.logspace(0, np.log2(df['index'].max()), base=2, num=20).astype(int))
    # df = df[df['index'].isin(log_spaced_indices)]

    fig = go.Figure()

    # fig = make_subplots(
    #     rows=1,
    #     cols=1,
    #     subplot_titles=["KL divergence(learned bigrams | dataset bigrams)"],
    #     horizontal_spacing=0.02,
    #     vertical_spacing=0.1,
    # )

    for i, model in enumerate(df["pretty_model_name"].unique()):
        df_model = df[df["pretty_model_name"] == model]
        color = px.colors.qualitative.Plotly[i] if qualitative else px.colors.sequential.Plasma_r[i]

        fig.add_trace(
            go.Scatter(
                x=df_model["step"],
                y=df_model["kl_div"],
                mode="lines+markers",
                marker=dict(size=5, symbol=marker_series[i]),
                name=model,
                line=dict(color=color)
            )
        )

    fig.update_layout(
        width=1000,
        height=600,
        legend=dict(
            x=0.98,
            y=0.98,
            xanchor="right",
            yanchor="top",
            font=dict(size=8),
            bgcolor="rgba(255, 255, 255, 0.85)"
        ),
        margin=dict(l=20, r=20, t=50, b=60),
        xaxis_title="Training step",
        yaxis_title="KL Divergence",
        title="KL Divergence(Learned | Dataset Bigrams) over Training Steps"
    )
    tick_values, tick_texts = base_2_log_ticks(df["step"], step=2)
    fig.update_xaxes(
        type="log", tickvals=tick_values, ticktext=tick_texts
    )
    fig.update_yaxes(range=[0, 0.00001])

    fig.write_image(str(image_name), format="pdf")

def main():
    debug = False
    os.makedirs(Path.cwd() / "images", exist_ok=True)

    model_series="Pythia"
    num_samples = 8192 # 1024
    steps = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    ngram_path = "/mnt/ssd-1/lucia/es/es-bigrams.pkl"
    image_name = Path.cwd() / "images" / "learned-bigram-kl-div.pdf"
    model_metadata = [
        ("pythia-14m", "14M", 8),
        # ("pythia-70m", "70M", 8),
        # ("pythia-160m", "160M", 8),
        # ("pythia-410m", "410M", 8),
        # ("pythia-1b", "1B", 8),
        # ("pythia-1.4b", "1.4B", 8),
        # ("pythia-2.8b", "2.8B", 8),
        # ("pythia-6.9b", "6.9B", 8),
        # ("pythia-12b", "12B", 8),
    ]
    
    with open(ngram_path, "rb") as f:
        bigram_counts = pickle.load(f).toarray().astype(np.float32)
        non_zero = bigram_counts.sum(axis=1) > 0
        print(non_zero.sum())
        bigram_counts[~non_zero] = 1 / 50_277
        flattened_data_bigram_probs = torch.tensor(
            bigram_counts / (bigram_counts.sum(axis=1) + np.finfo(np.float32).eps)
        ).flatten()
        non_zero = np.tile(non_zero[:, None], (1, bigram_counts.shape[1])).flatten()

    del bigram_counts
    
    model_dfs = []
    for model_name, pretty_model_name, num_chunks in model_metadata:
        kl_divs = []
        filtered_kl_divs = []
        for step in steps:
            print("loop")
            counts_path = Path('/') / "mnt" / "ssd-1" / "lucia" / "finetune" / f"2-finetune_bigram_{model_name}_{num_samples}_{step}.pkl"
            with open(counts_path, 'rb') as f:
                learned_bigram_probs = torch.tensor(pickle.load(f).toarray(), dtype=torch.float32)

                learned_bigram_probs[learned_bigram_probs.sum(axis=1) == 0] = 1 / 50_277
                assert np.all(
                    np.isclose(learned_bigram_probs.sum(axis=1), 1) | np.isclose(learned_bigram_probs.sum(axis=1), 0))
                learned_bigram_probs = learned_bigram_probs.flatten()
            print("opened")
            kl_divs.append(kl_divergence(learned_bigram_probs, flattened_data_bigram_probs).item())
            filtered_kl_divs.append(kl_divergence(learned_bigram_probs[non_zero], flattened_data_bigram_probs[non_zero]).item())
            print(filtered_kl_divs[-1])

        model_df = pd.DataFrame(
            {
                "step": steps,
                "kl_div": kl_divs,
                "filtered_kl_div": filtered_kl_divs
            }
        )
        model_df["model_name"] = model_name
        model_df["pretty_model_name"] = pretty_model_name
        model_dfs.append(model_df)
    df = pd.concat(model_dfs)
    df.to_csv(Path('/') / "mnt" / "ssd-1" / "lucia" / "finetune" / "2-finetune_bigram_kl_divs.csv")

    df = pd.read_csv(Path('/') / "mnt" / "ssd-1" / "lucia" / "finetune" / "2-finetune_bigram_kl_divs.csv")
    plot(df, image_name, debug, model_series)


if __name__ == "__main__":
    main()
