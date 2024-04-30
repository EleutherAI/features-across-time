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
from scripts.plots.plot_ngram import write_garbage, base_2_log_ticks, hex_to_rgba

def plot(
    df: pd.DataFrame, image_name: str, debug: bool, model_series: str, qualitative=False
):
    print("plot")
    # df = df[df['step'] != 0] #  Step = 0 gives final step, which we collect elsewhere
    if not debug:
        write_garbage()

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
    fig.update_yaxes(range=[0, 0.00001])

    fig.write_image(str(image_name), format="pdf")

def main():
    debug = False
    os.makedirs(Path.cwd() / "images", exist_ok=True)

    model_series="Pythia"
    num_samples = 1024
    steps = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    ngram_path = "/mnt/ssd-1/lucia/es/es-bigrams.pkl"
    image_name = Path.cwd() / "images" / "learned-bigram-kl-div.pdf"
    model_metadata = [
        ("pythia-14m", "14M", 8),
        ("pythia-70m", "70M", 8),
        ("pythia-160m", "160M", 8),
        ("pythia-410m", "410M", 8),
        ("pythia-1b", "1B", 8),
        ("pythia-1.4b", "1.4B", 8),
        # ("pythia-2.8b", "2.8B", 8),
        # ("pythia-6.9b", "6.9B", 8),
        # ("pythia-12b", "12B", 8),
    ]
    
    with open(ngram_path, "rb") as f:
        bigram_counts = pickle.load(f).toarray().astype(np.float32)
        data_bigram_probs = torch.tensor(
            bigram_counts / (bigram_counts.sum(axis=1) + np.finfo(np.float32).eps)
        )

    print(data_bigram_probs[:10, :10])
    del data_bigram_probs, bigram_counts
    
    model_dfs = []
    for model_name, pretty_model_name, num_chunks in model_metadata:
        kl_divs = []
        for step in steps:
            print("loop")
            counts_path = Path.cwd() / "output" / f"finetune_bigram_{model_name}_{num_samples}_{step}.pkl"
            with open(counts_path, 'rb') as f:
                learned_bigram_probs = torch.tensor(pickle.load(f).toarray(), dtype=torch.float32)
                print(learned_bigram_probs[:10, :10])
                del learned_bigram_probs, learned_bigram_probs
    #         print("opened")
    #         kl_divs.append(kl_divergence(learned_bigram_probs.flatten(), data_bigram_probs.flatten()).item())

    #     model_df = pd.DataFrame(
    #         {
    #             "step": steps,
    #             "kl_div": kl_divs,
    #         }
    #     )
    #     model_df["model_name"] = model_name
    #     model_df["pretty_model_name"] = pretty_model_name
    #     model_dfs.append(model_df)
    # df = pd.concat(model_dfs)
    # df.to_csv("finetune_bigram_kl_divs.csv")

    df = pd.read_csv("finetune_bigram_kl_divs.csv")
    plot(df, image_name, debug, model_series)


if __name__ == "__main__":
    main()
