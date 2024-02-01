import math
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plot_ngram import base_2_log_ticks, hex_to_rgba
from plotly.subplots import make_subplots
import plotly.express as px

def plot_step_index_lines(df: pd.DataFrame, label: str, log=True):
    fig = go.Figure()

    # Assuming 'step' is the column to distinguish different lines
    for step, group in df.groupby('step'):
        fig.add_trace(go.Scatter(x=group['index'], y=group[f'mean_{label}_bpb'], mode='lines+markers',
                                 name=f'Step {step}'))
    fig.update_layout(
        title=f'BPB over indices for models at different steps',
        xaxis_title='Index',
        yaxis_title=label  # Using z_label directly here for the y-axis title
    )
    fig.update_xaxes(type="log" if log else "linear")

    return fig


def main(debug: bool):
    image_name = Path.cwd() / 'images' / f'in-context.pdf'
    if not debug:
        # Garbage data to work around Kaleido bug: https://github.com/plotly/plotly.py/issues/3469
        fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
        fig.write_image(image_name, format="pdf")
        time.sleep(2)

    num_samples = 1024

    os.makedirs(Path.cwd() / "images", exist_ok=True)

    bpb_coefficient = 0.3650388
    marker_series = [
        "circle", "square", "diamond", "cross", "x", 
        "triangle-up", "triangle-down", "triangle-left", "triangle-right", 
        "pentagon", "hexagon", "octagon", "star", "hexagram"
    ]

    df = pd.read_csv(
        Path.cwd() / "output" / f"pythia-12b_{num_samples}_steps.csv"
    )
    tick_values, tick_texts = base_2_log_ticks(df["index"])

    # Add missing step column
    n = 2048
    # steps = ["000"] + [f"{2**i:03}" for i in range(int(math.log2(359)) + 1)] + ['359']
    steps = [16, 256, 1000, 143_000]
    group_numbers = df['index'].eq(0).cumsum() - 1
    df['step'] = group_numbers.apply(lambda x: steps[x] if x < len(steps) else steps[-1])
    
    # Make index 1-indexed
    df['index'] = df['index'] + 1

    # Filter to log spaced samples
    max_index = df['index'].max()
    log_spaced_indices = np.unique(np.logspace(0, np.log2(max_index), num=int(np.log2(max_index)) + 1, base=2).astype(int))
    df = df[df['index'].isin(log_spaced_indices)]

    # Order data
    step_order = sorted(df['step'].unique())

    fig = make_subplots(
        rows=1, cols=2, shared_xaxes=True, shared_yaxes=True, 
        subplot_titles=["Unigram loss over sequence positions", "Bigram loss over sequence positions"],
        horizontal_spacing=0.02,
        vertical_spacing=0.05)

    for idx, ngram in enumerate(["unigram", "bigram"]):
        df[f'mean_{ngram}_bpb'] = df[f'mean_{ngram}_loss'] * bpb_coefficient
        df[f"bottom_conf_{ngram}_bpb"] = df[f"bottom_conf_{ngram}_loss"] * bpb_coefficient
        df[f"top_conf_{ngram}_bpb"] = df[f"top_conf_{ngram}_loss"] * bpb_coefficient

        # For some reason the legend items are listed in reverse order
        for i, step in enumerate(reversed(step_order)):
            group_df = df[df['step'] == step]
            color = px.colors.sequential.Plasma_r[i + 1]
            transparent_color = hex_to_rgba(color, opacity=0.17)

            fig.add_trace(go.Scatter(x=group_df['index'], y=group_df[f'top_conf_{ngram}_bpb'], fill=None, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1, col=idx + 1)
            fig.add_trace(go.Scatter(x=group_df['index'], y=group_df[f'bottom_conf_{ngram}_bpb'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor=transparent_color, showlegend=False, hoverinfo='skip'), row=1, col=idx + 1)
            fig.add_trace(go.Scatter(x=group_df['index'], y=group_df[f'mean_{ngram}_bpb'], mode='lines+markers',  marker=dict(size=5, symbol=marker_series[i]), name=f'{step:,}', line=dict(color=color), showlegend=idx==1), row=1, col=idx + 1)

    fig.update_layout(
        width=1000, 
        height=400, 
        legend=dict(
            x=1.02,
            y=0.5,
            xanchor='left', 
            yanchor='middle',
            font=dict(size=8),
            title="Step",
        ),
        margin=dict(l=20, r=20, t=50, b=60)
    )

    fig.update_xaxes(title_text="", type="log", tickvals=tick_values, ticktext=tick_texts)
    fig.update_yaxes(title_text="Loss", title_font=dict(size=12), title_standoff=10, row=1, col=1)
    fig.add_annotation(
        dict(
            text="Token index",
            xref="paper", yref="paper",
            x=0.5, y=-0.2,
            showarrow=False,
            font=dict(size=14)
        )
    )
    fig.write_image(image_name, format="pdf")    

    # # Plot
    # labels = ["unigram", "bigram"] # "random", 
    # filtered_df = df[df['index'] <= 10]

    # for label in labels:
    #     # Plot Anthropic's in-context learning metric
    #     ratio_df = df.groupby('step').apply(lambda x: x[x['index'] == 49][f'mean_{label}_loss'].values[0] /
    #                                         x[x['index'] == 499][f'mean_{label}_loss'].values[0] if len(x[x['index'] == 49]) > 0 and len(x[x['index'] == 499]) > 0 else np.nan)
    #     fig = go.Figure(
    #         [go.Scatter(x=ratio_df.index, y=ratio_df, mode='lines+markers')],
    #         layout=dict(
    #             title=f'Loss Ratio at 50th vs. 500th Token Index for {label}',
    #             xaxis_title='Step',
    #             yaxis_title='Loss Ratio (50th Index / 500th Index)'
    #         )
    #     )
    #     fig.update_xaxes(type="log")

    #     fig.write_image(Path.cwd() / 'images' / f'{label}_loss_ratio.png')

    #     # Plot BPB over first few token indices
    #     filtered_df[f'mean_{label}_bpb'] = filtered_df[f'mean_{label}_loss'] * bpb_coefficient
    #     filtered_df[f'{label}_grad'] = filtered_df.groupby('step')[f'mean_{label}_bpb'].transform(lambda x: np.gradient(x, x.index))
    #     fig = plot_step_index_lines(filtered_df, f'mean_{label}_bpb', log=False)
    #     fig.write_image(Path.cwd() / 'images' / f'{label}_bpb_first_indices.png')    

    #     # Plot BPB over token indices
    #     df[f'mean_{label}_bpb'] = df[f'mean_{label}_loss'] * bpb_coefficient
    #     df[f'{label}_grad'] = df.groupby('step')[f'mean_{label}_bpb'].transform(lambda x: np.gradient(x, x.index))
    #     fig = plot_step_index_lines(df, label)
    #     fig.write_image(Path.cwd() / 'images' / f'{label}_bpb_indices.png')    

        


if __name__ == "__main__":
    main(debug=False)
