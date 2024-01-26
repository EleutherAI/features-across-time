import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def main():
    num_samples = 1024
    os.makedirs(Path.cwd() / "images", exist_ok=True)

    
    df = pd.read_csv(
        Path.cwd() / "output" / f"means_ngrams_model_{'mistral-7b'}_{num_samples}.csv"
    )
    filtered_df = df[df['index'] <= 10]

    bpb_coefficient = 0.3366084909549386 / math.log(2)
    labels = ["random", "unigram", "bigram"]
    for label in labels:
        df[f'{label}_grad'] = np.gradient(df[f'mean_{label}_loss'])
        df[f'mean_{label}_bpb'] = df[f'mean_{label}_loss'] * bpb_coefficient

        trace1 = go.Scatter(x=df[f'index'], y=df[f'mean_{label}_bpb'], mode='lines', name='Mean BPB')
        trace2 = go.Scatter(x=df[f'index'], y=df[f'{label}_grad'], mode='lines', name='Gradient')
        layout = go.Layout(title=f'Mistral {label} sequence BPB with gradient', xaxis_title='Token Index', yaxis_title='Value')
        fig = go.Figure(data=[trace1, trace2], layout=layout)

        fig.write_image(Path.cwd() / 'images' / f'{label}_grad.png')

        # Loss at 500th token vs. 50th
        ratio = df.at[49, f'mean_{label}_loss'] / df.at[499, f'mean_{label}_loss']
        print(f'50th index loss : 500th index loss on {label} sequence: {ratio}')
        
        
        filtered_df[f'{label}_grad'] = np.gradient(filtered_df[f'mean_{label}_loss'])
        filtered_df[f'mean_{label}_bpb'] = filtered_df[f'mean_{label}_loss'] * bpb_coefficient

        trace1 = go.Scatter(x=filtered_df[f'index'], y=filtered_df[f'mean_{label}_bpb'], mode='lines', name='Mean BPB')
        trace2 = go.Scatter(x=filtered_df[f'index'], y=filtered_df[f'{label}_grad'], mode='lines', name='Gradient')
        layout = go.Layout(title=f'Mistral {label} sequence BPB with gradient', xaxis_title='Token Index', yaxis_title='Value')
        fig = go.Figure(data=[trace1, trace2], layout=layout)

        fig.write_image(Path.cwd() / 'images' / f'{label}_grad_zoomed.png')


if __name__ == "__main__":
    main()
