from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plot_ngram import add_kl_data, base_2_log_ticks, hex_to_rgba, kaleido_workaround
from plotly.subplots import make_subplots


def plot_loss_and_divergence(df: pd.DataFrame, image_name: str, debug: bool):
    if not debug:
        kaleido_workaround()

    tick_values, tick_texts = base_2_log_ticks(df["step"], step=2)
    bpb_coefficient = 0.3650388

    div_metadata = [
        (
            "unigram_logit_kl_div",
            "D<sub>KL</sub>(unigram model || Pythia) across time",
            [0, 7],
            2,
            2,
        ),
        (
            "bigram_logit_kl_div",
            "D<sub>KL</sub>(bigram model || Pythia) across time",
            [0, 7],
            2,
            1,
        ),
    ]
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
    entropies = [2.89, 2.04]

    # log_spaced_indices = np.unique(np.logspace(0, np.log2(df['index'].max()), base=2, num=20).astype(int))
    # df = df[df['index'].isin(log_spaced_indices)]

    fig = make_subplots(
        rows=2,
        cols=2,
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=[
            "Unigram sequence loss across time",
            "Bigram sequence loss across time",
        ]
        + [label[1] for label in div_metadata],
        horizontal_spacing=0.02,
        vertical_spacing=0.1,
    )

    for idx, ngram in enumerate(["unigram", "bigram"]):
        df[f"mean_{ngram}_bpb"] = df[f"mean_{ngram}_loss"] * bpb_coefficient
        df[f"mean_{ngram}_bpb_bottom_conf"] = (
            df[f"bottom_conf_{ngram}_loss"] * bpb_coefficient
        )
        df[f"mean_{ngram}_bpb_top_conf"] = (
            df[f"top_conf_{ngram}_loss"] * bpb_coefficient
        )

        for i, model in enumerate(df["pretty_model_name"].unique()):
            df_model = df[df["pretty_model_name"] == model]
            color = px.colors.qualitative.Plotly[i]
            transparent_color = hex_to_rgba(color, opacity=0.17)

            fig.add_trace(
                go.Scatter(
                    x=df_model["step"],
                    y=df_model[f"mean_{ngram}_bpb_top_conf"],
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
                    x=df_model["step"],
                    y=df_model[f"mean_{ngram}_bpb_bottom_conf"],
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
                    x=df_model["step"],
                    y=df_model[f"mean_{ngram}_bpb"],
                    mode="lines+markers",
                    marker=dict(size=5, symbol=marker_series[i]),
                    name=model,
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
            x1=143000,
            y1=entropies[idx],
            line=dict(color="black", width=2, dash="dot"),
            row=1,
            col=idx + 1,
        )

    for label, pretty_label, y_range, row, col in div_metadata:
        for i, model in enumerate(df["pretty_model_name"].unique()):
            df_model = df[df["pretty_model_name"] == model]
            color = px.colors.qualitative.Plotly[i]
            transparent_color = hex_to_rgba(color, opacity=0.17)
            df_model[f"top_conf_{label}_bpb"] = (
                df_model[f"top_conf_{label}"] * bpb_coefficient
            )
            df_model[f"bottom_conf_{label}_bpb"] = (
                df_model[f"bottom_conf_{label}"] * bpb_coefficient
            )
            df_model[f"mean_{label}_bpb"] = df_model[f"mean_{label}"] * bpb_coefficient

            fig.add_trace(
                go.Scatter(
                    x=df_model["step"],
                    y=df_model[f"top_conf_{label}_bpb"],
                    fill=None,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=df_model["step"],
                    y=df_model[f"bottom_conf_{label}_bpb"],
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=transparent_color,
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=df_model["step"],
                    y=df_model[f"mean_{label}_bpb"],
                    mode="lines+markers",
                    marker=dict(size=5, symbol=marker_series[i]),
                    name=model,
                    line=dict(color=color),
                    showlegend=False,
                ),  # row==1 and col==2
                row=row,
                col=col,
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
            bgcolor="rgba(255, 255, 255, 0.85)",
        ),
        margin=dict(l=20, r=20, t=50, b=60),
    )

    fig.update_xaxes(
        title_text="", type="log", tickvals=tick_values, ticktext=tick_texts
    )

    fig.update_yaxes(
        title_text="Loss", title_font=dict(size=12), title_standoff=10, row=1, col=1
    )
    fig.update_yaxes(range=[1.8, 4.5], row=1, col=1)
    fig.update_yaxes(range=[0, 4.5], row=2, col=1)
    fig.update_yaxes(
        title_text="KL divergence",
        title_font=dict(size=12),
        title_standoff=10,
        row=2,
        col=1,
    )
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

    fig.write_image(image_name, format="pdf")


def plot_warmups(debug: bool):
    num_samples = 1024
    for model_size in [14, 70]:
        model_metadata = [
            (f"pythia-{model_size}m", f"{model_size}M (fast warmup)"),
            (f"pythia-{model_size}m-warmup01", f"{model_size}M (slow warmup)"),
        ]
        model_dfs = []
        for model_name, pretty_model_name in model_metadata:
            model_df = pd.read_csv(
                Path.cwd()
                / "output"
                / f"means_ngrams_model_{model_name}_{num_samples}.csv"
            )
            supplementary_kl_div_path = (
                Path.cwd()
                / "output"
                / f"means_ngrams_model_{model_name}_{num_samples}_kl_div.csv"
            )
            model_df = add_kl_data(model_df, supplementary_kl_div_path)

            model_df["step"] = model_df["step"] + 1
            model_df["model_name"] = model_name
            model_df["pretty_model_name"] = pretty_model_name

            model_dfs.append(model_df)
        df = pd.concat(model_dfs)

        image_name = Path.cwd() / "images" / f"warmups-{model_size}m.pdf"
        plot_loss_and_divergence(df, image_name, debug)


def main():
    debug = False

    plot_warmups(debug)


if __name__ == "__main__":
    main()
