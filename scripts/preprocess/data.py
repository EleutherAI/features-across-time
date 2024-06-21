from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import numpy as np


def main(
    data_path: Path,
    num_samples: int
):
    model_metadata = [
        ("pythia-14m", "14M"),
        ("pythia-70m", "70M"),
        ("pythia-160m", "160M"),
        ("pythia-410m", "410M"),
        ("pythia-1b", "1B"),
        ("pythia-1.4b", "1.4B"),
        ("pythia-2.8b", "2.8B"),
        ("pythia-6.9b", "6.9B"),
        ("pythia-12b", "12B"),
        ("pythia-14m-warmup01", "14M"),
        ("pythia-70m-warmup01", "70M")
    ] + [
        (f"pythia-14m-seed{i}", "14M") for i in range(3, 10)
    ] + [
        (f"pythia-70m-seed{i}", "70M") for i in range(1, 10)
    ] + [
        (f"pythia-160m-seed{i}", "160M") for i in range(8, 10)
    ] + [
        (f"pythia-410m-seed{i}", "410M") for i in range(1, 5)
    ]
    dfs = []
    for model_name, pretty_model_name in model_metadata:
        dfs = [
            pd.read_csv(data_path / '24-06-14' / f"ngram_{model_name}_{num_samples}.csv"),
            pd.read_csv(data_path / '24-06-14' / f"ngram_{model_name}_{num_samples}_[3].csv"),
        ]

        all_columns = set()
        for df in dfs:
            df["model_name"] = model_name
            df["pretty_model_name"] = pretty_model_name

            all_columns.update(df.columns)

        for df in dfs:
            for col in all_columns:
                if col not in df.columns:
                    df[col] = np.nan

        concatenated_df = pd.concat(dfs, ignore_index=True)
        # breakpoint()
        # df containing all the data for a step, where each row has all the columns, some nan and others not
        merged_df = concatenated_df.groupby('step').agg(
            lambda x: x.dropna().iloc[0] if x.dropna().any() else np.nan
        ).reset_index()
        merged_df.to_csv(data_path / f"ngram_{model_name}_{num_samples}.csv")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default="output")
    parser.add_argument("--num_samples", type=int, default=1024)
    args = parser.parse_args()

    main(
        Path(args.data_path),
        args.num_samples,
    )