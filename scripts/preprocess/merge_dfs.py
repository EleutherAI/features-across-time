from pathlib import Path
from argparse import ArgumentParser
import time

import numpy as np
import pandas as pd


def main(backup_dir: Path, data_path: Path, num_samples: int):
    if not backup_dir.exists():
        backup_dir.mkdir(parents=True)

    model_metadata = (
        [
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
            ("pythia-70m-warmup01", "70M"),
        ]
        + [(f"pythia-14m-seed{i}", "14M") for i in range(3, 10)]
        + [(f"pythia-70m-seed{i}", "70M") for i in range(1, 10)]
        + [(f"pythia-160m-seed{i}", "160M") for i in range(8, 10)]
        + [(f"pythia-410m-seed{i}", "410M") for i in range(1, 5)]
    )
    
    for model_name, pretty_model_name in model_metadata:
        base_df_path = data_path / f"ngram_{model_name}_{num_samples}.csv"
        if base_df_path.exists():
            base_df = pd.read_csv(base_df_path)
            base_df.to_csv(backup_dir / f'{time.time()}_{base_df_path.name}', index=False)
        
        df_paths = [
            Path.cwd() / "output" / f"ngram_{model_name}_1024.csv",
            base_df_path,
        ]
        dfs = [pd.read_csv(path) for path in df_paths if path.exists()]
        if not dfs:
            continue

        all_columns = set()
        for df in dfs:
            all_columns.update(df.columns)
            df["model_name"] = model_name
            df["pretty_model_name"] = pretty_model_name

        for df in dfs:
            for col in all_columns:
                if col not in df.columns:
                    df[col] = np.nan

        concatenated_df = pd.concat(dfs, ignore_index=True)
        # Each groupby df contains all data for a step, including nans
        merged_df = (
            concatenated_df.groupby("step")
            .agg(lambda x: x.dropna().iloc[0] if x.dropna().any() else np.nan)
            .reset_index()
        )
        
        merged_df.to_csv(base_df_path, index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--backup-dir",
        "-b",
        default="output/backup",
        help="Directory to store the original versions of files modified by this script.",
    )
    parser.add_argument("--data_path", type=str, default="output")
    parser.add_argument("--num_samples", type=int, default=1024)
    args = parser.parse_args()

    main(
        Path(args.backup_dir), 
        Path(args.data_path),
        args.num_samples
    )
 