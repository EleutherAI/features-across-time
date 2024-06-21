from pathlib import Path
import os

import numpy as np
import pandas as pd

def main():
    for model_name in (
        [
            "pythia-14m",
            "pythia-70m",
            "pythia-160m",
            "pythia-410m",
            "pythia-1b",
            "pythia-1.4b",
            "pythia-2.8b",
            "pythia-6.9b",
            "pythia-12b",
            "pythia-14m-warmup01",
            "pythia-70m-warmup01",
        ]
        + [f"pythia-14m-seed{i}" for i in range(3, 10)]
        + [f"pythia-70m-seed{i}" for i in range(1, 10)]
        + [f"pythia-160m-seed{i}" for i in range(8, 10)]
        + [f"pythia-410m-seed{i}" for i in range(1, 5)]
    ):
        dfs = [
            pd.read_csv(Path.cwd() / 'output' / f"ngram_{model_name}_1024.csv"),
        ]
        additional_df_paths = [
            Path.cwd() / 'output' / 'dev' / f"ngram_{model_name}_1024.csv",
            Path.cwd() / 'output' / f"ngram_{model_name}_1024_[3]_actual_divs.csv",
            Path.cwd() / 'output' / f"ngram_{model_name}_1024_[4]_divs.csv",    
        ]
        for path in additional_df_paths:
            if os.path.exists(path):
                dfs.append(pd.read_csv(path))

        all_columns = set()
        for df in dfs:
            all_columns.update(df.columns)

        for df in dfs:
            for col in all_columns:
                if col not in df.columns:
                    df[col] = np.nan

        concatenated_df = pd.concat(dfs, ignore_index=True)
        merged_df = concatenated_df.groupby('step').agg(
            lambda x: x.dropna().iloc[0] if x.dropna().any() else np.nan
        ).reset_index()    
        merged_df.to_csv(Path.cwd() / 'output' / 'dev' / f"ngram_{model_name}_1024.csv", index=False)



if __name__ == "__main__":
    main()