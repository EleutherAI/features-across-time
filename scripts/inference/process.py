import pandas as pd
from pathlib import Path
import os

models = [
    # ("pythia-14m", 8),
    # ("pythia-70m", 4),
    # ("pythia-160m", 4),
    # ("pythia-410m", 4),
    # ("pythia-1b", 4),
    # ("pythia-1.4b", 4),
    # ("pythia-2.8b", 4),
    # ("pythia-6.9b", 1),
    ("pythia-12b", 1),
]
# ] + [(f"pythia-14m-seed{i}", 8) for i in range(1, 10)] \
#   + [(f"pythia-70m-seed{i}", 2) for i in range(1, 10)] \
#   + [(f"pythia-160m-seed{i}", 4) for i in range(1, 10)] \
#   + [(f"pythia-410m-seed{i}", 2) for i in range(1, 5)] \
#   + [(f"pythia-14m-warmup01", 8), (f"pythia-70m-warmup01", 4)]

for model, n in models:    
    df = pd.read_csv(Path.cwd() / 'output' / '24-06-05' / f'means_ngrams_model_{model}_1024.csv')

    # df column labels were     1-kl, 2-kl, 1-js, 2-js
    # but data append order was 1-kl, 1-js, 2-kl, 2-js
    # so we need to swap the columns
    # if 'seed' in model:
    #     # pass
    #     df.rename(columns={
    #         'mean_1-gram_logit_kl_div': 'mean_2-gram_logit_kl_div',
    #         'top_conf_1-gram_logit_kl_div': 'top_conf_2-gram_logit_kl_div',
    #         'bottom_conf_1-gram_logit_kl_div': 'bottom_conf_2-gram_logit_kl_div',

    #         'mean_1-gram_logit_js_div': 'mean_1-gram_logit_kl_div',
    #         'top_conf_1-gram_logit_js_div': 'top_conf_1-gram_logit_kl_div',
    #         'bottom_conf_1-gram_logit_js_div': 'bottom_conf_1-gram_logit_kl_div',

    #         'mean_2-gram_logit_kl_div': 'mean_1-gram_logit_js_div',
    #         'top_conf_2-gram_logit_kl_div': 'top_conf_1-gram_logit_js_div',
    #         'bottom_conf_2-gram_logit_kl_div': 'bottom_conf_1-gram_logit_js_div',
    #     }, inplace=True)
    # else:
    #     df.rename(columns={
    #         'mean_2-gram_logit_kl_div': 'mean_1-gram_logit_js_div',
    #         'top_conf_2-gram_logit_kl_div': 'top_conf_1-gram_logit_js_div',
    #         'bottom_conf_2-gram_logit_kl_div': 'bottom_conf_1-gram_logit_js_div',

    #         'mean_1-gram_logit_js_div': 'mean_2-gram_logit_kl_div',
    #         'top_conf_1-gram_logit_js_div': 'top_conf_2-gram_logit_kl_div',
    #         'bottom_conf_1-gram_logit_js_div': 'bottom_conf_2-gram_logit_kl_div',
    #     }, inplace=True)
    
    # Remove artifact from hacky df manipulation
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Add in data from rate limited checkpoints
    if os.path.exists(Path.cwd() / 'output' / f'means_ngrams_model_{model}_1024_steps.csv'):
        steps_df = pd.read_csv(Path.cwd() / 'output' / f'means_ngrams_model_{model}_1024_steps.csv')
        steps_df.index = steps_df['step']
        # identify the steps of the rows in df that contain nans and replace them with the row 
        # at the corresponding step in steps df
        common_cols = df.columns.intersection(steps_df.columns)
        nan_indices = df[df.isnull().any(axis=1)].index
        print("nan indices", nan_indices)
        for idx in nan_indices:
            step = df.at[idx, 'step']
            if step in steps_df.index:
                print(step, steps_df.loc[step].isnull().any(), common_cols)
            if step in steps_df.index and not steps_df.loc[step].isnull().any():
                df.loc[idx, common_cols] = steps_df.loc[step, common_cols].values

    print("done", df.head(20))
    df.to_csv(
          Path.cwd() / 
          'output' / 
          '24-06-05' / 
          f'means_ngrams_model_{model}_1024.csv', index=False
        )
