# features-across-time
Understanding how features learned by neural networks evolve throughout training

NOTE: Apologies for the sloppy code in many places. We plan to clean it up in the next few weeks.

### Reproduce Pythia Suite Experiments

Run the following Python scripts from the command line to collect and plot data over training steps:

```
python -m scripts.preprocess.build_ngram_data --n <n1> <n2>
python -m scripts.inference.ngrams --n <n1> <n2>
```

To plot existing data, use:

```
python -m scripts.plot.plot_single_rows
```
