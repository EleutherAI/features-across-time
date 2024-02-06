# features-across-time
Understanding how features learned by neural networks evolve throughout training

NOTE: Apologies for the sloppy code in many places. We plan to clean it up in the next few weeks.

### Reproduce Pythia 410M Experiments

Run the following Python scripts from the command line to collect and plot data over training steps:

```
python scripts/divergences.py
python scripts/ngram_samples.py
python scripts/shuffled_samples.py
```

To plot existing data, use:

```
python scripts/plot_steps.py [--divergences] [--ngram_samples] [--shuffled_samples]
```
