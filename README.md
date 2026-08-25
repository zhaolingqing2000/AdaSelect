# AdaSelect reproducible discrete-event experiments

This project implements the supplied protocol without fitting to paper placeholders.

Run from this directory using Python 3.10+:

```text
pytest
python run_experiments.py --mode quick
python run_experiments.py --mode full
```

Quick output is isolated under `results/quick/`. Full output is written to
`results/raw/` and `results/summary/`; only full CSVs feed the generated paper files.

Baseline labels: CBAA is an adapted implementation following the decentralized
auction principle of [6]; LC-DTA is an adapted low-communication implementation
inspired by [4]; AS is an adapted feedback-driven selective implementation inspired
by [1]; CAA is an adapted capability-aware auction inspired by [2]. SS, DF, and
Static Full are the explicitly defined AdaSelect variants.

