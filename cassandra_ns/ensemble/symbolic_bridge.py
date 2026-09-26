"""Ensemble contract: when config.ensemble.prometheus_graph points
to a consistency graph, extract subject-predicate-object triplets
from generations, map them to a symbolic signal {{0, 0.5, 1}} by
consistency, and fit a 2-feature logistic ensemble (probe score,
symbolic signal) inside validation folds only; report probe-alone
vs signal-alone vs ensemble on the same extractable subset with
paired bootstrap. Without a graph the ensemble is marked
unavailable with its cause. load_ensemble_results() serves
notebooks."""
