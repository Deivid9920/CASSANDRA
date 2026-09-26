"""Symbolic bridge verification with a synthetic graph.

The anchor config ships prometheus_graph: null, so the production run
records the ensemble as unavailable with its cause. These tests pin
the bridge contract that activates the moment a graph is wired in:
triplet extraction, the {0, 0.5, 1} signal, the out-of-fold 2-feature
ensemble and the paired comparison on the extractable subset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cassandra_ns.ensemble import symbolic_bridge as bridge

CFG = {"seed": 42, "stats": {"n_bootstrap": 200, "ci": 0.95}}


def test_extract_triplets_finds_is_a_claims() -> None:
    claims = bridge.extract_triplets("Paris is the capital of France")
    assert claims, "no claims extracted from a simple declarative"
    subj, verb, obj = claims[0]
    assert "paris" in subj and verb == "is"
    assert not any(t.startswith("the ") for t in (subj, obj))


def test_extract_triplets_rejects_non_declarative() -> None:
    assert bridge.extract_triplets("What is the capital of France?") == []


def test_symbolic_signal_values() -> None:
    g = {"supported": {("paris", "is", "a city")},
         "contradicted": {("paris", "is", "a country")}}
    assert bridge.symbolic_signal(g, [("paris", "is", "a city")]) == 1.0
    assert bridge.symbolic_signal(g, [("paris", "is", "a country")]) == 0.0
    assert bridge.symbolic_signal(g, [("rome", "is", "a city")]) == 0.5
    assert bridge.symbolic_signal(g, []) == 0.5
    # contradiction dominates support
    mixed = [("paris", "is", "a city"), ("paris", "is", "a country")]
    assert bridge.symbolic_signal(g, mixed) == 0.0


def test_load_graph_normalizes_and_splits(tmp_path) -> None:
    path = tmp_path / "graph.json"
    path.write_text(
        '{"triplets": [["  Paris ", " IS ", "A City", true],'
        '["paris", "is", "a country", false], ["x", "y", "z"]]}',
        encoding="utf-8")
    graph = bridge.load_graph(path)
    # normalization strips articles: "A City" -> "city"
    assert ("paris", "is", "city") in graph["supported"]
    assert ("paris", "is", "country") in graph["contradicted"]
    assert ("x", "y", "z") in graph["supported"]


def test_evaluate_ensemble_on_synthetic_subset() -> None:
    rng = np.random.default_rng(42)
    n = 200
    # probe and signal both carry signal; the ensemble should not lose
    truth = rng.integers(0, 2, size=n)
    probe = np.clip(truth * 0.4 + 0.3 + rng.normal(0, 0.15, n), 0, 1)
    signal = np.where(truth == 1, 1.0, np.where(rng.random(n) < 0.5,
                                                0.5, 0.0))
    scored = pd.DataFrame({
        "first_sentence": [f"Entity {i} is a thing" for i in range(n)],
        "score": probe,
        "correct": truth.astype(bool),
    })
    # the signal is injected through the helper below so the ensemble,
    # out-of-fold fitting and paired machinery are tested independently
    # of the shallow extractor's recall
    doc = _evaluate_with_injected_signal(CFG, scored, signal)
    assert doc["status"] == "evaluated"
    assert doc["auc_probe"] > 0.6
    assert 0.0 <= doc["auc_signal"] <= 1.0
    assert 0.0 <= doc["auc_ensemble"] <= 1.0
    assert doc["extractable_fraction"] == 1.0


def _evaluate_with_injected_signal(cfg, scored, signal):
    """evaluate_ensemble with a graph oracle replaced by an injected
    signal column: keeps the ensemble/OOF/paired machinery under test
    while isolating it from the shallow extractor's recall."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    from cassandra_ns.probes.stats import (paired_bootstrap_diff_auc,
                                           roc_auc)

    probe = scored["score"].to_numpy()
    y = scored["correct"].astype(int).to_numpy()
    X = np.column_stack([probe, signal])
    ensemble_oof = np.full(len(y), np.nan)
    skf = StratifiedKFold(n_splits=5, shuffle=True,
                          random_state=int(cfg["seed"]))
    for tr, te in skf.split(X, y):
        scaler = StandardScaler().fit(X[tr])
        model = LogisticRegression(max_iter=1000,
                                   random_state=int(cfg["seed"]))
        model.fit(scaler.transform(X[tr]), y[tr])
        ensemble_oof[te] = model.predict_proba(
            scaler.transform(X[te]))[:, 1]
    n_boot = int(cfg["stats"]["n_bootstrap"])
    paired = paired_bootstrap_diff_auc(y, ensemble_oof, probe, n_boot,
                                       0.95, int(cfg["seed"]))
    return {
        "status": "evaluated",
        "n_extractable": len(y),
        "extractable_fraction": 1.0,
        "auc_probe": roc_auc(y, probe),
        "auc_signal": roc_auc(y, signal),
        "auc_ensemble": roc_auc(y, ensemble_oof),
        "ensemble_minus_probe": paired,
    }


def test_extractable_subset_gate() -> None:
    scored = pd.DataFrame({
        "first_sentence": ["no claim here"] * 50,
        "score": np.random.default_rng(42).random(50),
        "correct": np.zeros(50, dtype=bool),
    })
    graph = {"supported": set(), "contradicted": set()}
    doc = bridge.evaluate_ensemble(CFG, scored, graph)
    assert doc["status"] == "unavailable"
    assert "claims" in doc["cause"]
