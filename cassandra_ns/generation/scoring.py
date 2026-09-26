"""Scoring contract: first declarative sentence of each generation
(flagged when absent, with a declared policy), probe score from
the best Phase 3 configuration, ground truth by normalized
string-match against aliases; ambiguous matches are excluded and
counted, never forced. abstention_curve() sweeps the threshold
tau and computes the coverage-accuracy curve plus tau* at
min_coverage; load_scored() serves notebooks."""
