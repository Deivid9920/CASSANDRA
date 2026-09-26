"""Evaluation contract: aggregate out-of-fold predictions into the
table (model, layer, mode, type, AUC with bootstrap CI, AP, ECE,
Brier, paired p vs |logprob| baseline, Holm-adjusted
significance) written to logs/probe_results.jsonl and
docs/probe_table.md; AUC above 0.95 triggers the documented
leakage audit; leave-one-domain-out is the secondary analysis.
probe_summary_table() and load_oof() serve notebooks."""
