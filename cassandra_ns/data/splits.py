"""Split contract: StratifiedGroupKFold with groups = entity and
stratification = (domain, label); n_folds fixed folds written to
data/processed/folds.json with its sha256. A statement never
appears in two folds; rebuilding with the same seed reproduces
the file byte for byte."""
