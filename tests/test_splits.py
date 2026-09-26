"""Split contract tests: entity-grouped folds with no leakage.

Filled in Fase 1. The contract under test: StratifiedGroupKFold with
groups = entity and stratification = (domain, label); a statement never
appears in two folds; each fold keeps both classes of every domain with
at least two entities; rebuilding with the same seed reproduces the
fold assignment deterministically.
"""

from __future__ import annotations
