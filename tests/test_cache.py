"""Activation cache contract tests: round-trip fidelity and idempotency.

Filled in Fase 2. The contract under test: an activation reconstructed
from the fp16 memmap cache matches the direct forward pass within the
fp16 storage tolerance; a statement processed alone produces the same
activation as inside a batch (fp32 compute, no padding leakage); and a
second extraction run skips everything already cached (identical index
hash), i.e. idempotency is real, not nominal.
"""

from __future__ import annotations
