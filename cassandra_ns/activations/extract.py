"""Extraction contract: TransformerLens hooks on
blocks.{{L}}.hook_resid_post for the configured layers of each
model; capture the last-token vector (real lengths tracked, no
padding tokens) and the mean-pool over the statement; batch 32,
fp32 on CPU / fp16 on CUDA. A statement alone must produce the
same activation as inside a batch."""
