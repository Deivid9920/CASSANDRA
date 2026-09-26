"""Cache contract: memmap float16 store + CSV index (id, model,
layer, mode, offset); idempotent (re-runs skip statements already
cached by their sha256); atomic writes. load_index(config, model)
and load_vectors(config, model, layer, mode) expose the cache to
probes and notebooks with per-layer lazy loading."""
