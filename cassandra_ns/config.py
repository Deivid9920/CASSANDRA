"""load_config(path) -> dict: read config.yaml, validate every
section key-by-key (seed, models, data, splits, probes, stats,
generation, abstention, ensemble, colab) and return the dict.
Invalid or missing keys raise ValueError with the offending key."""
