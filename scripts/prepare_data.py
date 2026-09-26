"""CLI: --config config.yaml. Downloads/verifies the true/false
domain files, builds entities, folds.json and the EDA summary."""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.parse_args()
    raise SystemExit("not implemented yet: see the module "
                     "docstring for the API contract")


if __name__ == "__main__":
    main()
