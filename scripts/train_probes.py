"""CLI: --config config.yaml. Trains the full probe grid with
grouped CV and writes out-of-fold predictions."""

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
