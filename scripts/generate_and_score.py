"""CLI: --config config.yaml --model NAME. Generates TriviaQA
answers, scores them with the best probe and writes the scored
table plus the abstention analysis."""

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
