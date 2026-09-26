"""CLI: --config config.yaml. Executes the four notebooks in order
with nbconvert --execute --inplace; any failing cell fails the
run."""

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
