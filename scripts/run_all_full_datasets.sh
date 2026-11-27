#!/usr/bin/env bash
# General-ML benchmarks only. No NSL-KDD and no per-origin plot explosion.
set -euo pipefail
python -m mmcyber.cli preflight
for name in folktables_income folktables_employment adult bank_marketing compas german_credit taiwan_credit breast_cancer; do
  python -m mmcyber.cli run-all --config "configs/${name}_full.yaml" \
    --rashomon-tolerance 0.015 \
    --max-background 128 --max-explain 256 \
    --top-k-fraction 0.10 --plot-profile none \
    --aggregate-out-dir "runs/aggregate_${name}"
done
# Do not automatically run LIME on thousands of cases: run multi-explain
# on one pilot origin with a small --max-samples budget first.
