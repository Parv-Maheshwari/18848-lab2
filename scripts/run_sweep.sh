#!/bin/bash
# Submit a 4-split scaling sweep (train + dependent eval job per split).
#
# Usage:
#   scripts/run_sweep.sh baseline
#   scripts/run_sweep.sh conv model/tokenizer=conv
#
# Splits are submitted largest first so that the longest runs start first.
set -u

name=$1
shift

for split in p100 p50 p20 p10; do
    uv run _slurm/submit.py "$name" "$split" \
        --env.gres=gpu:h100-80:2 \
        --args "$@" "+split=$split"
done
