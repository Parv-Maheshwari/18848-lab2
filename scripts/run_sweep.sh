#!/bin/bash
# Submit a 4-split scaling sweep (train + dependent eval job per split).
#
# Usage:
#   scripts/run_sweep.sh baseline
#   scripts/run_sweep.sh conv model/tokenizer=conv
#
# Extra `submit.py` options (e.g., a different cluster environment or results
# directory) can be passed via `SUBMIT_ARGS`, and the GPU request via `GRES`:
#   GRES=gpu:h100:2 SUBMIT_ARGS="--env.name psc-robo --env.account cis220039p \
#       --path.results results_robo" scripts/run_sweep.sh baseline
#
# Splits are submitted largest first so that the longest runs start first.
set -u

name=$1
shift

for split in p100 p50 p20 p10; do
    # shellcheck disable=SC2086
    uv run _slurm/submit.py "$name" "$split" \
        --env.gres="${GRES:-gpu:h100-80:2}" ${SUBMIT_ARGS:-} \
        --args "$@" "+split=$split"
done
