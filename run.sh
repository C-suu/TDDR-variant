#!/usr/bin/env bash
set -e
cd -- "$(dirname -- "$0")"

PYTHON="${PYTHON:-python}"
SEEDS="${SEEDS:-1 2 3 4 5}"

# Columns: algorithm, Ant-v2, HalfCheetah-v2, Hopper-v2, Walker2d-v2.
# InvertedDoublePendulum-v2 uses upsilon=1 for all methods.
while read -r algorithm ant halfcheetah hopper walker; do
  for setting in "Ant-v2:$ant" "HalfCheetah-v2:$halfcheetah" \
                 "Hopper-v2:$hopper" "Walker2d-v2:$walker" \
                 "InvertedDoublePendulum-v2:1.0"; do
    env="${setting%:*}"
    upsilon="${setting#*:}"
    for seed in $SEEDS; do
      if [[ "${DRY_RUN:-0}" == 1 ]]; then
        echo "python main.py --policy $algorithm --env $env --seed $seed --qweight $upsilon"
      else
        "$PYTHON" main.py --policy "$algorithm" --env "$env" \
          --seed "$seed" --qweight "$upsilon" \
          --max_timesteps 1000000 --start_timesteps 10000 \
          --eval_freq 5000 --batch_size 128
      fi
    done
  done
done <<'PARAMETERS'
DADC    0.3  0.0  0.9  0.4
DASC    0.5  0.0  0.9  0.8
SASC    0.0  0.2  0.8  0.0
DADC-R  0.6  0.3  0.6  0.2
DASC-R  0.9  0.1  0.4  0.4
SASC-R  0.8  0.8  0.8  0.8
PARAMETERS
