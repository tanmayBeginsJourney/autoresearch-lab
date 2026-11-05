#!/bin/bash
# 12 single-thread jobs, 4 at a time: abs x 3 seeds x 3 coeff_imp, sq x 3 seeds x coeff 3e-3.
cd "$(dirname "$0")"
PY=/home/user/lab/venvs/research/bin/python
jobs=()
for s in 0 1 2; do for c in 1e-3 3e-3 1e-2; do jobs+=("abs $s $c"); done; done
for s in 0 1 2; do jobs+=("sq $s 3e-3"); done
printf '%s\n' "${jobs[@]}" | xargs -P 4 -L 1 bash -c 'OMP_NUM_THREADS=1 '$PY' run.py --target $0 --seed $1 --coeffs $2 --train_steps 4000 --spd_steps 1500 --out results/$0_s$1_c$2.json > results/$0_s$1_c$2.log 2>&1'
