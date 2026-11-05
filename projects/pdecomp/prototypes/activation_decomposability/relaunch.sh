#!/bin/bash
# Second wave after the first wave was killed externally: only arms missing from results/*.json.
cd "$(dirname "$0")"
PY=/home/user/lab/venvs/research/bin/python
run() { OMP_NUM_THREADS=1 $PY run.py --target abs --seed $1 --coeffs $2 --acts $3 --train_steps 4000 --spd_steps 1500 --out results/abs_s$1_c$2_w2.json > results/abs_s$1_c$2_w2.log 2>&1 & }
run 0 1e-3 bilinear; run 0 3e-3 bilinear; run 1 1e-3 bilinear
run 1 3e-3 relu,gelu,silu,bilinear; run 2 1e-3 relu,gelu,silu,bilinear; run 2 3e-3 relu,gelu,silu,bilinear
wait
