#!/bin/bash
# Usage: run.sh SEED WARMUP STEPS BATCH [extra args]  -> $LAB_ROOT/tmp/B1/runs/tms52_s{SEED}_w{WARMUP}_{STEPS}.json
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LAB_TMP=/root/lab/tmp/B1
export PYTHONPATH=/home/user/autoresearch-lab/projects/pdecomp/src
S=$1; W=$2; N=$3; B=$4; shift 4
OUT=/root/lab/tmp/B1/runs/tms52_s${S}_w${W}_${N}.json
exec /root/lab/venvs/research/bin/python -m spd_min.train_tms --n-features 5 --n-hidden 2 --C 20 \
  --coeff-imp 3e-3 --pnorm 1 --steps $N --batch $B --seed $S --faith-warmup $W --threads 1 --out $OUT "$@"
