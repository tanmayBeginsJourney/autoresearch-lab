---
name: experiment-loop
description: Run experiments efficiently and reproducibly on the Slurm cluster via researchctl - smoke, exploratory, confirmatory; right-sizing GPUs; failure triage; resource and token efficiency. Use whenever writing experiment code or submitting jobs.
---

# Experiment loop

## Config (`projects/P/configs/<name>.yaml`)
```yaml
name: actbias-sweep            # [a-z0-9-]
project: actbias
kind: smoke | exploratory | confirmatory
hypothesis: H001               # required for confirmatory
env: research | interp | kernels
entry: projects/actbias/src/train.py   # receives --params <run_dir>/params.json
profile: s24 | s48 | cpu | h200
time: "02:00:00"
seeds: [0, 1, 2]
grid: {activation: [gelu, gelu_centered], lr: [1e-3, 3e-4]}   # array task per point x seed
params: {steps: 20000, d_model: 512}
model: {id: Qwen/Qwen3-0.6B, revision: <commit sha>}           # optional; pin for confirmatory
data: {id: ..., revision: ...}
eval: {splits: [val]}          # 'test' only with prereg final_test: true, once
primary_metric: {name: val_loss, goal: min}
```
The entry point uses labkit:
```python
import labkit
with labkit.init() as run:
    torch.manual_seed(run.seed)  # labkit already seeds; repeat if you re-seed
    ...
    run.log(step, loss=..., tok_per_s=...)
    if run.should_stop: run.save(state, step); run.requeue()
    run.finish(val_loss=..., **other_final_metrics)
```

## Loop
1. `researchctl submit CFG --dry-run`: check the gates, task count and GPU-hours.
2. Smoke run (`kind: smoke`, a few hundred steps): does it run, is the loss falling, what are the
   throughput and peak memory? Fix the profile before scaling.
3. Exploratory sweeps: find the regime and the variance. Use small grids. Kill bad arms early.
4. Confirmatory: exactly the preregistered design. No peeking-driven changes; if you must change
   something, write a new prereg.
5. `researchctl wait EXP --timeout 240`, then `researchctl collect EXP`, then commit the record.
6. Failures: `researchctl status EXP`, then `researchctl inspect EXP/TASK`. Fix the root cause, then
   `researchctl retry EXP --reason "..."`. Never resubmit blindly.

## Efficiency checklist (apply before any job over 2 GPU-hours)
- Right-size: estimate memory (params x bytes + optimizer state + activations). s24 fits about 1B
  params for bf16 training with AdamW and a small batch, or about 7B for bf16 inference.
- Pack small runs: several seeds or grid points per GPU in one process, if each uses under 30% of
  the slice.
- Throughput: bf16 autocast, SDPA/flash attention, fused AdamW, `torch.compile` once correct, no
  `.item()` calls in the hot loop, and dataloader workers keep the GPU fed (watch tokens/s).
- Precompute and cache tokenized data on scratch once; never re-tokenize per run.
- Interpretability: cache activations once to scratch (fp16/bf16, sharded) and run analyses from
  the cache instead of rerunning the model.
- Early stopping on divergence (labkit records NaN), plus kill criteria for clearly losing arms.
- Kernels: benchmark with CUDA events, warmup, several repetitions, and report median and IQR.
  MIG slices share memory bandwidth and are not exclusive full GPUs, so final kernel numbers need a
  stated environment.

## Recording
Commit the experiment record (`projects/P/experiments/<id>.yaml`) after collect. Add one dated line
to PROJECT.md "Findings" per meaningful result, including negative ones.
