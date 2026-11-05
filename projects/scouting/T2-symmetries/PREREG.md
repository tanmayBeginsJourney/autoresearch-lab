# Preregistration: optimizer-aware gauge fixing (Adam checkpoint -> Muon fine-tuning)
Written 2026-11-05 before running `gauge_muon.py`. Exploratory CPU kill-test, not confirmatory.

Hypothesis: an Adam-pretrained checkpoint sits at a point of its symmetry orbit that suits Adam, not
Muon. Muon is equivariant to orthogonal maps of each matrix but not to the diagonal (ReLU per-neuron,
RMSNorm-gain) gauge. Teleporting into a balanced gauge (fold RMSNorm gains into W_u, equalize each
neuron's ||W_u row|| and ||W_d column||; zero function change, checked to 1e-4) improves Muon fine-tuning.

Setup: residual ReLU MLP, 3 blocks, d=64, d_ff=256. AdamW pretraining on teacher A (2000 steps),
fine-tuning on related teacher B (300 steps). Seeds 0,1,2 paired across arms. LR tuned per arm on a
validation batch (Adam 3 values, Muon 4 values); test batch reported once.

Arms: adam_native (reference), muon_native (strongest Muon baseline), muon_canon (treatment),
muon_random (log-normal sigma=1 neuron scales; sensitivity control).

Success: muon_canon B-test loss lower than muon_native by >= 5% relative on average AND lower on all
3 seeds. Kill: mean improvement < 5% or any seed worse. Secondary (not decisive): A-test forgetting,
and whether muon_random is worse than muon_native (if not, Muon is insensitive to this gauge in the toy).
