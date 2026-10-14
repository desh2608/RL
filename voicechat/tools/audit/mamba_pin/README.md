# Mamba kernel numerics probes (RL audit 2026-10-13, analysis/rl-mamba-audit-20261013/REPORT.md)

- `tune_ssd.py --probe OUT.json`: one process per GPU runs the per-rank Mamba SSD forward+backward of our student
  (8 heads x 64, d_state 128, chunk 128, packed seq_idx, FP32 states) and records chosen kernel configs, forward/grad
  hashes and timing. `det_probe.sbatch` runs it on 8 GPUs with free autotuning and with `MAMBA_DETERMINISTIC=1`.
- Result (det-v1): free autotuning -> 7 kernels choose different configs across GPUs, 3 distinct forwards; deterministic
  mode -> forward AND gradients bitwise identical on all 8 GPUs, within 1.2e-4 of free autotuning, not slower.
- Do not pin configs by editing `Autotuner.configs` (pin-v1): several tile workspaces are sized at import from the
  smallest BLOCK_SIZE_N over all configs; a pinned larger block leaves uninitialised tiles and gradients off by 1e8.
  Use mamba_ssm's deterministic mode via `policy.megatron_cfg.env_vars.MAMBA_DETERMINISTIC: "1"`.
