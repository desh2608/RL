# Acceptance and audit tools (`scripts/audit/`)

All of these read only artifacts that a run directory already has, or are CPU-runnable tests. Unless noted they are
stdlib-only.

## Smoke / training acceptance: `smoke_check.py`

```bash
python3 scripts/audit/smoke_check.py RUN_DIR --tokenizer <policy HF> [--ref OTHER_RUN_DIR]
```

| Check | Pass condition | Bug |
|---|---|---|
| 1. System rendering | no system segment in the decoded closed_trajectories is a list repr | B3 |
| 2. Group diversity | groups with mean length < 200 tokens have on average ≥ 10/16 distinct outputs | B4 |
| 3. Advantages | `Advantages stats` in driver.log within \|5.5\| | B5 |
| 4. Mismatch | prints `token_mult_prob_error`, `gen_kl_error`, reward for comparison with a reference run | B8, B9 |

Reference values after the fixes (Nemotron 3.5 30B-A3B SFT start, P512×K16 smoke): mismatch 1.015–1.018, gen_kl
0.0006–0.0008, advantages exactly ±4.

## Per-step sample audit: `dump_audit.py`

```bash
python3 scripts/audit/dump_audit.py RUN/train/exp_001/closed_trajectories/step5.jsonl
```

What it checks:
- whether the loss mask is one contiguous span at the end of the sequence;
- whether rollout logprobs on the prompt are zero;
- reward values per agent, shaped vs unshaped;
- whether the advantage is constant within a sample.

Note: under `force_on_policy_ratio` the dumped `prev_logprobs` are all zero (not computed), so its mismatch statistics are
not meaningful.

## Advantage fix test: `test_adv_std_scope.py`

Runs with the driver venv (torch, pydantic):

```bash
PYTHONPATH=<repo> python scripts/audit/test_adv_std_scope.py [--legacy-module <old advantage_estimator.py>] [closed_trajectories/stepN.jsonl ...]
```

It covers three things:
- defaults are bit-identical to the old code (requires the old module);
- with the new keys a lone failure / lone success gets −4 / +4 and all-equal groups stay 0;
- on real dumps it recomputes the legacy advantages, matches them against the recorded values, and prints the share of
  gradient weight carried by the top 1% of samples.

## Weight diff: `weight_diff_hf.py`

Run on a CPU node; I/O is about 2× the checkpoint size:

```bash
python scripts/audit/weight_diff_hf.py --a <start HF> --b <RL step N HF> --out REPORT.json
```

Per category (vision, sound, router bias, router weight, MoE experts, Mamba parts, attention, norms, embedding, lm_head)
it reports the fraction of changed elements, an ulp-distance histogram and the relative L2.

Use it to check that:
- frozen parts are bit-identical;
- there are no outlier changes (which would point to an export or conversion bug);
- the update has a plausible size: at lr 2e-6 for 20 steps, about 6% of bf16 elements move by 1–2 ulps.

## Mamba determinism probe: `mamba_pin/`

`det_probe.sbatch` runs `tune_ssd.py --probe` once per GPU on one 8-GPU node: forward + backward, the chosen kernel
configs and output hashes, to compare across GPUs.

Use `--probe` only. The pinning helpers used by `--pin` and by the tune mode were removed from `mamba_numerics.py` (see
B2).

## Gradient-sync probe: `gradsync_check/`

It drives the real worker `train()` through torchrun and compares per-parameter gradients of DP4 against a single-replica
DP1 reference on the same batch.

- About 10 min on 1+4 nodes.
- The HF import cache is about 62 GB per layout; delete it afterwards.
- Re-run after changing the parallel layout or upgrading Megatron-Bridge.
- It can be adapted to compare `grad_reduce_in_fp32` true vs false (the risk item in BUGS_FIXED).

## TensorBoard reader: `tb_scalars.py`

A minimal event-file reader with no tensorboard or protobuf dependency:

```bash
python3 scripts/audit/tb_scalars.py RUN/train/exp_001/tensorboard/events.out... '^train/(reward|token_mult_prob_error)$'
```
