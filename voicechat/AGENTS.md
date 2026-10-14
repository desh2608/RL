# For coding agents working on nemo-rl-voicechat

Read `voicechat/README.md`, then `docs/OFFLINE_PIVOT_RL.md` or `docs/ONLINE_RL.md` for the task at hand. Read
`docs/BUGS_FIXED.md` before changing anything: every entry is a real failure we hit.

## Conventions

- Workspace root is `$NVC_RL_ROOT`; `$P` / `$R` in the scripts refer to it. `voicechat/tools/` is mounted in the workspace
  as `scripts/`, so `scripts/xxx` in the docs means `voicechat/tools/xxx`.
- Every training run is an immutable directory `runs/<kind>-v1-<UTC>Z/` built by `scripts/make_run.py` (config, data,
  source tarball, script copies). **Never edit a submitted run directory**; build a new one.
- Training outputs live in `experiments/<lineage>/{checkpoints,hf,monitor,eval}`. One writer per lineage at a time
  (the submit guard takes a lock).
- `make_run.py --resume` re-packs the **current** source and requires it to match the smoke's source. Do not edit
  `src/pipeline/nemo_rl` while a run still needs a resume.
- Always pass `--project $NVC_RL_ROOT` to `make_run.py` (the scripts dir is a symlink, so the default project path is wrong).
- Secrets (e.g. the hub simulator's API key) live only in `$NVC_RL_ROOT/.secrets/` (mode 600) and are injected through the
  environment; never put them in configs or logs.

## Offline workflow (details in docs/OFFLINE_PIVOT_RL.md)

1. Candidate rows (example: `scripts/pivot_data/example_build_candidates.py`)
2. `scripts/pivot_data/clean_candidates.py`: bring rows to the eval prompt shape
3. `scripts/profile_passrate/profile_candidates.sbatch`: K=8 pass rate of the start policy
4. `scripts/pivot_data/build_bundle.py`: keep 0 < pass < 1, cap routes, write the bundle
5. `voicechat/configs/render_recipe.py`: template → concrete recipe
6. `make_run.py --kind smoke` → `submit_run.sh` → `scripts/audit/smoke_check.py` must print ACCEPT
7. `make_run.py --kind train --smoke-run <smoke>` → `submit_run.sh`
8. `scripts/finish_chain.sbatch`: resume to the target step, build the hf-aligned eval view, optionally launch eval

## Rules for code changes

- Changes to numerics, loss, advantages or prompt rendering need: a CPU-runnable test (see
  `scripts/audit/test_adv_std_scope.py`), defaults that keep the old behaviour with the new behaviour behind a config key,
  and a smoke checked with `smoke_check.py` and the dump audit.
- Record new findings in `docs/BUGS_FIXED.md` (format in `docs/CONTRIBUTING.md`) and open an MR.
- Judge a run by more than the reward curve:
  - `train/token_mult_prob_error`: about 1.017 after the fixes; a value persistently above 1.03 means something is wrong;
  - `train/gen_kl_error`: about 0.0007;
  - `Advantages stats` in driver.log: about ±4;
  - the number of dropped zero-variance groups;
  - per-group sample diversity (`dump_audit.py`).
