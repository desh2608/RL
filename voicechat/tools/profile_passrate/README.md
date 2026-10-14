# profile_passrate: per-prompt pass rate of a policy with the real Gym verifiers

Measures each prompt's pass rate under the policy you are about to train (K rollouts, T=1.0, thinking on), using the same
NeMo Gym resources servers as RL training, so that only prompts with 0 < pass < 1 (prompts that produce a GRPO signal)
go into a bundle. It does not use NeMo-RL. It drives a stock `vllm serve` (TP1 DP8 EP8 on one 8-GPU node) and Gym's own
CLI (`gym env start`, `gym eval run --no-serve`).

| File | Role |
|---|---|
| `profile_candidates.sbatch` | batch entry for offline pivot candidates (`pivot_data/clean_candidates.py` output); env `NVC_MODEL_PATH`, `NVC_MODEL_LABEL`, optional `NVC_PROFILE_SERVICES_YAML` |
| `launch_profile.sh` | inside the container: Ray head → `vllm serve` → render self-check (`RENDER_CHECK.json`) → `gym env start` → `gym eval run` per route → `aggregate.py` |
| `gym_profile_pivot.yaml` | Gym services for the pivot routes (same verifiers as training) |
| `gym_profile_services.yaml`, `gym-profile-tau2.yaml` | Gym services for library routes / tau2 episodes |
| `sample_routes.py` | sample N prompts per route from the RL data library into `routes.json` + `<route>/sample.jsonl` |
| `aggregate.py` | per-prompt pass rate, histograms, token / cap-hit / invalid-tool stats → `per_prompt.jsonl`, `route_summary.json`, `SUMMARY.md` |
| `manual_route.sh` | run one route against already-started services (debugging) |
| `launch_tau2_profile.sh`, `stackB_tau2_config.py`, `aggregate_tau2.py` | full tau2 episodes with a user simulator, using the eval harness's pinned tau2 runtime (online-style profiling) |

Key flags of `launch_profile.sh`:

| Flag | Default | Meaning |
|---|---|---|
| `--samples` | (required) | directory with `routes.json` |
| `--out` | (required) | output directory |
| `--routes` | all routes | comma-separated `route_key` list |
| `--k` | 8 | rollouts per prompt |
| `--max-num-seqs` | 48 | vLLM concurrency per engine |
| `--max-model-len` | 98304 | context limit |
| `--content-format` | `string` | system messages rendered as plain text, as in eval (BUGS_FIXED B3) |
| `--mamba-cache-dtype` | `float32` | match the training rollout setting |
| `--deadline-epoch` | — | stop starting new routes before this time |
| `--resume` | — | skip routes that are already finished |
| `--check-only` | — | only check the template and imports |

**Profile under the same conditions as the training rollouts** (temperature, content format, SSM cache dtype,
`max_output_tokens`). Otherwise the selected "mixed" prompts can turn all-correct or all-wrong in training.

**Remove the per-repeat seed before training.** The materialized inputs (`rollouts/<route>/rollouts_materialized_inputs.jsonl`)
carry a per-repeat sampling seed in `responses_create_params.metadata.extra_body`. Never use those rows for training
without removing it (`pivot_data/build_bundle.py` does; BUGS_FIXED B4).
