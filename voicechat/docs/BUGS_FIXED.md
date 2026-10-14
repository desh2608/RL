# Bugs found and fixed

As of 2026-10-14 this code went through about six debugging rounds:
- around 09-20: resume, packing, logging;
- 09-29: diagnosis of why RL did not move scores;
- 10-01..03: online pipeline stability;
- 10-06: gradient-sync audit;
- 10-13: Mamba numerics;
- 10-13: full low-level audit.

Each entry: symptom / cause / fix / how to check. "Fixed" means the code or the template defaults include the fix;
"config" means it is enabled by a recipe key (already set in the templates).

## Training signal and prompt rendering (largest impact)

### B3 System messages rendered as a Python list repr (fixed, config)
- **Symptom**: decoded rollouts start with `<|im_start|>system\n[{'type': 'text', 'text': '...\n...'}]`: the whole policy
  on one line, with literal `\n`. The eval side renders plain text. Every Gym RL run since 2026-10-02 had this; all
  tau2/CTU/telecom-style rows were affected.
- **Cause**: the vLLM HTTP server embedded in NeMo-RL defaults to `chat_template_content_format="auto"`
  (`nemo_rl/models/generation/vllm/vllm_worker_async.py`). Gym sends content-part lists, auto-detection picks the openai
  format, and the template's `| string` on the system message prints the list. Stock `vllm serve` (used for profiling)
  does the same.
- **Fix**:
  - training: `policy.generation.vllm_cfg.http_server_serving_chat_kwargs.chat_template_content_format: "string"`;
  - profiling: `launch_profile.sh --content-format string` (default).
- **Check**: `smoke_check.py` item 1; `RENDER_CHECK.json` in profile output.

### B4 All K samples of a prompt shared one seed (fixed, data)
- **Symptom**: at T=1.0, ~100-token answers produced only 2–6 distinct outputs per group of 16, with long shared
  prefixes. Prompts profiled as mixed became all-correct or all-wrong in training (58% of groups dropped).
- **Cause**: the profiler's materialized inputs carry `responses_create_params.metadata.extra_body = {"seed": k}` per
  repeat. Using them as training rows kept seed 0; Gym merges extra_body into the request, and NeMo-RL does not change it
  when replicating a row K times.
- **Fix**: `pivot_data/build_bundle.py` strips the seed and asserts; `clean_candidates.py` asserts too.
- **Check**: `smoke_check.py` item 2. After the fix all 48/48 groups had 16 distinct outputs.

### B5 Leave-one-out std plus length penalty blew up advantages (fixed, config)
- **Symptom**:
  - with 15 correct and 1 wrong, the wrong sample got −17 to −20 (−26 before clipping);
  - with 1 correct and 15 wrong, the correct one got only +1;
  - with 2 correct, the correct ones got +4.48.

  1–1.5% of samples carried 10–46% of the gradient weight per step.
- **Cause**: the std is taken over the *other* 15 samples (`nemo_rl/algorithms/utils.py::calculate_baseline_and_std_per_prompt`).
  The length penalty spreads correct rewards over 1 ± 0.2, so the std of 15 correct samples is ~0.06; the std of 15 zeros
  is 0 and then nothing is normalised.
- **Fix**: two new keys in `nemo_rl/algorithms/advantage_estimator.py` (defaults keep the old behaviour):
  - `grpo.adv_estimator.std_scope: group` (whole-group std);
  - `grpo.adv_estimator.std_source: unshaped_reward` (std of the verifier reward before the length penalty).

  Also set `grpo.advantage_clip_low/high: ±5`.
- **Check**:
  - `scripts/audit/test_adv_std_scope.py`: defaults are bit-identical to the old code; with the new keys a lone
    failure / lone success gets −4 / +4;
  - `Advantages stats` in driver.log stay at ±4.

### B6 `parallel_tool_calls: false` hid extra calls from the verifier (fixed, data)
- **Symptom**: the model emitted two `<tool_call>`s, the single-call verifier still gave 1, and the multi-call tokens were
  reinforced.
- **Cause**: with `parallel_tool_calls: false` vLLM returns only the first call, so the verifier's
  `tool_call_count > 1 → 0` check never fires. The chat template never uses the flag either.
- **Fix**: force `parallel_tool_calls: true` on every row (asserted in `clean_candidates.py` and `build_bundle.py`).

### B7 RL prompts did not have the eval prompt shape (fixed, data)
- History reasoning was kept, but the eval clients (bfcl, tau2) never send previous reasoning back.
- `ENVIRONMENT REMINDER: You have N turns left` left over from tau2 sampling; the eval has none.
- Prompts ended with assistant text: the teacher spoke and then called a tool in the same turn, and after pivoting the
  policy had to open a second assistant turn after its own message, which never happens in eval.
- **Fix**: `clean_candidates.py` (same rules as the SFT converter).

## Numerics

### B1 Mamba training SSM state dtype silently BF16 (fixed)
- **Symptom**: recipes set `policy.megatron_cfg.mamba_training_ssm_states_dtype: float32`, but the model ran with BF16.
- **Cause**: NeMo-RL never forwarded the key to the Megatron model config; mamba_ssm accepts state_dtype, so nothing
  failed.
- **Fix**: `nemo_rl/models/megatron/mamba_numerics.py` plus 3 lines in `setup.py`. Every rank prints
  `[MAMBA_NUMERICS] {...}` and asserts the dtype of every MambaMixer.
- **Check**: grep `MAMBA_NUMERICS` in driver.log; expect `"state_dtypes": {"torch.float32": 23}`.

### B2 Mamba SSD Triton kernels autotuned independently per rank (fixed, config)
- **Symptom**: same weights and inputs gave different logprobs and gradients on different ranks.
- **Fix**: `policy.megatron_cfg.env_vars.MAMBA_DETERMINISTIC: "1"` (mamba_ssm's own deterministic mode). On 8 GPUs the
  forward and gradients are bit-identical, with no slowdown.
- **Do not** pin configs by editing `Autotuner.configs`: some workspaces are sized at import from the smallest
  BLOCK_SIZE_N, and pinning a larger block leaves uninitialised tiles (gradients off by 1e8). Probe: `scripts/audit/mamba_pin/`.

### B8 Logits rounded to bf16 on both engines, the main train/rollout mismatch (fixed, config)
- **Symptom**: `train/token_mult_prob_error` ≈ 1.034, `gen_kl_error` ≈ 0.0027, already at step 1 (fresh weights), i.e.
  numerical.
- **Fix**: enable the fp32 LM head on both engines:
  - learner: `policy.megatron_cfg.fp32_lm_head: "tf32"` (inputs are exact bf16 values, so TF32 tensor cores are
    numerically equivalent and near baseline speed);
  - rollout: `policy.generation.vllm_cfg.env_vars.NRL_VLLM_FP32_LM_HEAD: "1"`.

  One side alone makes things worse; `setup.py` rejects a mismatch.
- **Measured** (together with B9): mismatch 1.034 → 1.017, gen_kl 0.0027 → 0.0007, step time +15–20%.

### B9 vLLM SSM cache in fp16 with stochastic rounding (fixed, config)
- **Cause**: on A100 the stochastic rounding path is a software emulation: |x| < 6.1e-5 flushes to zero, |x| ≥ 65520
  becomes Inf (no saturation). Every decode step writes the state back through it, while the learner keeps fp32 states.
- **Fix**: `vllm_kwargs.mamba_ssm_cache_dtype: float32`, and drop `enable_mamba_cache_stochastic_rounding` /
  `mamba_cache_philox_rounds`. This is the upstream Nemotron default.
- **Note**: the eval harness still uses fp16 + stochastic rounding; model-to-model comparisons stay fair, but absolute
  scores may be depressed.

### B10 TIS lower bound 0.2 lifted corrupt tokens (fixed, config)
- **Cause**: one `token_mult_prob_error = 8.19` spike came from about 35 isolated tokens whose rollout logprob exceeded
  the learner's by 14–17 nats. The 0.2 floor raised their weights (≈ e^-15) to 0.2, so they were still trained on.
- **Fix**: `loss_fn.truncated_importance_sampling_ratio_min: null` (floor 0).
- **Unresolved**: the root cause of the spike is unknown; the suspect is a token/logprob misalignment after a refit
  preempts requests. Learner logprobs are not dumped under `force_on_policy_ratio`.

### B11 MoE router expert bias drifted during RL (fixed, config)
- **Symptom**: `e_score_correction_bias` drifted ~0.001 per step in all 23 layers, independent of the learning rate.
- **Fix**: `moe_router_bias_update_rate: 0` + `freeze_moe_router: true`.
- **Check**: `scripts/audit/weight_diff_hf.py` between start and step N: router weights and biases must be bit-identical
  (verified).
- **Note**: the bias stays fp32 only through `MoEFloat16Module`, which is used only with `freeze_moe_router: true`; the
  bias sits near 56, where a bf16 cast wipes it out.

### Risk: gradients accumulated in bf16 (unchanged, risk)
`grad_reduce_in_fp32: false` (all upstream RL recipes). Each DP rank accumulates at least 236 microbatches per step;
estimated relative error ~3–4% per gradient element. Not measured yet; see `scripts/audit/gradsync_check` and CHECKS.md.

## Pipeline and infrastructure

### B12 `truncated` flag and overlong filtering (fixed)
`truncated` used to compare the total token count with `max_model_len`, so turns that hit `max_new_tokens` were never
flagged. On the Gym paths it now means the turn hit `max_new_tokens`
(`tests/unit/experience/test_nemo_gym_truncation.py`).

### B13 Periodic validation crashed training on the Gym path (worked around)
`validate()` groups by the training group size while validation has 4 samples per prompt, raising
`Expected one _ng_task_index per prompt group`. `make_run.py --kind train` disables periodic validation by default
(`--keep-gym-validation` to keep it).

### B14 Engine affinity sent all tau2 rollouts to one engine (fixed, Gym)
Every tau2 episode starts with the same system prompt and opening line. The affinity key now runs up to the first
assistant reply after the first user message (Gym `responses_api_models/vllm_model/app.py`, test `test_engine_affinity.py`).

### B15 Requests kept hitting a dead vLLM engine (mitigated, Gym)
After a refit, rarely `RuntimeError: deque mutated during iteration`, then `EngineDeadError` / HTTP 500 from that engine.
Gym's `vllm_model` now puts an endpoint on a 120 s cooldown after a 5xx and fails over to a healthy one. The
`EngineDeadError` count in monitoring should be 0.

### B16 Resume with a different learner node count failed on RNG state (fixed, config)
RNG state is sharded by (PP, TP, DP×CP); when DP changes set `policy.megatron_cfg.checkpoint.load_rng: false`.

### B17 Three defects in hybrid distributed optimizer (HDO) resume (fixed, third-party patches)
- FP32 masters were not synchronised after a DP-reshardable load;
- CPU copies of parameters that were already FP32 were not restored;
- a non-persistent DCP template overwrote the restored step.

Fixes in `voicechat/third_party/megatron-lm-patches/` and `megatron-bridge-patches/`. Resume receipts carry
`master_model_exact_equal` (fatal on mismatch with `NVC_HDO_MASTER_MODEL_STRICT=1`).

### B18 Omni packing and context parallelism (fixed, third-party patches)
Omni CP1/PP1 attention must keep the physical packing layout; two-way context-parallel text packing was added
(Megatron-Bridge patches 0001, 0002).

### B19 Gradient-sync audit (verified OK)
In another KD stack, the provider deep-copied the config while `finalize_model_grads_func` was bound only to the original,
silently skipping DP sync. Here:
- the callback is bound to the DDP-wrapped model's config;
- DP4 vs DP1 gradients differ by 0.81% relative L2;
- the negative control (no callback) differs by 142%.

Probe: `scripts/audit/gradsync_check/`; re-run after changing the parallel layout or upgrading Bridge.

### B20 Configuration lessons
- **Match the eval generation budget**: `policy.generation.max_new_tokens` must equal the eval budget (BFCL: 2048 per
  turn including reasoning). Training with 8192 grew an 870-token thinking template; 11% of eval turns were then
  truncated and multi-turn fell from 44.9 to 36.6.
- **Keep the parallel verifier strict**: `allow_superset: true` + f1 gives "correct set + 1 extra call" 0.8, which teaches
  extra calls that leak into single-call prompts. Use `allow_superset: false` + `binary_strict`.
- **Keep enough message golds**: when they are only 12% and concentrated on many-tool prompts, irrelevance drops.
- **`make_run.py --nodes N`** forces N−2 generation nodes; do not pass it for real training.

### B21 Two defects in the finish chain (fixed, `finish_chain.sbatch`)
- The chain (64G) and the 64G prepare job it submits shared a per-user 88G memory limit on the same partition, so the
  child never started and the parent waited forever. The chain now requests 20G.
- The resume job id was read from `SUBMISSIONS`, but RL run dirs record `JOB_ID`. An empty id looked like a finished
  job and triggered a second resume build. It now reads `JOB_ID` and checks the id is numeric.

## Files changed relative to the upstream baseline (audit entry points)

```
nemo_rl/algorithms/{advantage_estimator,grpo}.py              B5; async collection, zero-variance drops, closed_trajectories dump
nemo_rl/algorithms/async_utils/{replay_buffer,trajectory_collector}.py
nemo_rl/environments/nemo_gym.py, nemo_rl/experience/{rollouts,rollout_manager}.py    B12; Gym result conversion
nemo_rl/models/generation/vllm/{config,vllm_generation,vllm_worker_async}.py          refit / prefix-cache reset, serving kwargs
nemo_rl/models/megatron/{setup,mamba_numerics,hdo_resume}.py, models/policy/workers/megatron_policy_worker.py   B1, B8, B17
nemo_rl/utils/{length_penalty,training_guardrails}.py, nemo_rl/data/full_pass.py
3rdparty/Gym-workspace/Gym: rollout_collection.py, vllm_model/app.py (B14, B15), openai_model/app.py (multiple base_url),
  tau2/app.py, single_step_tool_use_with_argument_comparison/* (parallel batch verification), math_with_judge/app.py
```
