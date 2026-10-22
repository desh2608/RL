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

### B22 RoPE frontend fixed statistics fail for batched waveforms (fixed)
- **Symptom:** real step7000 Nano35 audio preflight on IAD (job 14017681) loaded all 487 sound tensors, then failed with `shape [2, 128] is invalid for input of size 128`.
- **Cause:** the saved frontend has one global mean/std vector per mel channel; the standalone preprocessor normalization expects batch-shaped statistics when called with multiple waveforms.
- **Fix:** Bridge materializes each waveform separately, trims to its valid frames, then pads the resulting mel tensors. This follows native MCore `NemoAudioProcessor.materialize`; the trained statistics and encoder math stay unchanged.
- **Check:** `test_fixed_global_normalization_materializes_per_waveform` compares a two-wave batch against individual preprocessing and verifies raw-wave and mel encoder outputs agree. GPU retry uses the actual saved checkpoint.

### B23 RoPE HF checkpoint has V3 processor metadata with V2 source names (fixed)
- **Symptom:** step7000 Nano35 HF processor preflight (IAD job14018446) fails with `processing has no attribute NemotronH_Nano_Omni_Reasoning_V3Processor`.
- **Cause:** `preprocessor_config.json` names V3 processor/image/video classes, but the shipped Python files define `NemotronNanoVLV2*`. Model/config already carry historical aliases; the processor files do not.
- **Fix:** Bridge's NeMo audio asset packaging adds only missing known aliases, retaining the original processor implementations and metadata. Original HF checkpoint is untouched. The learner adapter also reuses source-decoded waveforms, preserving file-path and singleton NumPy input support.
- **Check:** `test_export_repairs_historical_processor_labels_without_replacing_code` checks aliases, idempotence, and preservation of existing classes; `test_processor_adapter_uses_decoded_singleton_waveforms` checks decoded-path/singleton equality. GPU Bridge/AutoProcessor preflight must confirm saved token counts and mel lengths. Applies to offline and online audio paths.

### B24 Tau RoPE checkpoint frozen vision projector width differs from default (fixed)
- **Symptom:** distributed HF import in IAD job14018749 fails at `vision_projection.encoder.linear_fc1.layer_norm_weight`: expected5120, checkpoint1280.
- **Cause:** the checkpoint saves `downsample_ratio=1.0`, while Bridge hard-codes 2x pixel shuffle (ratio0.5) and a four-times wider projector input.
- **Fix:** NeMo audio provider translation carries the saved ratio into a Bridge family configuration field. Model projector geometry and patch handling honor0.5 or1.0; existing default0.5 is preserved. HF export preserves the saved ratio. No evaluation launcher changes.
- **Check:** `test_saved_vision_projector_geometry_preserves_frozen_weights` verifies both widths and patch handling. Full HF import must complete with the checkpoint's1280-wide frozen projector. Applies to offline/online audio training when full multimodal weights are retained.

### B25 HF SoundConfig serialization loses the NeMo backend label (fixed)
- **Symptom:** strict export after successful two-update job14019801 writes all7125 tensors, but JSON says `sound_config.model_type=parakeet` with `encoder_config.architecture=rope_transformer`. Native generation would choose the wrong encoder.
- **Cause:** HF `PretrainedConfig.to_dict` writes the class-level model_type. Historical `SoundConfig` declares class defaultParakeet while storing the saved backend per instance. Historical bridge aliases also bypassed Omni config/export hooks.
- **Fix:** Bridge preserves the explicit backend in provider/HF config translation and packaging, installs a per-instance SoundConfig serializer in exported assets, and routes historical NeMo labels through the Omni export hooks. The requested Tau HF wrapper has the same serializer override. Parakeet defaults remain unchanged.
- **Check:** `test_export_and_hf_reserialization_preserve_nemo_backend` reproduces the class-default bug and checks corrected JSON plus reserialization. GPU smoke must reload exported AutoConfig, assert backendNeMo before/after `to_dict`, and verify saved projector ratio1. Applies to offline/online audio exports and generation dispatch.

### B26 Gym audio sidechannel missing from learner trajectories (fixed; end-to-end validation pending)
- **Symptom:** Gym can generate with metadata audio, but the learner receives no audio tensors for the returned audio placeholder tokens.
- **Cause:** the Gym model server already handles `audio_data`/`audio_path`/`audio_paths`; the RL Gym postprocessor only attaches image model inputs.
- **Fix:** decode the original metadata sidechannel with SoundFile and the same mono convention, use the Bridge processor frontend, attach packed `sound_clips`/`sound_length` on the first trainable turn, and require exact per-clip rollout placeholder counts. Text-only behavior is unchanged. Paths must be absolute and audio must already use the encoder sampling rate (16kHz for this checkpoint); pre-resample a shared input file once rather than using different engine resamplers. Relative audio roots are not accepted implicitly on the learner.
- **Check:** eight CPU regressions in `tests/unit/models/test_nemo_gym_audio.py`; the actual offline RL smoke must additionally show audio-bearing learner batches, finite updates, and successful subsequent rollout/refit. CPU checks alone do not certify the complete training path.

### B27 Native vLLM processor rejects saved vision ratio1 before audio generation (fixed; GPU retry pending)
- **Symptom:** full17-shard model load succeeds, then an audio-only request fails in DynamicResolutionImageTiler's ratio<1 assertion.
- **Cause:** the native model and saved frozen projector accept ratio1, while the pinned processor constructor assumes a twofold reduction even when no image is supplied.
- **Fix:** Bridge's explicit NeMo runtime dispatch wraps that constructor: retain its setup and select unit reduction for the saved ratio1. Existing ratio0.5 delegates unchanged; unsupported ratios still fail. Model/checkpoint geometry is preserved.
- **Check:** `test_native_tiler_preserves_unit_scale_and_default_geometry` checks token arithmetic and default behavior; full native audio generation must pass before RL submission. This does not claim image evaluation acceptance.

## B28. Fresh-source learner cannot build its dataset helper

- **Symptom:** Bridge learner initialization fails with `cannot open output file helpers_cpp.cpython-313-x86_64-linux-gnu.so: Read-only file system`, after the HF import succeeds.
- **Cause:** The launcher mounts the extracted source read-only, while Bridge unconditionally invokes MCore's dataset-helper Makefile on each learner node. A fresh third-party checkout has no helper compiled for the container's Python ABI. `NVC_SKIP_MEGATRON_DATASET_HELPERS` is not consumed by this pinned Bridge initializer.
- **Fix:** Overlay just the node-local extracted MCore `datasets` directory as writable. The original immutable run/archive and the remaining source mount are unchanged; native helper compilation and its barrier are retained.
- **Check:** `bash -n voicechat/tools/gpu_job.sh`, frozen launcher tests, and a fresh native smoke must get past helper compilation on both learner nodes.

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


## Pivot certification: SDG gate rejection is not an error label (G5 draft)

The October 22 audit found that the old AReaL pivot collection could reward a
consent re-ask after the caller had already agreed. It selected the first SDG
online-gate rejection and reused the older detailed read-back gates, rather
than independently proving a first agent error under the approved consent rules.

The new `areal-v2:` evaluator adds a hard K4 eligibility check: the mined proof
must explicitly establish that valid prior consent is absent. A true or unknown
consent status gives reward zero. K5 instead requires a typed gold write and
rejects another consent question. Other write-kind positives must also carry
prior authorization or proof that consent is inapplicable. Mutation checks are
in `resources_servers/tau_pivot_stateful/test_areal_v2.py` in the vendored Gym.

This guard is a development change. A version-pinned relaxed checker, independent
native first-divergence replay, all HTTP controls and a strategist audit are
still required for the replacement data. `final-20261018` remains unchanged.


## 2026-10-22 — G5: separate prior consent from content grading

The draft G5 evaluator trusted an eligibility boolean for K4. A stale false
label could therefore reward a consent question even when the public context
already contained a valid confirmation and yes. The evaluator now recomputes
consent with the independent INBOX31 detector; missing or contradictory context
fails closed. The 45-case kind/consent battery includes this stale-label attack,
negation, changed requests, choices, different actions and declarative readbacks.
A committed write closes permission for that write, while a different member of
a confirmed batch does not erase permission for the remaining members.

This fix does not complete G5 certification. The draft K3 lexical refusal scorer
still fails adversarial cases, and ambiguous or mismatched native sources remain
held. No G5 training bundle has been published.


## 2026-10-22 — G5 INBOX33: affirmative restatements and source traces

The independent detector missed 16 of Fable's 30 not-given cases because it
read restated action verbs as changed requests and required an interrogative
offer. It now recognizes grounded yes/no offers and an explicit go-ahead after
stated write details. A separate add-on does not erase assent for the original
write. All 60 Fable-labelled audit cases are checked as regression fixtures.

Choice, reason and payment answers do not provide new consent. Information-only
answers after a prior assent have an explicit uncertainty hold: they cannot
produce a K4-positive re-ask, even if a stale miner supplies a false boolean.
This protects the consent hard gate while the remaining segment ambiguity is
independently audited. The v3.2 content graders are unchanged.

K3 is deferred, and its versioned evaluator always returns zero. Native mining
preserves these sources as holds. Native reference replacement witnesses for
K1/K2, wrong-ID lookup errors for K6, and executable continuation for K7 are
candidates pending full controls/audit. B0 uses the T20 prompt; caller rewrites
are recorded, while any tool-call argument or result byte difference is held.
No replacement G5 training bundle is yet accepted or published.


## 2026-10-22 — G5 INBOX34: asymmetric segment assent veto

The second audit found nine clear confirmations among thirty not-given cases,
including affirmative restatements and go-aheads after declarative details. K4
now additionally requires no caller assent anywhere in the current write segment.
Clarifications and payment/reason answers do not erase earlier assent. A caller
question in the affirmative reply is held until a later clean assent. Explicit
negations and clearly different named targets/items are ignored with evidence.

`consent_label` is the broader census classification (given/hold/K4_candidate).
`authorized_prior_consent` still requires the turn49 positive witness and a
given classification: a newly recognized K4 veto alone cannot create a K5 or
write-positive reward. The original `valid_prior_consent` field is retained as
the turn49 detector result for provenance. The evaluator recomputes eligibility
from the public prefix, so a stale false miner label cannot override the veto.

The frozen turn49 SDG/B0 jobs remain candidate inventories under INBOX33; their
labels must be updated under INBOX34 before controls/audit/certification. Content
checker bytes and the old published bundle remain unchanged.

## 2026-10-22 — G5 INBOX35: reads before consent; precise re-ask witnesses

An absence-of-assent guard alone admitted consent pivots at authentication or
lookup boundaries. Several task evaluation ledgers omitted reads entirely.
The native G5 boundary now retains ordered read/auth calls from the recorded
reference trajectory before exact task-gold writes, with their origin explicit.
Every committed prefix result still needs exact native replay. Native due-step
checks are recomputed by callbacks; a proof boolean cannot make a later write
become due. Writes/explicit completion claims before a required lookup propose
K6; premature closes before that lookup propose K7. Missing reference anchors
and unresolved orderings remain holds.

K4 additionally needs a feasible write that is the next native ledger step.
K5 additionally needs a grounded agent question offer and clean affirmative,
without a caller question or surviving caller change. Go-ahead-only witnesses
still veto K4 but cannot create a K5 positive. All tool-positive kinds require
the returned native step to be due. New G5 candidate caches require fresh native
qualification; the published v1/AQ verifier paths are unchanged.

Pinned consent callbacks carry their approved source bytes and materialize them
under a local content-addressed directory after transport. No frozen checker is
edited, and no module is imported from Lustre. This stages the checker closure;
it does not rebuild the existing native-state cache.

A due lookup also requires its arguments to be available in the public context;
otherwise a gold ZIP or ID could become a target unavailable to the agent.
K4 has the same readiness check for its write arguments (for example, an
unprovided cancellation reason is a hold). These are conservative literal
readiness checks, not semantic matching of open fields. Alternative successful
authentication methods discharge an outstanding auth step only when native
execution returns the exact same user identity. Audited labels contradicted by
actual tool/result prefixes are quarantined for strategist review, never forced
with case-specific scoring exceptions.
