# Offline (single-step pivot) RL

**Use case**: you have dialogue trajectories (teacher samples, public data, synthetic tasks close to the eval) and want to
correct a specific behaviour, e.g. replying in text where a tool call is due, not executing a write after the user
confirms, or calling a tool on an irrelevant request.

**How it works**: each row gives the history up to some turn; the policy generates only the next turn. The Gym verifier
`single_step_tool_use_with_argument_comparison` compares it with the gold action (tool name, argument keys, argument
values) and returns 0/1. A message gold means "reply in text, do not call a tool"; any text reply passes.

`P=$NVC_RL_ROOT` below.

## 0. Start policy

An HF directory used as policy, tokenizer and rollout model. Exports from NeMo-RL / Megatron-Bridge need an
**hf-aligned view** first (`scripts/prepare_eval_hf_aligned.{py,sbatch}`), which restores the tokenizer and generation
metadata; otherwise the chat template differs from the eval side. An HF directory that has already been evaluated is the
safest start.

## 1. Candidate rows (route samples)

Layout (same as `scripts/profile_passrate`): `SAMPLES/routes.json` (one entry per route: `route_key`, `agent`,
`sample_path`, `n_prompts`, ...) and `SAMPLES/<route_key>/sample.jsonl` (Gym rows with `responses_create_params.input`
= history, `tools`, `expected_action` = gold, `agent_ref`).

`scripts/pivot_data/example_build_candidates.py` is one data recipe: it splits the shared pools into 11 routes by the
behaviour they target:
- BFCL multi-turn: should call, several calls, should stop;
- irrelevance: hard no-call, near-miss;
- single-step argument values;
- tau2 writes;
- telecom, tau_mms, CTU.

Adapt it to your pools. Pool indexing tools: `scripts/pivot_data/{index_pivot_pools,summarize_pivot_index,materialize_pivot_rows}.py`.
Difficulty ledger: `scripts/pivot_data/build_difficulty_ledger.py` recovers always-correct prompts from the zero-variance
drops in a training log so later bundles can exclude them.

## 2. Clean to the eval prompt shape (required)

```bash
python3 scripts/pivot_data/clean_candidates.py --src SAMPLES --out CLEAN     # CPU job; a few minutes
```

What it does (BUGS_FIXED B4, B6, B7):
- removes history reasoning items;
- strips the tau2 `ENVIRONMENT REMINDER`;
- removes assistant text emitted in the same turn just before the gold write call (otherwise the policy is asked to open a
  second assistant turn after its own message);
- drops prompts that end with an unanswered call;
- forces `parallel_tool_calls: true`;
- ensures no sampling seed;
- drops duplicate prompts.

`CLEAN/CLEAN_MANIFEST.json` reports per-route changes.

## 3. Pass-rate profile of the start policy (K=8)

```bash
NVC_MODEL_PATH=<start HF> NVC_MODEL_LABEL=<name> \
  sbatch ... scripts/profile_passrate/profile_candidates.sbatch $P/analysis/<name>/g1 CLEAN route_a,route_b 8
```

- One 8-GPU node: stock `vllm serve` (TP1 DP8 EP8) with the same Gym verifiers as training
  (`scripts/profile_passrate/gym_profile_pivot.yaml`). About 2 h per 6–8k prompts; split routes over several jobs.
- At start it writes `RENDER_CHECK.json` and exits unless system messages render as plain text (B3).
- Profile conditions must match the training rollouts: T=1.0, content format string, fp32 SSM cache, same
  `max_output_tokens`. Otherwise prompts selected as "mixed" can become all-correct in training.
- Output per group: `per_prompt.jsonl` (pass rate per prompt), `route_summary.json`, `SUMMARY.md`.

## 4. Build the bundle

```bash
python3 scripts/pivot_data/build_bundle.py --profile analysis/<name> --clean-manifest CLEAN/CLEAN_MANIFEST.json \
  --out data/<bundle> --bundle-name <bundle> --start-policy <start HF> --cap route_a=1200 --cap route_b=400
```

- Keeps only prompts with 0 < pass < 1; all-correct and all-wrong groups carry no GRPO signal and are dropped in
  training anyway.
- Asserts on every row: no seed, `parallel_tool_calls` true, no history reasoning, no reminder, no trailing assistant item.
- A route may reuse an older profile with `--reuse OLD:route`, but only if cleaning changed none of its rows and none of
  its rows has a system message.
- Steps ≈ rows / `num_prompts_per_step`; async GRPO cycles through the data (`max_num_epochs`).

## 5. Render the recipe

```bash
python3 voicechat/configs/render_recipe.py --template $P/config/offline-pivot-rl.template.json --out $P/config/<run>.json \
  --student <start HF> --bundle $P/data/<bundle> --lineage <lineage> --services $P/config/gym-services-offline-pivot.yaml \
  [--set grpo.max_num_steps=20 --set policy.megatron_cfg.optimizer.lr=2e-6 ...]
```

Template defaults (all fixes on):

| Item | Default |
|---|---|
| Resources | 14 nodes: 10 rollout + 4 learner |
| Parallelism | learners TP4 / CP2 / EP8 with sequence packing; rollout TP1 × DP8 × EP8 per node |
| Batch | P512 × K16, GBS 8192 |
| Learning rate | lr 2e-6, cosine to 2e-7, warmup 5, 20 steps |
| Advantage | leave-one-out baseline + whole-group std of the unshaped reward, clip ±5 |
| Loss | `force_on_policy_ratio`; TIS truncated to [0, 5]; no reference KL |
| Length | `max_new_tokens` 2048 per turn (BFCL eval budget); length penalty coefficient 0.2 |
| Numerics | fp32 LM head on both engines, fp32 vLLM SSM cache, `MAMBA_DETERMINISTIC=1`, fp32 Mamba training states, system rendered as text |

- When changing the batch, keep `policy.train_global_batch_size` = `generation_batch_size` = P×K; `make_run` checks this.
- When changing lr or steps, keep `grpo.max_num_steps`, `policy.megatron_cfg.train_iters` and `scheduler.lr_decay_iters`
  consistent.

## 6. Smoke (3 nodes, 3 steps, strict resume 3 → 4)

Run inside the training container (driver venv):

```bash
export HF_MODULES_CACHE=/tmp/hfmod NVC_INPUT_MODE=gym; source /workspace/debug/scripts/runtime_env.sh
/opt/nemotronvc/venvs/driver/bin/python /workspace/debug/scripts/make_run.py --kind smoke --project $P \
  --recipe $P/config/<run>.json --services $P/config/gym-services-offline-pivot.yaml --bundle $P/data/<bundle> \
  [--new-student-reason '<why this start policy>']
```

Then submit it from the host with `bash scripts/submit_run.sh $P/runs/smoke-v1-<UTC>Z`.

If `make_run` fails on `git status` with "dubious ownership" (the container user is root), run
`git config --global --add safe.directory '*'` first.

After the smoke ends: `END_CODE` must be 0 and `SMOKE_PASS.json` must exist. Then:

```bash
python3 scripts/audit/smoke_check.py $P/runs/smoke-v1-<UTC>Z --tokenizer <start HF>
```

It prints ACCEPT only if all of these hold:
- no system message renders as a Python list;
- short-answer groups have on average ≥ 10/16 distinct generations;
- advantages stay within ±5.5;
- mismatch and KL are reported for comparison with a reference run.

## 7. Train and finish

```bash
make_run.py --kind train ... --smoke-run $P/runs/smoke-v1-<UTC>Z      # same arguments as the smoke
bash scripts/submit_run.sh $P/runs/train-v1-<UTC>Z                   # prints [SUBMITTED] job=...
TJ=$(cat $P/runs/train-v1-<UTC>Z/JOB_ID)
sbatch ... scripts/finish_chain.sbatch $TJ <lineage> <target step> $P/config/<run>.json \
  $P/config/gym-services-offline-pivot.yaml $P/data/<bundle> $P/runs/smoke-v1-<UTC>Z <start HF> "<new-student-reason or empty>"
```

**Segments**: each segment is bounded by the 4-hour partition limit. The trainer saves an MCore checkpoint and exits about
3 h 40 min after start. It also saves an ft checkpoint every 5 steps and exports HF every 10 steps.

**finish_chain**:
1. waits for the job;
2. if the target step has not been reached, builds and submits a `--resume` run with the same inputs (at most 2 rounds);
3. builds the hf-aligned view of `hf/step_N`;
4. with `NVC_EVAL_ROOT` set, launches bfcl_v3 and tau2.

Log: `$P/logs/finish-chain-<job>.log`.

**Where to look**:
- `runs/<run>/driver.log`: `Step N/…`, `Advantages stats`, `Dropped zero-variance prompt group`;
- `runs/<run>/train/exp_001/tensorboard`: readable with `scripts/audit/tb_scalars.py`, no tensorboard needed;
- `runs/<run>/train/exp_001/closed_trajectories/stepN.jsonl`: 256 full samples per step with tokens, masks, advantages
  and vLLM logprobs.

## 8. Eval

Eval uses the NemotronVC_eval harness. Evaluate the hf-aligned view with the harness tokenizer
`eval/text/tokenizers/student-kevin-nvc4-sft-18k-fix-mistral`.
