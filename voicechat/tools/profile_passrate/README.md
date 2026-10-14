# profile_passrate — HF399 pass-rate profile with the real Gym verifiers (1 GPU node)

Purpose: measure HF399's per-prompt pass rate (K rollouts, temperature 1.0, thinking on) with the
same NeMo Gym resources servers the RLVR job uses, so prompts can be pre-filtered to pass-rate
0.2–0.8 before training (RL_DIAGNOSIS_20260929.md section 3.1 item 5 and 3.3). It does **not**
use NeMo-RL; it drives a stock `vllm serve` (TP1 DP8 EP8, the topology the eval workspace
already proved for this model family on this image) and Gym's own CLI (`gym env start`,
`gym eval run --no-serve`).

Everything runs on one `interactive_singlenode` allocation (8xA100, 4 h) inside the project
container `runtime/gym-v1/NemotronVC-Gym34-vllm0271-v1.sqsh` via `scripts/interactive.sh`.
Items marked **VERIFY** are assumptions that could not be checked from the login node; the
launcher checks them in `--check-only` mode before anything touches a GPU.

Files:

| file | role | where it runs |
|---|---|---|
| `sample_routes.py` | sample N prompts per route from the v43 library, write `routes.json` + per-route `sample.jsonl` | login node (gzip/json only) |
| `gym_profile_services.yaml` | the 18-server Gym config (2 vLLM wrappers, 8 resources servers, 8 agents) | read by the launcher |
| `launch_profile.sh` | Ray head -> `vllm serve` -> `gym env start` -> `gym eval run --no-serve` per route -> `aggregate.py` | inside the container on the GPU node |
| `aggregate.py` | per-prompt pass rate, histograms, token/cap/invalid-tool stats, `keep_pass_0.2_0.8.jsonl`, `SUMMARY.md` | login node or container |

A frozen sample set (256 prompts x 8 routes, seed 20260930) already exists at
`analysis/rl-plan-20260930/passrate-profile-plan/samples-v1/` (see its `SAMPLE_MANIFEST.json`);
you can point `--samples` at it instead of re-sampling.

## Routes covered (priority order = run order)

| key | `agent_ref.name` (= Gym agent block) | source rows / eligible after filters (samples-v1) | cap (`--max-output-tokens`) | steps | benchmark it speaks to |
|---|---|---|---|---|---|
| single_step_tool | `single_step_tool_use_with_argument_comparison_agent` | 19456 / 11277 (7721 `message` rows + 458 dups dropped) | 16384 | 1 | BFCL v3, tau2 |
| toolcall_schema | `toolcall_schema_single_step_tool_use_with_argument_comparison_agent` | 4096 / 1237 (1474 `message` + 1385 dups) | 16384 | 1 | BFCL v3 |
| search_pivot | `search_pivot_single_step_tool_use_with_argument_comparison_agent` | 16384 / 3907 (537 `message` + 11940 dups) | 16384 | 1 | agentic search |
| structured_outputs | `structured_outputs_simple_agent` | 2048 / 2010 | 65536 | 1 | IFEval/IFBench-adjacent |
| instruction_following | `instruction_following_simple_agent` | 7168 / 7158 (10 over input budget) | 65536 | 1 | IFEval / IFBench |
| workplace_assistant | `workplace_assistant_simple_agent` | 1024 / 278 (746 dups; 256 sampled = 92% of the unique set) | 16384 | <=6 | BFCL multi-turn / tau2 |
| multichallenge | `multichallenge_simple_agent` | 2048 / 686 (1346 dups, 16 over input budget) | 65536 | 1 + judge (policy judges itself) | MultiChallenge |
| conversational_tool_use | `conversational_tool_use_agent` | 8349 / 7743 scenarios (606 over input budget) | 8192 per step | <=50 + user/tool/judge sims | tau2 / tau3 |

Note: every sampled row on 7 of the 8 routes carries a historical `pass_rate` in {0.25, 0.5, 0.75}
(the v43 source pool was already pass-rate filtered for an internal model); the HF399 profile
is expected to differ, which is the point of running it.

Not runnable on this image without rebuilding it: `ifbench` (no pre-baked venv, clones GitHub at
start), `mcqa`, `xlam_fc`, `toolsandbox`, `tau2` agent (container probe in
`analysis/rl-plan-20260930/env-runnability/container_probe.log`). `code_gen`/`math_with_judge`
are deliberately excluded (lowest benchmark priority; code_gen would also need the sandbox sidecar).

Sampler filters (all counted in `MANIFEST.json`): rows whose `agent_ref.name` differs from the
route; `expected_action.type == "message"` rows on the three single-step tool routes (the frozen
verifier gives any text 1.0 there -> zero variance); rows whose historical `input_tokens/max` +
cap + 1024 > 98304 (would hit vLLM's context limit and be scored 0 as `finish_reason=length`);
byte-duplicates by `uuid`/`id`/content hash (the v43 pool repeats rows for weighting).

## Step by step

All commands from the workspace root `P=$NVC_RL_ROOT`
on the login node unless stated. `export TZ=UTC`.

### 1. Sample prompts (login node, ~1 min)

```bash
STAMP=$(date -u +%Y%m%dT%H%MZ)
PROFILE=$P/analysis/passrate-profile-$STAMP
python3 $P/scripts/profile_passrate/sample_routes.py --out $PROFILE/samples --n 256 --seed 20260930
# subset / per-route overrides:
#   --routes single_step_tool,toolcall_schema,instruction_following
#   --overrides '{"conversational_tool_use": {"n": 128}}'
```

Or reuse the frozen set: `PROFILE_SAMPLES=$P/analysis/rl-plan-20260930/passrate-profile-plan/samples-v1`.

### 2. Hold a GPU node (login node)

```bash
JOB=$(bash $P/scripts/interactive.sh alloc gpu | tail -n1)     # interactive_singlenode, 1 job per user, 4 h
bash $P/scripts/interactive.sh status
```

The first `exec`/`shell` imports the 82 GB image (about 2 min on a GPU node; the helper
serialises it with flock — never start two first-time imports for the same job).

### 3. Preflight inside the container (no GPU work yet, ~1 min)

```bash
bash $P/scripts/interactive.sh exec $JOB "bash $P/scripts/profile_passrate/launch_profile.sh \
  --samples $PROFILE/samples --out $PROFILE --check-only"
```

This verifies: model dir, worktree Gym root (`/workspace/nemo-rl/3rdparty/Gym-workspace/Gym`),
`/opt/nemotronvc/venvs/vllm027/bin/vllm`, `/opt/nemotronvc/venvs/nemo-gym/bin/{gym,ray,python}`
(**VERIFY**: the `gym` console script; the launcher falls back to
`python -c 'from nemo_gym.cli.main import main; main()'`), the ten pre-baked server venvs under
`/opt/nemotronvc/gym-v1/venvs/`, 8 GPUs, chat-template rendering with
`enable_thinking=True, truncate_history_thinking=False`, and prints the vLLM version (expect 0.27.1).

### 4. Launch (detached from the login node)

`interactive.sh exec` is a blocking `srun`; if it dies the step's processes are killed, so keep it
alive with `nohup`/`setsid` (or a tmux) on the login node:

```bash
END=$(date -d "$(squeue -j $JOB -h -o %e)" +%s)          # allocation end -> launcher deadline guard
mkdir -p $PROFILE
nohup bash $P/scripts/interactive.sh exec $JOB "bash $P/scripts/profile_passrate/launch_profile.sh \
  --samples $PROFILE/samples --out $PROFILE --k 8 --max-num-seqs 48 --max-model-len 98304 \
  --deadline-epoch $END" > $PROFILE/launch.login.log 2>&1 &
```

Options: `--routes k1,k2` (subset, priority order kept), `--resume` (Gym `resume_from_cache`:
re-dispatches only missing `(task, rollout)` pairs of an existing `rollouts.jsonl`),
`--assumed-tps 4000` (aggregate output tokens/s used only by the deadline guard),
`--mamba-cache-dtype float16` (mirror v8 training's fp16 SSM cache instead of the eval's fp32),
`--max-num-seqs 16` (v8 training value; use if 48 x DP8 runs out of KV/mamba cache — **VERIFY**).

### 5. Monitor

```bash
tail -f $PROFILE/logs/launch-$JOB.log            # phases: Ray ready -> vLLM ready -> Gym ready -> RUN/DONE per route
tail -f $PROFILE/logs/vllm-$JOB.log              # weights load ~5-10 min from lustre; look for data_parallel_size=8
tail -f $PROFILE/logs/gym-$JOB.log               # "All 18 / 18 servers ready!"
cat $PROFILE/timing.jsonl                        # per-route wall time / rollout counts
# live Gym status (port printed in the launch log):
bash $P/scripts/interactive.sh exec $JOB '/opt/nemotronvc/venvs/nemo-gym/bin/gym env status ++head_server.host=127.0.0.1 ++head_server.port=<GYM_HEAD_PORT>'
```

Per-route artefacts land in `$PROFILE/rollouts/<route_key>/`: `rollouts.jsonl` (one result per
rollout, appended as they finish), `rollouts_failures.jsonl` (transport/judge failures),
`rollouts_materialized_inputs.jsonl` (the K-expanded inputs), `rollouts_aggregate_metrics.json`
(Gym's own per-agent metrics), `eval-run-<job>.log`.

### 6. Aggregate (login node is fine; the launcher also runs it at the end)

```bash
python3 $P/scripts/profile_passrate/aggregate.py --out $PROFILE
cat $PROFILE/SUMMARY.md
```

Outputs: `per_prompt.jsonl` (route, source ordinal, raw-line sha256, K rewards, pass rate,
mean reward, zero-variance flag, output-token mean/max, cap hits, invalid-tool-call text hits,
historical pass rate), `route_summary.json`, `keep_pass_0.2_0.8.jsonl` (the prompts to keep;
join back to the library by `route_file` + `source_ordinal_1based` or `raw_line_sha256`, which is
the same hash `sources/super35_v43/rows.index.jsonl.gz` carries), `SUMMARY.md`.

Optional Gym-native profile of the same data: inside the container,
`gym eval profile --inputs .../rollouts_materialized_inputs.jsonl --rollouts .../rollouts.jsonl`
writes `rollouts_reward_profiling.jsonl` (per-task mean/std/histogram) next to the rollouts.

### 7. Release

```bash
bash $P/scripts/interactive.sh release $JOB      # removes /raid/scratch/$USER/nvcrl-$JOB (all caches) and scancels this job only
```

### Second allocation for what did not fit

`interactive_singlenode` allows one job per user, 4 h. Expect the three tool routes +
structured_outputs + instruction_following to fit in one allocation; workplace_assistant,
multichallenge and conversational_tool_use may need a second one. Re-run with
`--routes workplace_assistant,multichallenge,conversational_tool_use --resume` (same `--out`).
The deadline guard skips a route when its estimate does not fit; skipped routes are recorded in
`timing.jsonl` with `"skipped": true`.

## What the launcher does (so you can debug it)

1. Ports: one 1500-port block per `SLURM_JOB_ID % 32` slot (bind-checked): vLLM `base`,
   Ray `base+100..108` + workers `+200..399`, Gym head `base+450`, Gym servers `base+500..999`.
   Same scheme as `NemotronVC_eval/eval/runtime/internal/run_domain_inside_container.sh:60-90`.
2. Caches on node-local scratch `/raid/scratch/$USER/nvcrl-$JOB/profile/` (HF, triton,
   inductor, vLLM, uv); Ray temp dir `/tmp/nvr-$JOB` (kept short for the AF_UNIX path limit).
3. Environment mirrors `scripts/runtime_env.sh` for a stock server: `PYTHONPATH` =
   `HF_MODULES_CACHE:/workspace/nemo-rl:<Gym root>`, `NEMO_GYM_EXTRA_ROOTS=<Gym root>`,
   `NEMO_GYM_VENV_DIR=/opt/nemotronvc/gym-v1/venvs`, `UV_OFFLINE=1`, `HF_HUB_OFFLINE=1`,
   `RAY_DEFAULT_PYTHON_VERSION_MATCH_LEVEL=minor`, API keys unset.
4. `ray start --head` from the nemo-gym venv (`--num-gpus=0`; Gym's `RunHelper` insists on a
   Ray address).
5. `vllm serve` with exactly the eval's DP8/EP8 flags
   (`run_domain_inside_container.sh:436-452`) plus `--max-model-len 98304 --max-num-seqs 48
   --gpu-memory-utilization 0.85 --seed 42 --generation-config vllm`; readiness = `GET /v1/models`
   with `Authorization: Bearer dummy`.
6. `gym env start --config gym_profile_services.yaml --model hf399 --model-url
   '["http://127.0.0.1:<port>/v1"]' --model-api-key dummy ++head_server.* ++port_range_*
   ++ray_head_node_address=... ++skip_venv_if_present=true ++uv_venv_dir=/opt/nemotronvc/gym-v1/venvs
   ++results_dir=<out>/gym-results ++cache_dir=<scratch>/gym-cache ++nemo_gym_log_dir=<out>/logs/gym-servers`;
   readiness = `gym env status` reports `18 servers found (18 healthy`.
7. Per route: `gym eval run --no-serve <same flags> --agent <agent> --input sample.jsonl
   --output rollouts.jsonl --num-repeats K --temperature 1 --top-p 1 --max-output-tokens <cap>
   --concurrency <c> ++num_repeats_add_seed=true ++upload_rollouts_to_wandb=false`,
   wrapped in `timeout --signal=INT` sized to the remaining allocation.
8. `aggregate.py`, then cleanup in the order Gym (INT/TERM/KILL) -> Ray (TERM) -> vLLM (TERM),
   touching only PIDs it started. `PROFILE_EXIT.json` records the exit code.

## VERIFY list (could not be confirmed from the login node)

- `/opt/nemotronvc/venvs/nemo-gym/bin/gym` exists (console script). Fallback built in.
- `--max-num-seqs 48` per DP rank at `max_model_len 98304` with `gpu_memory_utilization 0.85`
  fits (v8 training used 16 per rank with the NeMo-RL server; the eval used 4–8 at 131072).
  If vLLM logs preemption warnings or OOM at load, use `--max-num-seqs 16`.
- The `gym env status` health string with exactly 18 servers (count derived from the YAML;
  changes if you edit it).
- multichallenge judged by HF399 itself with thinking on and `max_output_tokens 8192`: check
  `rollouts_failures.jsonl` is empty and rewards are not all 0 before trusting that route.
- conversational_tool_use cost: ~1 policy call + 1 user-sim call per step, a tool-sim call per
  tool call, and one judge call per agent message at the end; the 25k-token/rollout estimate is a
  guess. Watch `timing.jsonl`; run it last (it is).
- `++results_dir`/`++cache_dir`/`++nemo_gym_log_dir` keys are Gym 0.5.1 global-config keys
  (`nemo_gym/global_config.py:82-94`); if the CLI rejects one, drop it — the only effect is
  where Gym writes `results/`/`cache/` (default: under the Gym root, i.e. the worktree).
- `--generation-config vllm` is required so HF399's `generation_config.json`
  (`temperature 0.6, top_p 0.95, max_new_tokens 16384`) does not override request parameters.

## What can go wrong and how to check

| symptom | cause | check / fix |
|---|---|---|
| `interactive.sh alloc gpu` pending | `interactive_singlenode` allows 1 running job per user | `squeue -u $USER`; release the older job first |
| first `exec` hangs ~2 min then fails with `pyxis: dir_scan ... File exists` | two concurrent first-time imports of the same container name | never run two `exec`/`shell` before `logs/interactive/container-<job>.ready` exists; `interactive.sh` flocks this, but a manual `srun --container-name=nvcrl-<job>` bypasses the lock |
| preflight `MISSING .../.venv/bin/python` | route needs a venv the image does not bake (ifbench, mcqa) | drop the route or rebuild the image; `UV_OFFLINE=1` means Gym cannot build venvs in-job |
| vLLM dies at load with a CUDA OOM or `KV cache` error | 48 seqs/rank too many for 98304 ctx | `--max-num-seqs 16`; or `--gpu-mem-util 0.90` |
| vLLM ready but `data_parallel_size=8` not in the log | topology fell back | inspect `logs/vllm-<job>.log` head; the eval asserts this string (`run_domain_inside_container.sh:522-528`) |
| `gym env start` exits: `config_paths entry ... not found` | cwd is not the Gym root or `--config` path wrong | the launcher `cd`s to `$GYM_ROOT`; pass an absolute YAML path |
| `gym env start` stuck at `N / 18 servers ready` | one server crashed at import | `logs/gym-servers/<server>.log`; typical: nltk data (`/opt/nemotronvc/nltk-data` is baked), spacy absent in instruction_following venv (v8 ran without it), port collision inside `base+500..999` |
| `Could not connect to the head server` from `gym eval run` | `++head_server.host/port` differ from `env start` | both use `GYM_COMMON`; check `logs/launch-<job>.log` port line |
| many rows with `reward 0`, `response.status == incomplete`, tiny `output_tokens` | prompt + cap exceeded `max_model_len` (vllm_model maps the 400 error to `finish_reason=length`) | `aggregate.py` reports `incomplete_fraction`; lower the route cap or tighten `--input-margin` in the sampler |
| `rollouts_failures.jsonl` non-empty | judge/transport failures (multichallenge, conversational_tool_use) | Gym retries 3x (`NEMO_GYM_MAX_ROLLOUT_ATTEMPTS`); inspect `_ng_failure_class`; `--resume` re-dispatches only missing pairs |
| Ray fails: `AF_UNIX path length` | temp dir too long | launcher uses `/tmp/nvr-<job>`; do not override `TMPDIR` to a long path |
| Ray fails: port in use | slot collision with a leftover process on a shared node | node is `--exclusive`; the launcher re-slots automatically; never `ray stop` (other users' jobs) |
| judge or simulator needs network | not here: judge = simulator = the local policy server; no `inference-api.nvidia.com` dependency | `env | grep -i proxy` is empty on compute nodes; outbound works anyway (probe log) |
| `srun` complains about stdin | `interactive.sh exec` already redirects `</dev/null` | keep using the helper |
| out of time mid-route | 4 h hard limit | `--deadline-epoch` guard skips routes that will not fit; partial `rollouts.jsonl` is still valid for `aggregate.py` and `--resume` |
| node-local scratch left behind | job killed before cleanup | `interactive.sh release JOB` removes `/raid/scratch/$USER/nvcrl-JOB` |

## Time budget (estimate; no measured DP8 throughput survives on disk)

Per route, 256 prompts x K8 = 2048 rollouts. Output tokens per rollout: tool routes ~200–700
(HF399 v8 measured 452/526/218), structured_outputs ~6.1k, instruction_following ~6.7k (v8),
workplace_assistant ~4k (6 steps), multichallenge ~3k + 4 judge calls (~12k with thinking),
conversational_tool_use ~25k incl. simulators (guess). At an assumed 4000 aggregate output
tokens/s on DP8: tool routes ~5 min each, SO/IF ~50 min each, WA ~35 min, MC ~1.7 h, CTU ~3.5 h.
Plus ~15 min startup (image attach, weight load, 18 servers). Hence the two-allocation plan above;
`timing.jsonl` gives the real numbers after the first run and `--assumed-tps` should be updated
from `route_summary.json:output_tokens_per_second`.

## 2026-09-30 实跑后的补充（job 35838167，8×A100，polar3）

- 实测：vLLM TP1/DP8 加载 HF399 约 4 分钟（不是 5–10）；单步 route 2048 条 rollout 5–12 分钟；多轮环境受 episode 串行延迟限制，vLLM 384 槽位只用了 30–50 个，应多路并发（`manual_route.sh` 可对同一 Gym head 起多条采集）。
- `launch_profile.sh` 新增 `--services-only`（只起 vLLM/Ray/Gym 并常驻）和环境变量 `NVC_PORT_BASE`（固定端口块）；健康轮询已加 `++allow_openai_version_skew=true`（原缺失导致 15 分钟后误清理）。
- `manual_route.sh --resume` 可续跑；`launch_tau2_profile.sh` 用评测工作区 pinned runtime（cb588bb1）跑 tau2 agent，配置由 `stackB_tau2_config.py --sims key=model,...` 生成；`analyze_stage0.py` / `finalize_stage0.sh` 汇总。
- 上限 65536 = `max_model_len` 时 IF/SO/MC 全部 400 作废；IF 需用 16384。multichallenge 用 policy 自评 judge 会给空回答打分，不可用。
- 结果目录：`analysis/passrate-profile-20260930/`（STAGE0_REPORT.md、SUMMARY.md、STAGE0_TABLES.md、per_prompt.jsonl、keep_pass_0.2_0.8.jsonl、rollouts/、stackB*/）。
