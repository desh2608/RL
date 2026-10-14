# nemo-rl-voicechat

A fork of NVIDIA NeMo-RL for RL on Nemotron-H hybrid Mamba/MoE models (Nemotron 3.5 Omni 30B-A3B and its SFT descendants),
focused on tool calling and agentic dialogue (tau2, BFCL). Stack: NeMo-RL async GRPO + NeMo Gym, Megatron-Core learners,
vLLM 0.27.1 rollouts.

Everything outside `voicechat/` is the NeMo-RL tree with our fixes. `voicechat/` holds our tooling and docs:

| Path | Content |
|---|---|
| `voicechat/AGENTS.md` | **Read first (coding agents)**: conventions, workflow, what to check |
| `voicechat/docs/OFFLINE_PIVOT_RL.md` | Offline (single-step pivot) RL: candidates → cleaning → pass-rate profile → bundle → smoke → train → resume → eval |
| `voicechat/docs/ONLINE_RL.md` | Online RL: multi-turn episodes sampled live by Gym agents against a user simulator |
| `voicechat/docs/BUGS_FIXED.md` | Bugs found and fixed (symptom, cause, fix, how to check) |
| `voicechat/docs/CHECKS.md` | Acceptance and audit tools |
| `voicechat/docs/CONTRIBUTING.md` | How to send findings back |
| `voicechat/tools/` | Orchestration and data tools (appear as `scripts/` in a workspace) |
| `voicechat/configs/` | Recipe templates (offline / online, all fixes on), Gym services files, `render_recipe.py` |
| `voicechat/third_party/` | Megatron-Bridge / Megatron-LM local patches and fetch script (large trees are not in git) |
| `voicechat/setup_workspace.sh` | Creates the workspace layout the tools expect |

## Two RL settings

- **Offline (pivot)**: each row is an existing dialogue prefix; the policy generates only the next turn (one tool call or
  one message) and the Gym verifier compares it with the gold action (0/1). No user simulator; cheap and stable; good for
  fixing specific behaviours. Rows are selected by profiling the start policy and keeping prompts with 0 < pass < 1.
- **Online**: whole episodes are sampled during training by a Gym agent (tau2, conversational tool use, workplace
  assistant, ...) talking to a user simulator (local gpt-oss-120b nodes or a hub API); reward comes from the episode outcome.

Both use the same NeMo-RL / Gym / vLLM code, so the fixes in `BUGS_FIXED.md` apply to both. The offline path was
re-validated after the 2026-10-13 fixes; the online path has the same fixes but has not been re-run end to end since (see
`ONLINE_RL.md`).

## Workspace contract

The tools expect a workspace root `$NVC_RL_ROOT` with `src/pipeline/nemo_rl` (this repo), `src/nemo-rl -> pipeline/nemo_rl`,
`scripts -> voicechat/tools`, `config/`, `runtime/gym-v1` (images, Gym venvs), `models/sim` (simulator), and
`runs/ experiments/ data/ logs/ receipts/`. `voicechat/setup_workspace.sh <dir>` creates it (shared assets are symlinked
read-only) and fetches the Megatron-Bridge tree. Environment read by the tools:

| Variable | Used by |
|---|---|
| `NVC_RL_ROOT` | every tool (workspace root; no default, so nothing writes into someone else's tree) |
| `NVC_SLURM_ACCOUNT` | `submit_run.sh` (space-separated list; the account with the earliest `--test-only` start wins) |
| `NVC_SLURM_ACCOUNT_CPU` | CPU / interactive jobs (`interactive.sh`, `finish_chain.sbatch`) |
| `NVC_EVAL_ROOT` (optional) | `finish_chain.sbatch` launches bfcl_v3 + tau2 through this eval harness |

## Shared assets on cs-oci (read-only)

| Asset | Path |
|---|---|
| Training image (vLLM 0.27.1 + Megatron + Gym, 82 GB) | `/lustre/fsw/portfolios/llmservice/users/cchen1/NemotronVC_rl/runtime/gym-v1/NemotronVC-Gym34-vllm0271-v1.sqsh` |
| Code sandbox image | `/lustre/fsw/portfolios/llmservice/users/cchen1/NemotronVC_rl/runtime/gym-v1/sandbox-x86_64-v1.sqsh` |
| Extra Gym venvs (tau2, openai_model, ...) | `/lustre/fsw/portfolios/llmservice/users/cchen1/NemotronVC_rl/runtime/gym-v1/venvs-ext` |
| User simulator gpt-oss-120b (MXFP4) | `/lustre/fsw/portfolios/llmservice/users/cchen1/NemotronVC_rl/models/sim/gpt-oss-120b` |
| Example start policy: nano35-final (HF) | `/lustre/fsw/portfolios/llmservice/users/cchen1/NemotronVC_sft/checkpoints/nano35-final` |
| Upstream RL data library (Super3.5 v43, 33 routes) | `/lustre/fsw/portfolios/llmservice/users/cchen1/data/library/rl/merged_20260928_super35_v43/` |
| tau2-style pivot pools (tau_mms / CTU / telecom / AReaL writes) | `/lustre/fsw/portfolios/llmservice/users/cchen1/NemotronVC_rl/data/tau2-src/pools/` |
| Online RL data example (tau2 telecom train split, CTU simulation, workplace, ...) | `/lustre/fsw/portfolios/llmservice/users/cchen1/NemotronVC_rl/data/agentic-teacher-v1/` (Gym assets in `assets/tau2_data`) |
| gorilla snapshot for the BFCL multi-turn generator | `/lustre/fsw/portfolios/llmservice/users/cchen1/NemotronVC_rl/data/bfcl-src/` |
| Exact source tree incl. Megatron-Bridge | `/lustre/fs11/portfolios/llmservice/projects/llmservice_nemo_speechlm/users/cchen1/NemotronVC_rl/src/pipeline/nemo_rl` |
| Eval harness (reference; use your own copy) | `/lustre/fsw/portfolios/llmservice/users/cchen1/NemotronVC_eval` |

`/lustre/fsw/portfolios/llmservice/users/cchen1` is a symlink to `/lustre/fs11/portfolios/llmservice/projects/llmservice_nemo_speechlm/users/cchen1`.

## Relation to upstream NeMo-RL

- Baseline: the super-v3.5-posttraining line of NeMo-RL plus fork commits (baseline `9a50d85`); files changed on top of
  it are listed at the end of `voicechat/docs/BUGS_FIXED.md`.
- Gym (`3rdparty/Gym-workspace/Gym`, upstream `5dd5408` + local changes) is vendored as plain files; upstream test files
  over 3 MB are omitted (`voicechat/third_party/SKIPPED_LARGE_FILES.txt`).
- Megatron-Bridge and its nested Megatron-LM carry local commits and are too large for git; get them with
  `voicechat/third_party/fetch_third_party.sh` (copy of the exact tree, or upstream clone + patches).
- No git submodules (`.gitmodules` removed).
