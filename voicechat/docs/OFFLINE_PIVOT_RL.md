# 离线（单步 pivot）RL 全流程

**适用场景**：已经有一批对话轨迹（教师采样、公开数据、评测同分布的合成任务），想针对某个具体行为做修正，比如"该调用时却只回复文本""确认后不执行写操作""无关请求误调用工具"。

**原理**：每行数据给出到某一轮为止的历史，模型只生成下一轮；Gym 验证器 `single_step_tool_use_with_argument_comparison` 把它和 gold 动作比较，按工具名、参数键、参数值判对错，给 0/1 奖励。message 类 gold 表示"这一轮应该回复文本、不调用工具"，任何文本回复都算对。

下面所有命令都在 cs-oci 登录节点上执行，并假设已经按 README 设好了 `NVC_RL_ROOT`、`NVC_SLURM_ACCOUNT` 和 `NVC_SLURM_ACCOUNT_CPU`。`P=$NVC_RL_ROOT`。

## 0. 起点模型

- RL 起点是一个 HF 目录，同时用作 tokenizer 和 rollout 模型。
- NeMo-RL 或 Megatron-Bridge 导出的 HF 需要先做一份 **hf-aligned 视图**（`scripts/prepare_eval_hf_aligned.sbatch`），恢复 tokenizer 和 generation 元数据，否则 chat template 与评测侧不一致。
- 直接用评测通过的 HF 目录作起点最稳妥。

## 1. 候选行（route 样本）

**格式**是 profile_passrate 的 samples 布局：
- `SAMPLES/routes.json`：每条 route 一个条目，包括 `route_key`、`agent`、`sample_path`、`n_prompts` 等；
- `SAMPLES/<route_key>/sample.jsonl`：每行一个 Gym 行，含 `responses_create_params.input`（历史）、`tools`、`expected_action`（gold）、`agent_ref`。

**示例**：`scripts/pivot_data/example_build_candidates.py` 从共享数据池按"要修的行为"分出 11 条 route：
- BFCL 多轮：该调用、多调用、该停；
- 无关请求：hard no-call、near-miss；
- 单步参数；
- tau2 写操作；
- telecom、tau_mms、CTU。

它是一份数据配方的例子，换成自己的数据池时可以照着改。

**数据池**（只读）：
- 上游 RL 数据库：`/lustre/fsw/portfolios/llmservice/users/cchen1/data/library/rl/merged_20260928_super35_v43/`
- tau2 方向的 pivot 数据池：`.../NemotronVC_rl/data/tau2-src/pools/`
- 索引工具：`scripts/pivot_data/{index_pivot_pools,summarize_pivot_index,materialize_pivot_rows}.py`

**难度账本**：`scripts/pivot_data/build_difficulty_ledger.py` 从训练日志里被丢弃的零方差组反推"全对"的 prompt，下一轮数据可以排除它们。

## 2. 清洗成评测形态（必须做）

```bash
sbatch --account=$NVC_SLURM_ACCOUNT_CPU --partition=cpu_interactive --time=00:30:00 --mem=16G --cpus-per-task=2 \
  --output=$P/logs/clean-%j.log --wrap "cd $P && python3 scripts/pivot_data/clean_candidates.py --src SAMPLES --out CLEAN" </dev/null
```

清洗做的事（对应 BUGS_FIXED 的 B4、B6、B7）：
- 去掉历史里的 reasoning 项；
- 去掉 tau2 采样残留的 `ENVIRONMENT REMINDER`；
- 去掉 gold 写调用之前同一轮的 assistant 文本，否则模型会被要求在自己说完话之后再开一轮；
- 丢掉以"未返回结果的调用"结尾的行；
- `parallel_tool_calls` 一律设为 true；
- 不留采样 seed；
- 去掉重复 prompt。

`CLEAN/CLEAN_MANIFEST.json` 记录每条 route 改了多少行。

## 3. 起点模型画像（K=8）

```bash
export NVC_MODEL_PATH=<起点 HF 目录> NVC_MODEL_LABEL=<名字>
sbatch --account=<GPU 账户> --partition=polar4 --time=04:00:00 --output=$P/analysis/<name>/g1-%j.log \
  scripts/profile_passrate/profile_candidates.sbatch $P/analysis/<name>/g1 CLEAN route_a,route_b 8 </dev/null
```

- **资源**：一个 8×A100 节点，原生 `vllm serve`（TP1、DP8、EP8），Gym 用和训练相同的验证器（`scripts/profile_passrate/gym_profile_pivot.yaml`）。
- **耗时**：约 6–8k 个 prompt 需要 2 小时，可以按 route 分几组并行提交。
- **自检**：启动时写出 `RENDER_CHECK.json`，确认 system 消息按纯文本渲染（BUGS_FIXED B3），不通过就直接退出。
- **画像条件必须和训练 rollout 一致**：温度 1.0、content format 为 string、fp32 SSM 缓存、`max_output_tokens`。否则选出来的"有对有错"在训练时可能变成全对。
- **输出**：每组的 `per_prompt.jsonl`（每个 prompt 的 pass rate）、`route_summary.json`、`SUMMARY.md`。

## 4. 生成 bundle

```bash
sbatch --account=$NVC_SLURM_ACCOUNT_CPU --partition=cpu_interactive --time=00:40:00 --mem=32G --cpus-per-task=2 \
  --output=$P/logs/bundle-%j.log --wrap "cd $P && python3 scripts/pivot_data/build_bundle.py --profile analysis/<name> \
  --clean-manifest CLEAN/CLEAN_MANIFEST.json --out data/<bundle> --bundle-name <bundle> --start-policy <起点 HF> \
  --cap route_a=1200 --cap route_b=400" </dev/null
```

- **选行规则**：只保留 0 < pass < 1 的 prompt。全对或全错的组没有 GRPO 信号，训练时会被直接丢弃。
- **逐行断言**：不带 seed、`parallel_tool_calls` 为 true、没有历史 reasoning、没有 reminder、不以 assistant 结尾。
- **复用旧画像**：一条 route 可以用 `--reuse OLD:route` 复用以前的画像，前提是清洗没有改动它的任何行，且行里没有 system 消息。
- **步数估算**：步数约为 `rows / num_prompts_per_step`。异步 GRPO 会循环使用数据（`max_num_epochs`）。

## 5. 渲染 recipe

```bash
python3 $P/src/pipeline/nemo_rl/voicechat/configs/render_recipe.py \
  --template $P/config/offline-pivot-rl.template.json --out $P/config/<run>.json \
  --student <起点 HF> --bundle $P/data/<bundle> --lineage <lineage 名> \
  --services $P/config/gym-services-offline-pivot.yaml \
  [--set grpo.max_num_steps=20 --set policy.megatron_cfg.optimizer.lr=2e-6 ...]
```

模板默认值（已包含所有修复）如下。

| 项 | 取值 |
|---|---|
| 资源 | 14 节点：10 rollout + 4 learner |
| 并行 | 训练 TP4、CP2、EP8，sequence packing；rollout 每节点 TP1×DP8×EP8 |
| batch | P512 × K16，GBS 8192 |
| 学习率 | lr 2e-6，cosine 到 2e-7，warmup 5，20 步 |
| advantage | 留一 baseline + 全组标准差（基于长度惩罚前的奖励），裁剪 ±5 |
| 损失 | `force_on_policy_ratio`；TIS 截断 [0, 5]；无参考 KL |
| 长度 | 每轮 `max_new_tokens` 2048，与 BFCL 评测预算对齐；长度惩罚系数 0.2 |
| 数值 | fp32 lm head（两端）、vLLM SSM 缓存 fp32、`MAMBA_DETERMINISTIC=1`、Mamba 训练 state fp32、system 按纯文本渲染 |

**改 batch 时**：`policy.train_global_batch_size` 和 `generation_batch_size` 都必须等于 P×K，`make_run` 会检查。

**改学习率或步数时**：同时改 `grpo.max_num_steps`、`policy.megatron_cfg.train_iters`、`scheduler.lr_decay_iters`，三者保持一致。

## 6. smoke（3 节点，3 步，严格从第 3 步续到第 4 步）

```bash
J=$(bash scripts/interactive.sh alloc cpu </dev/null | tail -1)
bash scripts/interactive.sh exec $J "export HF_MODULES_CACHE=/tmp/hfmod NVC_INPUT_MODE=gym; source /workspace/debug/scripts/runtime_env.sh; \
  /opt/nemotronvc/venvs/driver/bin/python /workspace/debug/scripts/make_run.py --kind smoke --project $P \
  --recipe $P/config/<run>.json --services $P/config/gym-services-offline-pivot.yaml --bundle $P/data/<bundle> \
  [--new-student-reason '<为什么换起点模型>']" </dev/null | grep -E "MAKE_RUN_OK|Error|require|Traceback"
bash scripts/interactive.sh release $J </dev/null
bash scripts/submit_run.sh $P/runs/smoke-v1-<UTC>Z </dev/null
```

- `make_run` 必须加 `--project $P`：脚本目录是软链，默认推导出的项目路径不对。
- 交互节点首次导入 82GB 镜像约 10 分钟。

smoke 结束后，确认 `END_CODE` 为 0、`SMOKE_PASS.json` 存在，然后跑验收（见 CHECKS.md）：

```bash
python3 scripts/audit/smoke_check.py $P/runs/smoke-v1-<UTC>Z --tokenizer <起点 HF>
```

四项都过才会输出 ACCEPT：
- system 不是列表字符串；
- 短回复的组内不同输出数 ≥ 10/16；
- advantage 在 ±5.5 以内；
- 失配和 KL 在合理范围（打印出来，与参考 run 对比）。

## 7. 正式训练与收尾

```bash
# 同样在交互 CPU 节点里构建
... make_run.py --kind train --project $P --recipe ... --services ... --bundle ... --smoke-run $P/runs/smoke-v1-<UTC>Z ...
bash scripts/submit_run.sh $P/runs/train-v1-<UTC>Z </dev/null       # 打印 [SUBMITTED] job=...
TJ=$(cat $P/runs/train-v1-<UTC>Z/JOB_ID)
sbatch --account=$NVC_SLURM_ACCOUNT_CPU --partition=cpu_interactive --time=20:00:00 --output=$P/logs/finish-chain-%j.out \
  scripts/finish_chain.sbatch $TJ <lineage> <目标步数> $P/config/<run>.json $P/config/gym-services-offline-pivot.yaml \
  $P/data/<bundle> $P/runs/smoke-v1-<UTC>Z <起点 HF> "<new-student-reason 或空串>" </dev/null
```

**训练分段**：
- 每段受 4 小时分区上限约束；
- 训练脚本在开始后 3 小时 40 分时保存 MCore 并退出；
- 每 5 步保存一次 ft checkpoint，每 10 步导出一次 HF。

**finish_chain 做的事**：
1. 等训练结束；
2. 如果还没到目标步数，用同一配置构建 `--resume` 并提交，最多两轮；
3. 对 `hf/step_N` 做 hf-aligned 视图；
4. 设了 `NVC_EVAL_ROOT` 时，提交 bfcl_v3 和 tau2。

日志在 `$P/logs/finish-chain-<job>.log`。

**看训练**：
- `runs/<run>/driver.log`：`Step N/…`、`Advantages stats`、`Dropped zero-variance prompt group`；
- `runs/<run>/train/exp_001/tensorboard`：可以用 `scripts/audit/tb_scalars.py` 在登录节点读，不依赖 tensorboard；
- `runs/<run>/train/exp_001/closed_trajectories/stepN.jsonl`：每步 256 条完整样本，含 token、掩码、advantage、vLLM logprob。

## 8. 评测

评测走 NemotronVC_eval harness（`./bin/unified-external-eval run --site cs ...`），只评 bfcl_v3 和 tau2 时，大约需要 1–3 个 GPU 节点、数小时。

harness 会往自己的工作区写结果，所以请复制一份到你自己的目录，设 `NVC_EVAL_ROOT`。参考路径：`/lustre/fsw/portfolios/llmservice/users/cchen1/NemotronVC_eval`（只读）。

评测必须用 hf-aligned 视图，tokenizer 用 harness 里的 `eval/text/tokenizers/student-kevin-nvc4-sft-18k-fix-mistral`。
