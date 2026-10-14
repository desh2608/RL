# 在线 RL（多轮 episode 实时采样）

在线 RL 和离线 RL 用的是同一套代码：NeMo-RL async GRPO + Gym + vLLM HTTP server。区别如下。

| | 离线 pivot | 在线 |
|---|---|---|
| 数据行 | 历史 + gold 下一步 | 任务描述（例如 tau2 任务 id 与用户场景、CTU 场景、workplace 任务） |
| 谁驱动多轮 | 不需要，只生成一轮 | Gym agent：`tau2`、`conversational_tool_use_agent`、`workplace_assistant_simple_agent` 等 |
| 用户一方 | 无 | 用户模拟器：本地 gpt-oss-120b 节点，或 hub API |
| 奖励 | 下一步与 gold 比较（0/1） | 任务结束后按结果判分（tau2 evaluator：数据库状态和必需动作；CTU 由 judge 判） |
| 训练的 token | 一轮 assistant 输出 | episode 里所有 assistant 轮；用户和工具的内容被掩码 |
| 资源 | 14 节点 | Ray 节点 + `--sim-nodes N` 个模拟器节点（不进 Ray，跑独立 vLLM） |

## 修复是否适用

`docs/BUGS_FIXED.md` 里的代码级和配置级修复，在线场景都生效：
- B3：system 按纯文本渲染；
- B5：advantage 修复；
- B8–B10：fp32 lm head、fp32 SSM 缓存、TIS 下限；
- B1、B2：Mamba 两项修复；
- B11–B16：路由偏置、引擎亲和、引擎宕机切换、续训等基础设施修复。

`voicechat/configs/online-agentic-rl.template.json` 已经带上了全部这些修复。

数据级的修复（B4 seed、B6 `parallel_tool_calls`、B7 提示形态）在在线场景里表现不同：
- 在线行同样**不能带采样 seed**，否则同一任务的 K 条 episode 会高度相同；
- 历史形态由 agent 在采样时实时生成，没有离线数据的残留问题，但要确认 agent 回传历史时对 reasoning 的处理与评测一致。

**验证状态（如实说明）**：
- 在线链路在 2026-10-01 到 10-03 跑通过：smoke 和训练都能正常起来。当时修掉了引擎亲和（B14）和引擎宕机切换（B15）。
- 10-13 那一批修复之后，在线链路**还没有重新跑过完整训练**。第一次用请先做 smoke，并跑 `smoke_check.py` 的四项检查。
- 历史问题：P256、每条 tau2 episode 约 10k token、16 轮时，单个模拟器节点约 14 req/s，step 0 超过 1 小时，会被集群的 30 分钟空闲回收杀掉。所以要先缩小 batch 或增加模拟器节点，保证 step 0 在 30 分钟内完成。

## 步骤

1. **数据**：示例 `/lustre/fsw/portfolios/llmservice/users/cchen1/NemotronVC_rl/data/agentic-teacher-v1/`（只读）。
   - 内容：tau2 telecom 非评测切分 6,154 行、CTU 12,821 行、workplace 2,564 行，另外混有单步 route；
   - Gym 资产（tau2 数据库）在 `assets/tau2_data`；
   - 自己做 bundle 时保持 gym-data-v1 格式（train.jsonl、val.jsonl、MANIFEST.json），不要带 seed。
2. **渲染 recipe**：

```bash
python3 $P/src/pipeline/nemo_rl/voicechat/configs/render_recipe.py \
  --template $P/config/online-agentic-rl.template.json --out $P/config/<run>.json \
  --student <起点 HF> --bundle <bundle 目录> --lineage <lineage> \
  --services $P/config/gym-services-online-localsim.yaml \
  --gym-asset-dir /lustre/fsw/portfolios/llmservice/users/cchen1/NemotronVC_rl/data/agentic-teacher-v1/assets
```

   模板默认：
   - 10 个 Ray 节点（8 rollout + 2 learner），P256 × K8；
   - `max_new_tokens` 8192（多轮 episode 需要更长）；
   - 其余修复项与离线模板相同。

3. **smoke / 训练**：在 `make_run.py` 的参数里加上模拟器节点：

```bash
... make_run.py --kind smoke --project $P --recipe $P/config/<run>.json --services $P/config/gym-services-online-localsim.yaml \
    --bundle <bundle> --sim-nodes 1 --sim-model-path $P/models/sim/gpt-oss-120b
```

   - `gpu_job.sh` 把分配到的最后 N 台机器用作模拟器（`scripts/sim_role.sh`，vLLM 0.27.1、TP1×DP8，约 40 req/s/节点），并把地址写进 `NVC_SIM_BASE_URL`（逗号分隔）。Gym 的 `openai_model` 按最少在途请求分发。
   - 并发：`sim_gptoss.max_concurrent_requests` 按 384 × 节点数设置。
   - 用 hub 模拟器：services 换成 `gym-services-online-hub.yaml`，不加 `--sim-nodes`，把自己的 key 放在 `$P/.secrets/nvidia-api-key`（权限 600）。key 只通过环境变量注入，不能写进配置。

4. **开跑后 20 分钟内**，用 `srun --jobid J --overlap -w NODE nvidia-smi` 看每个生成节点是否都在忙。全压在一个引擎上，说明引擎亲和的 key 有问题（B14）。

## 注意事项

- **训练 policy 不能兼任用户模拟器或 judge**：会角色错乱，还会出现假通过。
- **tau2 的 Gym venv**：在共享的 `runtime/gym-v1/venvs-ext` 里，版本钉在 tau2 commit `60c2a0db`，与评测 harness 一致。需要重建时用 `scripts/build_gym_venvs_ext.sh`，在 CPU 节点上跑，写的是你自己的 venvs-ext。
- **长度监控**：`checkpointing.length_monitor` 按 agent 设截断比例阈值。在线 episode 长，阈值可能要放宽，否则会提前停机。
