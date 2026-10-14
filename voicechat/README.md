# nemo-rl-voicechat

本仓库是 NVIDIA NeMo-RL 的 fork，用于在 cs-oci（cs-oci-ord）集群上对 Nemotron-H 系列混合 Mamba/MoE 模型（Nemotron 3.5 Omni 30B-A3B 及其 SFT 后代）做 RL，重点是工具调用和 agentic 对话（tau2、BFCL）。训练框架是 NeMo-RL async GRPO + NeMo Gym，训练端 Megatron-Core，rollout 端 vLLM 0.27.1。

`voicechat/` 以外是 NeMo-RL 代码树，带有我们的修复。`voicechat/` 里是我们自己的工具链和文档：

| 路径 | 内容 |
|---|---|
| `voicechat/AGENTS.md` | **给 coding agent 的首读文件**：工作规则、流程、常见坑 |
| `voicechat/docs/OFFLINE_PIVOT_RL.md` | 离线（单步 pivot）RL 全流程：候选数据 → 清洗 → 画像 → bundle → smoke → 训练 → 续训 → 评测 |
| `voicechat/docs/ONLINE_RL.md` | 在线 RL：多轮 episode 由 Gym agent 与用户模拟器实时交互采样 |
| `voicechat/docs/BUGS_FIXED.md` | 发现并修过的 bug（现象、原因、修法、怎么自检） |
| `voicechat/docs/CHECKS.md` | 验收与审计工具：smoke 验收、dump 审计、advantage 测试、权重对比、Mamba/梯度探针 |
| `voicechat/docs/CLUSTER_CS_OCI.md` | cs-oci 集群规则：分区、配额、容器、交互调试、礼仪 |
| `voicechat/docs/CONTRIBUTING.md` | 新发现怎么回流到本仓库 |
| `voicechat/tools/` | 编排脚本：在工作区里以 `scripts/` 的名字出现，见下文 |
| `voicechat/configs/` | recipe 模板（离线 / 在线，已带全部修复）、Gym services、`render_recipe.py` |
| `voicechat/third_party/` | Megatron-Bridge / Megatron-LM 的本地 patch 与拉取脚本（大树不进 git） |
| `voicechat/setup_workspace.sh` | 一条命令建好工作区 |

## 两种 RL 场景

- **离线（offline pivot）**：每行数据是一段已有的对话历史，模型只生成下一步（一次工具调用或一句回复），由 Gym 验证器与 gold 动作比较给 0/1 奖励。不需要用户模拟器，便宜、稳定，适合定点修行为。数据靠"候选 → 起点模型画像 → 选 0 < pass < 1"来挑。
- **在线（online）**：整段 episode 在训练时实时采样，由 Gym agent（tau2、conversational tool use、workplace assistant 等）驱动多轮交互，用户一方由模拟器（本地 gpt-oss-120b 节点或 hub API）扮演，按任务结果给奖励。

两种场景共用同一套 NeMo-RL / Gym / vLLM 代码，`BUGS_FIXED.md` 里的修复对两者都生效。区别只在数据行、Gym agent 和资源配置（在线多出模拟器节点）。离线流程在 2026-10-13 修复后已重新验证；在线流程的修复是同一份代码和配置，但修复后还没有重新跑过完整训练，见 `ONLINE_RL.md`。

## 快速开始（cs-oci 登录节点）

```bash
# 1. 选一个自己的工作区（lustre 上，有几百 GB 空间），把仓库 clone 到 src/pipeline/nemo_rl
WS=/lustre/fsw/portfolios/<portfolio>/users/$USER/rl-ws
mkdir -p $WS/src/pipeline && cd $WS/src/pipeline
git clone ssh://git@gitlab-master.nvidia.com:12051/cchen1/nemo-rl-voicechat.git nemo_rl   # 或 https://gitlab-master.nvidia.com/cchen1/nemo-rl-voicechat.git

# 2. 建工作区（软链共享的只读镜像/venv/模拟器，拷 Megatron-Bridge 树，约几分钟）
bash $WS/src/pipeline/nemo_rl/voicechat/setup_workspace.sh $WS

# 3. 环境变量（写进 ~/.bashrc；sbatch 会把它们带进作业）
export NVC_RL_ROOT=$WS
export NVC_SLURM_ACCOUNT="<GPU 账户> [<备选账户>]"   # submit_run.sh 对每个账户 --test-only 估算，取最早开始的
export NVC_SLURM_ACCOUNT_CPU=<CPU/交互作业账户>
# 可选：export NVC_EVAL_ROOT=<你自己的 NemotronVC_eval 评测工作区>
```

然后按 `voicechat/docs/OFFLINE_PIVOT_RL.md` 或 `ONLINE_RL.md` 走。

## cs-oci 上可直接引用的共享资产（只读）

以下路径对所有用户可读，直接引用或软链，**不要往里写**：

| 资产 | 路径 |
|---|---|
| 训练容器（vLLM 0.27.1 + Megatron + Gym，82GB） | `/lustre/fsw/portfolios/llmservice/users/cchen1/NemotronVC_rl/runtime/gym-v1/NemotronVC-Gym34-vllm0271-v1.sqsh` |
| Code sandbox 容器 | `/lustre/fsw/portfolios/llmservice/users/cchen1/NemotronVC_rl/runtime/gym-v1/sandbox-x86_64-v1.sqsh` |
| Gym 额外 venv（tau2、openai_model 等） | `/lustre/fsw/portfolios/llmservice/users/cchen1/NemotronVC_rl/runtime/gym-v1/venvs-ext` |
| 用户模拟器 gpt-oss-120b（MXFP4） | `/lustre/fsw/portfolios/llmservice/users/cchen1/NemotronVC_rl/models/sim/gpt-oss-120b` |
| 起点模型示例：nano35-final（HF） | `/lustre/fsw/portfolios/llmservice/users/cchen1/NemotronVC_sft/checkpoints/nano35-final` |
| 上游 RL 数据库（Super3.5 v43，33 条 route） | `/lustre/fsw/portfolios/llmservice/users/cchen1/data/library/rl/merged_20260928_super35_v43/` |
| tau2 方向的 pivot 数据池（tau_mms / CTU / telecom / AReaL 写操作） | `/lustre/fsw/portfolios/llmservice/users/cchen1/NemotronVC_rl/data/tau2-src/pools/` |
| 在线 RL 数据示例（tau2 telecom 训练任务、CTU 模拟、workplace 等） | `/lustre/fsw/portfolios/llmservice/users/cchen1/NemotronVC_rl/data/agentic-teacher-v1/`（含 `assets/tau2_data`） |
| BFCL 多轮生成器用的 gorilla 快照 | `/lustre/fsw/portfolios/llmservice/users/cchen1/NemotronVC_rl/data/bfcl-src/` |
| 完整原样源码树（含 Megatron-Bridge） | `/lustre/fs11/portfolios/llmservice/projects/llmservice_nemo_speechlm/users/cchen1/NemotronVC_rl/src/pipeline/nemo_rl` |
| 评测 harness（只读参考，自己的评测请复制一份） | `/lustre/fsw/portfolios/llmservice/users/cchen1/NemotronVC_eval` |

`/lustre/fsw/portfolios/llmservice/users/cchen1` 是 `/lustre/fs11/portfolios/llmservice/projects/llmservice_nemo_speechlm/users/cchen1` 的符号链接，两种写法指向同一处。

## 和上游 NeMo-RL 的关系

- 基线：上游 NeMo-RL 的 super-v3.5-posttraining 线，加上若干 fork 内提交（基线 commit `9a50d85`，见 `voicechat/docs/BUGS_FIXED.md` 末尾的文件清单）。
- 三方依赖：
  - Gym（`3rdparty/Gym-workspace/Gym`，上游 `5dd5408` 加本地修改）直接放在仓库里；超过 3MB 的上游测试数据没有放，清单在 `voicechat/third_party/SKIPPED_LARGE_FILES.txt`。
  - Megatron-Bridge 和它内嵌的 Megatron-LM 带本地提交，太大不进 git，由 `voicechat/third_party/fetch_third_party.sh` 从集群拷贝原样树，或从上游 clone 后打 patch。
- `.gitmodules` 已移除：本仓库不使用 git submodule。
