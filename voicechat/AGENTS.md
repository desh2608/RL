# 给 coding agent：在 nemo-rl-voicechat 上工作

先读 `voicechat/README.md`，再按任务读 `docs/OFFLINE_PIVOT_RL.md` 或 `docs/ONLINE_RL.md`。动手前读一遍 `docs/BUGS_FIXED.md`，那里每条都是真踩过的坑。

## 工作区约定

- 工作区根目录是 `$NVC_RL_ROOT`，由 `voicechat/setup_workspace.sh` 建立。脚本里所有 `$P` / `$R` 都是它。
- 本仓库的 `voicechat/tools/` 在工作区里叫 `scripts/`，文档里的 `scripts/xxx` 都指它。
- 每次训练都由 `scripts/make_run.py` 生成一个不可变的 `runs/<kind>-v1-<UTC>Z/` 目录：配置、数据、源码 tar、脚本拷贝都在里面。**不要改已经提交过的 run 目录**；要改就重新生成一个。
- 训练输出在 `experiments/<lineage>/{checkpoints,hf,monitor,eval}`。同一个 lineage 同时只能有一个写者，由 submit guard 加锁。
- `make_run.py --resume` 会用**当前**源码重新打包，并要求它与 smoke 的源码一致。训练没结束、续训还没生成时，不要改 `src/pipeline/nemo_rl` 里的代码，否则续训会被拒绝。

## 集群礼仪（cs-oci，详见 docs/CLUSTER_CS_OCI.md）

- 登录节点只做轻量的事：读文本、grep、sbatch / squeue / sacct。Python 数据处理、哈希大文件、起容器，一律放进 allocation（CPU 用 `cpu_interactive` 或 `cpu_short`）。
- 非交互的 `srun` / `sbatch` 都加 `</dev/null`；每条 sbatch / srun 都显式写 `--account` 和 `--partition`。
- 不要 `ray stop`、`pkill`、`killall`；只停自己作业里、自己起的进程。不要碰别人的作业。
- 集群会取消空闲超过 30 分钟的作业：交互分配用完立刻 `scripts/interactive.sh release JOB`。训练的 step 0 也必须在 30 分钟内完成。
- `cpu_interactive` 每用户上限是 48 核、88G、2 个作业。一个作业如果要在同一分区再提交子作业，两者内存加起来不能超过 88G，否则子作业永远排不上，父作业原地等死。
- 密钥（例如 hub 模拟器用的 NVIDIA API key）只放 `$NVC_RL_ROOT/.secrets/`，权限 600，通过环境变量注入，不写进任何配置文件或日志。

## 标准流程（离线）

1. 准备候选行，可参考 `scripts/pivot_data/example_build_candidates.py`。
2. `scripts/pivot_data/clean_candidates.py`：清洗成评测形态。
3. `scripts/profile_passrate/profile_candidates.sbatch`：起点模型 K=8 画像。
4. `scripts/pivot_data/build_bundle.py`：只留 0 < pass < 1 的 prompt，生成 bundle。
5. `voicechat/configs/render_recipe.py`：把模板渲染成具体 recipe。
6. 在交互 CPU 节点上运行 `make_run.py --kind smoke`，然后在登录节点运行 `submit_run.sh`。
7. `scripts/audit/smoke_check.py` 验收，结果是 ACCEPT 才能继续。
8. `make_run.py --kind train --smoke-run <smoke>`，然后 `submit_run.sh`。
9. `scripts/finish_chain.sbatch`：自动续训到目标步数，生成 hf-aligned 视图，可选地提交评测。

每一步的命令见 `docs/OFFLINE_PIVOT_RL.md`。

## 改代码的规矩

- 训练数值、loss、advantage、数据渲染上的改动，必须同时满足：
  - 有一个能在 CPU 上跑的测试（参考 `scripts/audit/test_adv_std_scope.py`）；
  - 默认值保持旧行为，新行为用配置项打开；
  - 在 smoke 上用 `smoke_check.py` 和 dump 审计验证过。
- 新发现的问题写进 `docs/BUGS_FIXED.md`，格式见 `docs/CONTRIBUTING.md`，再提 MR。
- 看训练是否健康，不要只看 reward 曲线，至少看：
  - `train/token_mult_prob_error`：修复后约 1.017，持续高于 1.03 说明有问题；
  - `train/gen_kl_error`：修复后约 0.0007；
  - driver.log 里的 `Advantages stats`：修复后在 ±4 左右；
  - 零方差丢弃的组数；
  - 同组样本的多样性（`dump_audit.py`）。
