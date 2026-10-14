# cs-oci（cs-oci-ord）集群要点

以下事实核实于 2026-09 至 10 月。集群配置会变，动手前用文中的命令复核。

## 基本约定

- 登录节点 `cs-oci-ord-login-0{1,2,3}.nvidia.com`，默认 shell 可能是 csh，脚本请用 `bash -s` heredoc 执行。登录节点时区是 PDT，脚本里统一 `export TZ=UTC`。
- **每条 sbatch / srun 都显式写 `--account` 和 `--partition`**：
  - 用户默认账户往往 fairshare 很差；
  - 默认分区 `batch` 在这里不存在。
  - 用 `sshare -U -l` 看自己的账户和 fairshare。用 `sbatch --test-only ...` 估算开始时间（不提交），估计通常偏悲观，因为它不模拟 backfill。
- 登录节点只做轻量的事：读文本、grep、提交和查询作业。非交互的 `srun` 加 `</dev/null`。
- 不要全局 `ray stop`、`pkill`、`killall`；只处理自己作业里、自己起的进程。

## 分区与配额

| 用途 | 分区 | 限制 |
|---|---|---|
| 多节点 GPU 训练 | `polar4` / `polar3` | 4 小时，每用户最多 128 节点，A100-80GB ×8/节点，本地盘 `/raid/scratch` |
| 单节点 GPU 交互 | `interactive_singlenode` | 1 节点、4 小时，每用户同时只能 1 个作业，通常几分钟内开始 |
| CPU 交互 | `cpu_interactive` | 24 小时，优先级最高，**每用户合计 48 核、88G 内存、2 个作业** |
| 短 CPU 任务 | `cpu_short` | 4 小时；可能要排队几个小时 |
| 长 CPU 任务 | `cpu` / `cpu_long` | 24 小时 / 7 天 |
| 低优先级 | `backfill_*` | 会被抢占，只放能重启的活 |

- `polar*` / `interactive*` 上必须申请 GPU（`--gres=gpu:8`），哪怕只是 `sleep`，否则 job_submit 会拒绝。
- 训练的 sbatch 必须带 `--no-requeue`：集群的 PreemptMode 是 REQUEUE，不带的话 frozen run 可能被静默重跑。
- 不要用逗号写多个分区（`--partition=polar4,polar3`），历史上这类作业基本起不来。
- **空闲 30 分钟会被回收**（`CANCELLED by 146504`）：交互分配用完立刻释放；训练的 step 0 必须在 30 分钟内完成。

## 容器（Pyxis / Enroot）

- 只能通过 `srun` 起容器（`#SBATCH --container-image` 无效）。同名容器在作业内共享。
- 82GB 的训练镜像首次挂载：CPU 节点约 6–10 分钟，GPU 节点约 2 分钟；同一个 allocation 里再次 attach 只要几秒。
- **同名容器的首次导入不能并发**：两个 srun 同时冷导入会把容器目录弄坏，之后所有 attach 都会失败；只能换一个容器名。
- GPU step 加 `--cpu-bind=none`，否则容器里 `nproc` 只有 1。
- 容器里 `source /workspace/debug/scripts/runtime_env.sh` 会打开 `set -e`，交互调试时记得之后 `set +e`。
- 镜像里的 venv：`/opt/nemotronvc/venvs/{driver,mcore,vllm027,vllm,nemo-gym}`。driver 跑主进程和各种工具；mcore 跑训练 worker；vllm027 是 vLLM 0.27.1。
- 容器内 `$USER` 是 root。Ray 的临时目录要用短路径（`/tmp/x-$JOB`），否则 AF_UNIX socket 路径会超过 107 字节。

## 交互调试：`scripts/interactive.sh`

```bash
J=$(bash scripts/interactive.sh alloc cpu </dev/null | tail -1)   # 也可以 alloc gpu、alloc multi N
bash scripts/interactive.sh shell $J                                 # 进容器的交互 shell
bash scripts/interactive.sh exec $J 'source /workspace/debug/scripts/runtime_env.sh; set +e; <命令>'
bash scripts/interactive.sh release $J                               # 删除节点本地 scratch 并取消这个作业
```

- 原理：用 `sbatch --wrap sleep` 占住 allocation，再用 `srun --overlap` 反复进容器，断线也不会丢。
- `exec` 起的每条命令都是一个 Slurm step。对 step 里的 bash 执行 `kill -9`，Slurm 会回收这个 step 里的所有进程。
- 用 `pgrep -f` 时，模式写成 `[b]racket` 形式，否则会匹配到自己。

## 其他坑

- Slurm 显示 `COMPLETED` 只说明作业退出了，不代表训练到了目标步数。要看 run 目录里的 `END_CODE`、`optimizer-progress.json` 和 checkpoint 状态。`END_CODE=74` 表示 idle guard 主动停机，Slurm 会显示 FAILED。
- NCCL、UCX 的环境变量由节点和镜像注入，不要覆盖。
- Gym 容器里执行 `gym env status` 或 `gym --version`，要加 `++allow_openai_version_skew=true`。
- 采样时 `max_output_tokens` 必须小于 `max_model_len − prompt 长度`，否则 vLLM 返回 400，rollout 全部是空输出，看起来却像跑完了。
