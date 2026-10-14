# 三方依赖

| 依赖 | 在仓库里？ | 基线 | 本地改动 |
|---|---|---|---|
| Gym（`3rdparty/Gym-workspace/Gym`） | 是（源码，不含 `.git`；超过 3MB 的上游测试数据见 `SKIPPED_LARGE_FILES.txt`） | NVIDIA-NeMo/Gym `5dd5408`（super-v3.5-posttraining） | 直接体现在仓库代码里，见 `../docs/BUGS_FIXED.md` 末尾的文件清单 |
| Megatron-Bridge（`3rdparty/Megatron-Bridge-workspace/Megatron-Bridge`） | 否（不含 .git 约 170MB，含内嵌的 Megatron-LM） | NVIDIA-NeMo/Megatron-Bridge `3961f399` | `megatron-bridge-patches/`，4 个：Omni 打包布局、两路 CP 文本打包、HDO 恢复校验、钉住 MCore 修复 |
| Megatron-LM（Megatron-Bridge 的 `3rdparty/Megatron-LM`） | 否 | `f2f0f7bfd` | `megatron-lm-patches/`，3 个：HDO master 参数恢复、FP32 CPU 副本恢复、保留已恢复的 step |
| Automodel | 不使用 | — | — |

拉取方法：

```bash
bash voicechat/third_party/fetch_third_party.sh               # 默认：从 cs-oci 共享工作区拷贝训练实际用的原样树（不含 .git）
bash voicechat/third_party/fetch_third_party.sh --upstream    # 从上游 clone 基线 commit，再打 patch（需要网络，且基线 commit 已公开）
bash voicechat/third_party/fetch_third_party.sh --large-files # 顺带补齐 Gym 里因体积跳过的文件
```

`setup_workspace.sh` 会自动执行默认方式。容器里的 `scripts/runtime_env.sh` 把 `/workspace/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src` 和其中的 `3rdparty/Megatron-LM` 加进 PYTHONPATH，所以这棵树必须存在。`.gitignore` 已忽略这两个目录。
