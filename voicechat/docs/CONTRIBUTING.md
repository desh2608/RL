# 新发现怎么回流

本仓库由 cchen1 维护（gitlab-master: `cchen1/nemo-rl-voicechat`）。欢迎合作者把修复和发现提回来。

1. 从 `main` 拉分支，分支名 `<你的名字>/<主题>`。
2. 代码改动遵守 `voicechat/AGENTS.md` 的"改代码的规矩"：
   - 默认值保持旧行为，新行为用配置项打开；
   - 写一个能在 CPU 上跑的测试，放在 `tests/unit/...` 或 `voicechat/tools/audit/`；
   - 在 smoke 上用 `smoke_check.py` 验证过。
3. 在 `voicechat/docs/BUGS_FIXED.md` 里加一条，编号顺延，包括：
   - **现象**：看到的指标或日志，写具体数字；
   - **原因**：代码位置；
   - **修法**：改了哪个文件，或者用哪个配置项打开；
   - **怎么自检**：用哪个工具，期望看到什么值；
   - 适用于离线、在线还是两者。
4. 只是发现、还没修的问题，同样写进 BUGS_FIXED，标"风险"或"未修"，并附验证方法。
5. 提 Merge Request，描述里写清楚上面几点和实测结果（run 路径可以写 cs-oci 上的绝对路径，别人能读到）。

**不要提交**：
- checkpoint、数据、run 目录、日志；
- 大于 3MB 的文件；
- 任何密钥。

数据和实验产物放在 cs-oci 上，在 MR 里给路径。

**三方依赖的改动**（Gym 可以直接改，它已经在仓库里）：Megatron-Bridge 和 Megatron-LM 的改动，在对应树里 `git format-patch`，追加到 `voicechat/third_party/*-patches/`，同时更新 `fetch_third_party.sh` 里的基线说明。
