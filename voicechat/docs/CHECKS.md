# 验收与审计工具（`scripts/audit/`）

这些工具都只依赖 run 目录里已有的产物，或者是能在 CPU 上跑的测试。除特别说明的以外，都只用标准库，可以在登录节点上读小文件。

## smoke / 训练验收：`smoke_check.py`

```bash
python3 scripts/audit/smoke_check.py RUN_DIR --tokenizer <policy HF> [--ref OTHER_RUN_DIR]
```

| 检查 | 通过条件 | 对应 bug |
|---|---|---|
| 1. system 渲染 | 解码 closed_trajectories，system 段没有一条是列表字符串 | B3 |
| 2. 组内多样性 | 平均长度不到 200 token 的组，不同输出数的均值 ≥ 10/16 | B4 |
| 3. advantage | driver.log 里 `Advantages stats` 的 \|min\|、\|max\| ≤ 5.5 | B5 |
| 4. 失配 | 打印 `token_mult_prob_error`、`gen_kl_error`、reward，与参考 run 对比 | B8、B9 |

修复后的参考值（Nemotron 3.5 30B-A3B 的 SFT 起点，P512×K16 smoke）：失配 1.015–1.018，gen_kl 0.0006–0.0008，advantage 恒为 ±4。

## 每步样本审计：`dump_audit.py`

```bash
python3 scripts/audit/dump_audit.py RUN/train/exp_001/closed_trajectories/step5.jsonl
```

检查内容：
- loss 掩码是否是连续一段、位于序列末尾；
- prompt 部分的生成 logprob 是否为 0；
- 各 agent 的奖励取值分布，以及 shaped 和 unshaped 奖励的差异；
- advantage 在样本内是否为常数。

注意：在 `force_on_policy_ratio` 下，dump 里的 `prev_logprobs` 全为 0（没有计算），所以它的"失配"统计不可用。

## advantage 修复测试：`test_adv_std_scope.py`

要在训练容器的 driver venv 里跑，需要 torch 和 pydantic：

```bash
PYTHONPATH=/workspace/nemo-rl /opt/nemotronvc/venvs/driver/bin/python scripts/audit/test_adv_std_scope.py \
  [--legacy-module <旧版 advantage_estimator.py>] [closed_trajectories/stepN.jsonl ...]
```

覆盖三件事：
- 默认值与旧代码逐位一致（要求同时提供旧模块）；
- 新配置下单个失败和单个成功分别是 -4 和 +4，全错组仍为 0；
- 在真实 dump 上用旧逻辑重算 advantage，结果应与记录值一致，并打印前 1% 样本所占的梯度权重。

## 权重对比：`weight_diff_hf.py`

在 CPU 节点上跑，读两份 HF checkpoint，I/O 约为 checkpoint 大小的 2 倍：

```bash
... python scripts/audit/weight_diff_hf.py --a <起点 HF> --b <RL 第 N 步 HF> --out REPORT.json
```

按类别输出：视觉、音频、路由偏置、路由权重、MoE 专家、Mamba、attention、norm、embedding、lm_head。每类给出被改动元素的比例、ulp 距离直方图和相对 L2。

用途：
- 冻结部分必须逐位不变；
- 没有异常大的改动（排查导出或转换错误）；
- 看 RL 实际改了多少：lr 2e-6、20 步，约 6% 的 bf16 元素变了 1–2 个 ulp。

## Mamba 确定性探针：`mamba_pin/`

`det_probe.sbatch` 在单节点 8 卡上，每张卡跑一次 `tune_ssd.py --probe`：前向加反向，记录选中的 kernel 配置和输出哈希，用来比较各卡结果是否逐位一致。

只用 `--probe` 模式。`--pin` 和调优模式依赖的固定配置函数已从 `mamba_numerics.py` 移除（见 B2 的注意事项）。

## 梯度同步探针：`gradsync_check/`

通过 torchrun 调用真实 worker 的 `train()`，在同一个 batch 上比较 DP4 与单副本 DP1 的逐参数梯度。

- 资源：1+4 个节点，约 10 分钟；
- HF 导入缓存每种布局约 62GB，用完要删；
- 换并行布局、升级 Megatron-Bridge 之后要重跑；
- 也可以改成比较 `grad_reduce_in_fp32` 为 true 和 false 时的梯度（BUGS_FIXED 里的风险项）。

## TensorBoard 读取：`tb_scalars.py`

最小实现的事件文件读取器，不依赖 tensorboard 或 protobuf：

```bash
python3 scripts/audit/tb_scalars.py RUN/train/exp_001/tensorboard/events.out... '^train/(reward|token_mult_prob_error)$'
```
