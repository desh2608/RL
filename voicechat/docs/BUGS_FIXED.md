# 发现并修过的问题

截至 2026-10-14，这份代码前后做过约 6 轮排查：
- 09-20 前后：续训、打包、日志；
- 09-29：RL 不涨分的诊断；
- 10-01 到 10-03：在线链路稳定性；
- 10-06：梯度同步审计；
- 10-13：Mamba 数值；
- 10-13：全链路底层审计。

每条按"现象 / 原因 / 修法 / 怎么自检"写。"已修"指代码或模板默认值已经包含修复；"配置"指要靠 recipe 设置，模板里已设好。

## 训练信号与数据渲染（影响最大）

### B3 system 消息被渲染成 Python 列表字符串（已修，配置）
- **现象**：解码 rollout，带 system 的行开头是 `<|im_start|>system\n[{'type': 'text', 'text': '...\n...'}]`，整段 policy 压成一行，换行成了字面的 `\n`。评测侧是纯文本。2026-10-02 起的每一次 Gym RL 都有这个问题，tau2/CTU/telecom 方向的行全部受影响。
- **原因**：NeMo-RL 内嵌的 vLLM HTTP server 默认 `chat_template_content_format="auto"`（`nemo_rl/models/generation/vllm/vllm_worker_async.py`）。Gym 发来的是 content-part 列表，auto 检测判成 openai 格式，模板对 system 做 `| string` 就打印出了列表。原生 `vllm serve`（画像用）同样如此。
- **修法**：
  - 训练：`policy.generation.vllm_cfg.http_server_serving_chat_kwargs.chat_template_content_format: "string"`；
  - 画像：`launch_profile.sh --content-format string`（默认值）。
- **自检**：`smoke_check.py` 第 1 项；画像目录里的 `RENDER_CHECK.json`。

### B4 同一 prompt 的 K 个采样共用一个 seed（已修，数据）
- **现象**：T=1.0 下，约 100 token 的回复，一组 16 条只有 2–6 种不同输出，还共享很长的前缀。画像时"有对有错"的 prompt，训练时大量变成全对或全错被丢弃（58% 的组被丢）。
- **原因**：画像工具的物化输入里有每次重复的 `responses_create_params.metadata.extra_body = {"seed": k}`。直接拿它当训练行，就留下了 seed=0；Gym 把 extra_body 合并进请求，NeMo-RL 复制 K 份时不会改 seed。
- **修法**：`pivot_data/build_bundle.py` 删掉 seed 并断言；`clean_candidates.py` 同样断言。
- **自检**：`smoke_check.py` 第 2 项。修复后 48/48 组都是 16 种不同输出。

### B5 advantage 用留一标准差，叠加长度惩罚后被放大（已修，配置）
- **现象**：一组 15 对 1 错时，答错的那条 advantage 是 -17 到 -20（裁剪前 -26）；1 对 15 错时，答对的那条只得 +1；2 对时得 +4.48。每步只有 1–1.5% 的样本，却占了 10–46% 的梯度权重。
- **原因**：标准差用的是"除自己以外 15 条"（`nemo_rl/algorithms/utils.py::calculate_baseline_and_std_per_prompt`）。长度惩罚把答对样本的奖励摊到 1±0.2，于是 15 条答对样本的标准差只有约 0.06。全错组其余样本的标准差为 0，又不做归一化。
- **修法**：`nemo_rl/algorithms/advantage_estimator.py` 新增两个配置项，默认值保持旧行为：
  - `grpo.adv_estimator.std_scope: group`（全组标准差）；
  - `grpo.adv_estimator.std_source: unshaped_reward`（用长度惩罚前的验证器奖励算标准差）。

  另外把 `grpo.advantage_clip_low/high` 设为 ±5。
- **自检**：
  - `scripts/audit/test_adv_std_scope.py`：默认值与旧代码逐位一致；新配置下单个失败和单个成功分别是 -4 和 +4；
  - 训练 driver.log 的 `Advantages stats` 恒为 ±4。

### B6 `parallel_tool_calls: false` 让验证器看不到多余的调用（已修，数据）
- **现象**：模型一次输出两个 `<tool_call>`，单调用验证器却给 1 分，整段多调用的 token 照样被正向训练。
- **原因**：请求设了 `parallel_tool_calls: false` 时，vLLM 只返回第一个调用，验证器的 `tool_call_count > 1 → 0` 检查永远不会触发。模型的 chat template 里也没有用到这个开关。
- **修法**：所有行统一设为 `parallel_tool_calls: true`（`clean_candidates.py` 和 `build_bundle.py` 都有断言）。

### B7 RL 行的提示形态与评测不一致（已修，数据）
- 历史里的 reasoning：评测客户端（bfcl 和 tau2）从不回传历史 reasoning，RL 行却保留了。
- `ENVIRONMENT REMINDER: You have N turns left`：tau2 采样残留，评测里没有。
- prompt 以 assistant 文本结尾：教师在同一轮里先说话再调用，切成 pivot 后，模型被要求在自己说完话之后再开一轮 assistant，评测中从不出现。
- **修法**：`clean_candidates.py`，规则与 SFT 转换一致。

## 数值

### B1 Mamba 训练时 SSM state 的 FP32 配置没有生效（已修）
- **现象**：recipe 设了 `policy.megatron_cfg.mamba_training_ssm_states_dtype: float32`，模型实际用的却是 BF16。
- **原因**：NeMo-RL 没有把这一项传到 Megatron 的 model config。mamba_ssm 支持 state_dtype，所以不会报错。
- **修法**：`nemo_rl/models/megatron/mamba_numerics.py` 加上 `setup.py` 里的 3 行。每个 rank 打印 `[MAMBA_NUMERICS] {...}`，并断言每个 MambaMixer 的 state dtype。
- **自检**：driver.log 里 grep `MAMBA_NUMERICS`，应看到 `"state_dtypes": {"torch.float32": 23}`。

### B2 Mamba SSD 的 Triton kernel 在各 rank 上独立 autotune（已修，配置）
- **现象**：同样的权重、同样的输入，不同 rank 选到不同的 kernel 配置，logprob 和梯度不一致。
- **修法**：`policy.megatron_cfg.env_vars.MAMBA_DETERMINISTIC: "1"`，即 mamba_ssm 自带的确定性模式。8 卡实测前向和梯度逐位一致，没有变慢。
- **注意**：不要手动修改 `Autotuner.configs` 来固定配置。部分 workspace 在 import 时按最小 BLOCK_SIZE_N 分配大小，固定成更大的 block 会留下未初始化的 tile，梯度偏差可达 1e8。探针见 `scripts/audit/mamba_pin/`。

### B8 两端 logits 都舍入到 bf16，这是训练/推理失配的主要来源（已修，配置）
- **现象**：`train/token_mult_prob_error` 约 1.034，`gen_kl_error` 约 0.0027。第 1 步（权重刚同步）就是这个值，所以几乎全是数值差异。
- **修法**：两端同时开 fp32 lm head：
  - 训练端 `policy.megatron_cfg.fp32_lm_head: "tf32"`（输入本来就是 bf16 精确值，用 TF32 张量核数值上等价，速度接近基线）；
  - vLLM 端 `policy.generation.vllm_cfg.env_vars.NRL_VLLM_FP32_LM_HEAD: "1"`。

  只开一端会更糟，`setup.py` 里有校验，两端不一致直接报错。
- **实测**：与 B9 一起，失配从 1.034 降到 1.017，gen_kl 从 0.0027 降到 0.0007。每步耗时增加约 15–20%。

### B9 vLLM 的 SSM 缓存用 fp16 + 随机舍入（已修，配置）
- **原因**：在 A100 上，随机舍入走软件实现：|x| < 6.1e-5 直接清零，|x| ≥ 65520 直接变成 Inf，不做饱和截断。每个 decode 步写回 state 都会经过这条路径。训练端的 state 是 fp32。
- **修法**：`vllm_kwargs.mamba_ssm_cache_dtype: float32`，去掉 `enable_mamba_cache_stochastic_rounding` 和 `mamba_cache_philox_rounds`。这也是上游 Nemotron recipe 的默认值。
- **注意**：评测 harness 目前仍用 fp16 + 随机舍入。模型之间对比是公平的，但绝对分可能被拉低。

### B10 重要性采样权重的下限 0.2 会抬高异常 token（已修，配置）
- **原因**：曾出现一次 `token_mult_prob_error = 8.19` 的尖峰：约 35 个孤立 token 上，生成端 logprob 比训练端高 14–17 nat。TIS 下限 0.2 把这些本应约等于 e^-15 的权重抬到了 0.2，等于在这些 token 上照样学习。
- **修法**：`loss_fn.truncated_importance_sampling_ratio_min: null`，即下限为 0。
- **未查明**：尖峰的根因不明，推测是权重刷新抢占请求后 token 与 logprob 错位。在 `force_on_policy_ratio` 下，训练端 logprob 不会写进 dump。

### B11 MoE 路由偏置在 RL 中漂移（已修，配置）
- **现象**：`e_score_correction_bias` 每步漂移约 0.001，23 层都有，和学习率无关。
- **修法**：`moe_router_bias_update_rate: 0` 加 `freeze_moe_router: true`。
- **自检**：`scripts/audit/weight_diff_hf.py` 对比起点和第 N 步的 HF 导出，路由权重和偏置应逐位不变（已实测）。
- **注意**：偏置保持 fp32 依赖 `MoEFloat16Module`，它只在 `freeze_moe_router: true` 时启用；偏置在 56 附近，一旦转成 bf16 就会被抹平。

### 风险项：梯度用 bf16 累加（未改，风险）
`grad_reduce_in_fp32: false`（上游所有 RL recipe 都是这样）。每个 DP rank 每步至少累加 236 个 microbatch，估计每个梯度元素有 3–4% 的相对误差。还没有实测。验证方法见 `scripts/audit/gradsync_check` 和 CHECKS.md。

## 链路与基础设施

### B12 `truncated` 标志与长度过滤（已修）
`truncated` 原来拿总 token 数和 `max_model_len` 比较，单轮写满 `max_new_tokens` 不会被标记。现在 Gym 路径上 truncated 表示单轮输出写满了 `max_new_tokens`（测试 `tests/unit/experience/test_nemo_gym_truncation.py`）。

### B13 Gym 路径的周期验证会让训练崩溃（已规避）
`validate()` 按训练的组大小分组，但验证集每个 prompt 只有 4 条，会抛 `Expected one _ng_task_index per prompt group`。`make_run.py --kind train` 默认关闭周期验证（`--keep-gym-validation` 可以保留）。

### B14 引擎亲和 key 把 tau2 rollout 全压到一个引擎上（已修，Gym）
tau2 的每条 episode 都以相同的 system 和固定开场白开头。亲和 key 现在取到第一条 user 之后的第一条 assistant 回复为止（Gym `responses_api_models/vllm_model/app.py`，测试 `test_engine_affinity.py`）。

### B15 vLLM 引擎猝死后请求一直打到死节点（已缓解，Gym）
刷新权重后，偶发 `RuntimeError: deque mutated during iteration`，之后该引擎返回 `EngineDeadError` 或 500。Gym 的 `vllm_model` 在收到 5xx 后，把该端点冷却 120 秒并切到下一个健康端点。监控里 `EngineDeadError` 的计数应为 0。

### B16 续训时改变 learner 节点数，RNG 状态加载失败（已修，配置）
RNG 状态按 (PP, TP, DP×CP) 分片保存。改变 DP 时必须设 `policy.megatron_cfg.checkpoint.load_rng: false`。

### B17 HDO（混合分布式优化器）续训恢复的三个缺陷（已修，三方 patch）
- 按 DP 重分片加载后，FP32 master 没有同步；
- 原本就是 FP32 的参数，它的 CPU 副本没有恢复；
- 非持久的 DCP 模板覆盖了已经恢复的 step。

修复见 `voicechat/third_party/megatron-lm-patches/` 和 `megatron-bridge-patches/`。续训 receipt 里有 `master_model_exact_equal`（`NVC_HDO_MASTER_MODEL_STRICT=1` 时不一致会直接报错）。

### B18 Omni 模型的打包与 CP（已修，三方 patch）
Omni 在 CP1/PP1 下的 attention 要保留物理打包布局；还加了两路 context parallel 的文本打包（Megatron-Bridge patch 0001、0002）。

### B19 梯度同步审计（已验证无问题）
在另一个 KD 栈里发现过：provider 深拷贝 config 后，`finalize_model_grads_func` 只绑在原对象上，DP 同步被静默跳过。本栈的审计结论是：
- 绑定在 DDP 包装后模型的 config 上；
- DP4 与 DP1 的梯度相对误差 0.81%；
- 负对照（不绑回调）误差 142%。

探针见 `scripts/audit/gradsync_check/`，换并行布局或升级 Bridge 后建议重跑。

### B20 配置层面的教训
- **生成长度要对齐评测预算**：`policy.generation.max_new_tokens` 必须与评测预算一致（BFCL 每轮 2048，含 reasoning）。曾经用 8192 训练，学出了 870 token 的固定思考模板，评测时 11% 被截断，multi-turn 从 44.9 掉到 36.6。
- **parallel 验证器要严格**：parallel 验证器用 `allow_superset: true` 加 f1 时，"正确集合 + 1 个多余调用"能得 0.8 分，会学出多调用，并泄漏到单调用 prompt。应该用 `allow_superset: false` 加 `binary_strict`。
- **message gold 不能太少**：只占 12% 且集中在多工具 prompt 时，irrelevance 必然下降。
- **`make_run.py --nodes N`**：会把生成节点强制设成 N−2，正式训练不要传。

### B21 收尾作业的两个缺陷（已修，`finish_chain.sbatch`）
- 收尾作业（64G）和它提交的 64G 准备作业都在 `cpu_interactive` 上，合计超过每用户 88G，准备作业永远排不上，收尾作业原地等死。现在收尾作业只用 20G。
- 续训作业号原来从 `SUBMISSIONS` 文件读，但 RL run 目录记录的是 `JOB_ID`。读到空值时，收尾作业误以为续训已结束，又去构建第二次续训。现在改为读 `JOB_ID`，并检查是否为数字。

## 相对上游基线改动的文件（审计入口）

```
nemo_rl/algorithms/{advantage_estimator,grpo}.py              B5，以及异步收集、零方差丢弃、closed_trajectories dump
nemo_rl/algorithms/async_utils/{replay_buffer,trajectory_collector}.py
nemo_rl/environments/nemo_gym.py, nemo_rl/experience/{rollouts,rollout_manager}.py    B12，Gym 结果转换
nemo_rl/models/generation/vllm/{config,vllm_generation,vllm_worker_async}.py          刷新/前缀缓存重置、serving kwargs
nemo_rl/models/megatron/{setup,mamba_numerics,hdo_resume}.py, models/policy/workers/megatron_policy_worker.py   B1、B8、B17
nemo_rl/utils/{length_penalty,training_guardrails}.py, nemo_rl/data/full_pass.py
3rdparty/Gym-workspace/Gym: rollout_collection.py, vllm_model/app.py（B14、B15）, openai_model/app.py（多 base_url）,
  tau2/app.py, single_step_tool_use_with_argument_comparison/*（并行批次验证）, math_with_judge/app.py
```
