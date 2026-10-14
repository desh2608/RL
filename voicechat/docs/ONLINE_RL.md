# Online RL (multi-turn episodes sampled live)

Online RL runs on the same code as offline RL (NeMo-RL async GRPO + Gym + vLLM HTTP server). The differences:

| | Offline pivot | Online |
|---|---|---|
| Data row | history + gold next step | task spec (tau2 task id + user scenario, CTU scenario, workplace task, ...) |
| Who drives the turns | nobody (one turn only) | Gym agent: `tau2`, `conversational_tool_use_agent`, `workplace_assistant_simple_agent`, ... |
| User side | none | user simulator: local gpt-oss-120b nodes or hub API |
| Reward | next step vs gold (0/1) | episode outcome (tau2 evaluator: DB state + required actions; CTU: judge) |
| Trained tokens | one assistant turn | every assistant turn of the episode; user/tool content masked |
| Resources | 14 nodes | Ray nodes + `--sim-nodes N` simulator nodes (outside Ray, standalone vLLM) |

## Which fixes apply

All code- and config-level fixes in `BUGS_FIXED.md` apply online as well:
- B3: system rendered as text;
- B5: advantages;
- B8–B10: fp32 LM head, fp32 SSM cache, TIS lower bound;
- B1, B2: the two Mamba fixes;
- B11–B16: router bias, engine affinity, engine failover, resume.

`voicechat/configs/online-agentic-rl.template.json` has them all enabled.

The data-level fixes behave differently online:
- online rows must also **carry no sampling seed**, or the K episodes of a task become near-identical;
- the history is produced live by the agent, so the offline leftovers (B6, B7) do not arise, but check that the agent
  passes back history reasoning the same way the eval does.

**Validation status**:
- The online path ran end to end on 2026-10-01..03 (smoke and training). The engine-affinity (B14) and failover (B15)
  fixes came out of those runs.
- After the 2026-10-13 fixes it has **not** been re-run end to end. Start with a smoke and the four `smoke_check.py` checks.
- Known risk: with P256, ~10k-token / 16-turn tau2 episodes and one simulator node (~14 req/s), step 0 took over an hour.
  Shrink the batch or add simulator nodes so step 0 finishes well within the cluster's idle-cancel window (30 min).

## Steps

1. **Data**: example at `/lustre/fsw/portfolios/llmservice/users/cchen1/NemotronVC_rl/data/agentic-teacher-v1/` (read-only).
   - Contents: tau2 telecom non-eval split 6,154 rows, CTU 12,821, workplace 2,564, plus some single-step routes.
   - Gym assets (tau2 DBs) are in `assets/tau2_data`.
   - Your own bundles must keep the gym-data-v1 layout (`train.jsonl`, `val.jsonl`, `MANIFEST.json`) and carry no seed.
2. **Recipe**:

```bash
python3 voicechat/configs/render_recipe.py --template $P/config/online-agentic-rl.template.json --out $P/config/<run>.json \
  --student <start HF> --bundle <bundle dir> --lineage <lineage> --services $P/config/gym-services-online-localsim.yaml \
  --gym-asset-dir /lustre/fsw/portfolios/llmservice/users/cchen1/NemotronVC_rl/data/agentic-teacher-v1/assets
```

   Template defaults:
   - 10 Ray nodes (8 rollout + 2 learner), P256 × K8;
   - `max_new_tokens` 8192 (multi-turn episodes need more);
   - otherwise the same fixes as the offline template.

3. **Smoke / train**: same as offline, plus simulator nodes in `make_run.py`:

```bash
make_run.py --kind smoke --project $P --recipe $P/config/<run>.json --services $P/config/gym-services-online-localsim.yaml \
  --bundle <bundle> --sim-nodes 1 --sim-model-path $P/models/sim/gpt-oss-120b
```

   - `gpu_job.sh` runs the last N hosts of the allocation as simulators (`scripts/sim_role.sh`, vLLM 0.27.1, TP1×DP8,
     about 40 req/s per node) and exports their URLs in `NVC_SIM_BASE_URL` (comma-separated). Gym's `openai_model`
     routes to the endpoint with the fewest in-flight requests.
   - Set `sim_gptoss.max_concurrent_requests` to 384 × nodes.
   - Hub simulator instead: use `gym-services-online-hub.yaml`, no `--sim-nodes`, and your own key in
     `$P/.secrets/nvidia-api-key` (mode 600; injected via the environment only).

4. **Within 20 minutes of the start**, run `nvidia-smi` on every generation node. If all load sits on one engine, the
   engine-affinity key is wrong (B14).

## Notes

- **Do not use the training policy as its own user simulator or judge**: it confuses roles and produces false passes.
- **tau2 Gym venv**: it lives in the shared `runtime/gym-v1/venvs-ext`, pinned to tau2 commit `60c2a0db` (same as the eval
  harness). Rebuild with `scripts/build_gym_venvs_ext.sh` only into your own venvs-ext.
- **Length monitor**: `checkpointing.length_monitor` has per-agent cap-hit thresholds. Online episodes are long; relax the
  thresholds or the run stops early.
