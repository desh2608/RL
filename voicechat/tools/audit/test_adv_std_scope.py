#!/usr/bin/env python3
"""Checks for grpo.adv_estimator.std_scope / std_source (RL audit 2026-10-13, analysis/rl-infra-audit-20261013).

1. Defaults reproduce the legacy estimator bit for bit (synthetic groups + real closed-trajectory dumps if given).
2. std_scope=group + std_source=unshaped_reward: a lone failure among 15 length-shaped successes gets about -4 (legacy -17),
   a lone success among 15 failures about +4 (legacy +1), all-equal groups stay 0.
3. On dumps (closed_trajectories/stepN.jsonl, 16-row blocks = groups), prints max |advantage| and the share of
   |advantage| x tokens carried by the top 1 % of samples, legacy vs fixed.
Usage: test_adv_std_scope.py [--module PATCHED.py] [--legacy-module ORIG.py] [DUMP.jsonl ...]
(run with the driver venv python, cwd = repo root so nemo_rl imports resolve)
"""
import argparse
import importlib.util
import json
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "nemo_rl", "algorithms", "advantage_estimator.py")  # repo root = voicechat/tools/audit/../../..


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.realpath(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(mod, rewards, unshaped, groups, **cfg):
    est = mod.GRPOAdvantageEstimator(mod.AdvEstimatorConfig(**cfg), None)
    mask = torch.ones(len(rewards), 1)
    rb = {"unshaped_total_reward": unshaped}
    return est.compute_advantage(prompt_ids=groups.unsqueeze(1), rewards=rewards, mask=mask,
                                 reward_valid_mask=torch.ones(len(rewards)), repeated_batch=rb)[:, 0]


def synthetic():
    g = torch.arange(4).repeat_interleave(16)
    unshaped = torch.zeros(64)
    unshaped[0:15] = 1.0          # group 0: 15 correct, 1 wrong
    unshaped[16] = 1.0            # group 1: lone success
    unshaped[32:48] = 1.0         # group 2: all correct (dropped upstream; must stay finite)
    rewards = unshaped.clone()    # group 3: all wrong
    shaped = torch.linspace(-0.1, 0.1, 15)  # real dumps: std of 15 shaped successes ~0.06
    rewards[0:15] += shaped
    rewards[32:48] += torch.linspace(-0.2, 0.2, 16)
    return rewards, unshaped, g


def dump_groups(path):
    rows = [json.loads(line) for line in open(path)]
    n = len(rows) // 16 * 16
    rows = rows[:n]
    rewards = torch.tensor([float(r["total_reward"]) for r in rows])
    unshaped = torch.tensor([float(r["unshaped_total_reward"]) for r in rows])
    ntok = torch.tensor([float(sum(r["token_loss_mask"])) for r in rows])
    groups = torch.arange(n // 16).repeat_interleave(16)
    for b in range(n // 16):
        assert len({rows[i]["agent_ref"]["name"] for i in range(16 * b, 16 * b + 16)}) == 1, f"block {b} mixes agents"
    return rewards, unshaped, groups, ntok


def concentration(adv, ntok):
    w = adv.abs() * ntok
    k = max(1, len(w) // 100)
    return float(w.topk(k).values.sum() / w.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default=DEFAULT)
    ap.add_argument("--legacy-module")
    ap.add_argument("dumps", nargs="*")
    a = ap.parse_args()
    new = load(a.module, "adv_new")
    old = load(a.legacy_module, "adv_old") if a.legacy_module else None
    fixed = dict(std_scope="group", std_source="unshaped_reward")

    rewards, unshaped, g = synthetic()
    leg = run(new, rewards, unshaped, g)
    if old is not None:
        ref = old.GRPOAdvantageEstimator(old.AdvEstimatorConfig(), None).compute_advantage(
            prompt_ids=g.unsqueeze(1), rewards=rewards, mask=torch.ones(64, 1), reward_valid_mask=torch.ones(64))[:, 0]
        assert torch.equal(leg, ref), "defaults changed the legacy advantages"
    fx = run(new, rewards, unshaped, g, **fixed)
    print("legacy lone failure %.3f lone success %.3f | fixed lone failure %.3f lone success %.3f"
          % (leg[15], leg[16], fx[15], fx[16]))
    assert leg[15] < -10 and abs(leg[16] - 1.0) < 1e-6, "legacy pathology not reproduced"
    assert -4.8 < fx[15] < -3.2 and 3.2 < fx[16] < 4.2, "fixed advantages not symmetric around +-4"
    assert torch.isfinite(fx).all() and fx[48:64].abs().max() == 0
    print("synthetic: OK")

    for p in a.dumps:
        rewards, unshaped, g, ntok = dump_groups(p)
        leg = run(new, rewards, unshaped, g)
        if old is not None:
            ref = old.GRPOAdvantageEstimator(old.AdvEstimatorConfig(), None).compute_advantage(
                prompt_ids=g.unsqueeze(1), rewards=rewards, mask=torch.ones(len(rewards), 1),
                reward_valid_mask=torch.ones(len(rewards)))[:, 0]
            assert torch.equal(leg, ref), f"{p}: defaults changed the legacy advantages"
        fx = run(new, rewards, unshaped, g, **fixed)
        dumped = []
        for line in open(p):
            r = json.loads(line)
            m = r["token_loss_mask"]
            dumped.append(r["advantages"][m.index(1)] if 1 in m else 0.0)
        dumped = torch.tensor(dumped[:len(leg)])
        agree = float((leg.clamp(-20, 20) - dumped).abs().max())
        print(json.dumps({"dump": os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(p)))) + "/" + os.path.basename(p),
                          "legacy_max_abs": round(float(leg.abs().max()), 2), "fixed_max_abs": round(float(fx.abs().max()), 2),
                          "legacy_top1pct_mass": round(concentration(leg.clamp(-20, 20), ntok), 3),
                          "fixed_top1pct_mass": round(concentration(fx, ntok), 3),
                          "legacy_vs_dumped_max_diff(-2 override ignored)": round(agree, 4)}))
    print("ALL_OK")


if __name__ == "__main__":
    main()
