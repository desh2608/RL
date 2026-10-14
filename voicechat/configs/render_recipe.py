#!/usr/bin/env python3
"""Render a recipe template (voicechat/configs/*.template.json) into a concrete NeMo-RL recipe for scripts/make_run.py.

Placeholders: @RL_ROOT@ (workspace), @STUDENT_HF@ (policy HF dir, also tokenizer and rollout model), @BUNDLE@ (gym-data-v1
dir with train.jsonl / val.jsonl), @LINEAGE@ (experiments/<lineage>/{checkpoints,hf,monitor}), @SERVICES_YAML@ (Gym
services), @GYM_ASSET_DIR@ (Gym data assets, e.g. tau2 data), @RUN_NAME@ (logger name).
Optional --set dotted.key=JSON overrides (e.g. --set grpo.max_num_steps=40 --set policy.megatron_cfg.optimizer.lr=3e-6).
Usage: render_recipe.py --template T --out $NVC_RL_ROOT/config/my-run.json --student HF --bundle DIR --lineage NAME
       --services YAML [--rl-root $NVC_RL_ROOT] [--gym-asset-dir DIR] [--run-name NAME] [--set k=v ...]
"""
import argparse
import json
import os
import re

REF_ASSETS = "/lustre/fsw/portfolios/llmservice/users/cchen1/NemotronVC_rl/data/agentic-teacher-v1/assets"  # read-only on cs-oci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rl-root", default=os.environ.get("NVC_RL_ROOT"))
    ap.add_argument("--student", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--lineage", required=True)
    ap.add_argument("--services", required=True)
    ap.add_argument("--gym-asset-dir", default=REF_ASSETS)
    ap.add_argument("--run-name")
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args()
    assert a.rl_root, "--rl-root or NVC_RL_ROOT required"
    for path, what in [(os.path.join(a.student, "config.json"), "student HF dir"), (os.path.join(a.bundle, "train.jsonl"), "bundle"),
                       (a.services, "services yaml")]:
        assert os.path.exists(path), f"{what} not found: {path}"
    text = open(a.template).read()
    subs = {"@RL_ROOT@": os.path.realpath(a.rl_root), "@STUDENT_HF@": os.path.realpath(a.student), "@BUNDLE@": os.path.realpath(a.bundle),
            "@LINEAGE@": a.lineage, "@SERVICES_YAML@": os.path.realpath(a.services), "@GYM_ASSET_DIR@": a.gym_asset_dir,
            "@RUN_NAME@": a.run_name or a.lineage}
    for k, v in subs.items():
        text = text.replace(k, v)
    left = sorted(set(re.findall(r"@[A-Z_]+@", text)))
    assert not left, f"unrendered placeholders: {left}"
    cfg = json.loads(text)
    for kv in a.set:
        key, val = kv.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = json.loads(val)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(cfg, open(a.out, "w"), indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
