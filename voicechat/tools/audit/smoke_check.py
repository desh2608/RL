#!/usr/bin/env python3
"""Acceptance checks for an RL run after the 2026-10-13 infra audit (analysis/rl-infra-audit-20261013/REPORT.md), from the
run's own artifacts (stdlib only, light enough for the login node):
1. system rendering: decoded rollouts (closed_trajectories) contain no Python-list system message;
2. sampling diversity: distinct generations per 16-sample group (v9 with seed=0: 2-6 for short answers);
3. advantages: driver.log "Advantages stats" min/max (pre-clip) and the dumped (post-clip) range;
4. train/rollout mismatch: tensorboard train/token_mult_prob_error and gen_kl_error per step, against a reference run.
Usage: smoke_check.py RUN_DIR [--ref REF_RUN_DIR] [--tokenizer HF_DIR]
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TB = os.path.join(HERE, "tb_scalars.py")


def tb(run, pat):
    ev = sorted(glob.glob(os.path.join(run, "train", "exp_*", "tensorboard", "**", "events*"), recursive=True), key=os.path.getmtime)
    if not ev:
        return "no tensorboard"
    return subprocess.run([sys.executable, TB, ev[-1], pat], capture_output=True, text=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--ref")
    ap.add_argument("--tokenizer", required=True, help="HF dir of the policy (tokenizer.json)")
    a = ap.parse_args()
    tok = json.load(open(os.path.join(a.tokenizer, "tokenizer.json")))
    inv = {v: k for k, v in tok["model"]["vocab"].items()}
    for t in tok.get("added_tokens", []):
        inv[t["id"]] = t["content"]
    dec = lambda ids: "".join(inv.get(i, "?") for i in ids).replace("Ġ", " ").replace("Ċ", "\n")
    ok = True
    dumps = sorted(glob.glob(os.path.join(a.run, "train", "exp_*", "closed_trajectories", "step*.jsonl")))
    print("dumps:", [os.path.basename(d) for d in dumps])
    sys_rows = repr_rows = 0
    distinct = []
    for d in dumps:
        rows = [json.loads(line) for line in open(d)]
        for r in rows:
            m = r["token_loss_mask"]
            prompt = dec(r["token_ids"][:m.index(1)] if 1 in m else r["token_ids"])
            for seg in prompt.split("<|im_start|>")[1:]:
                if seg.startswith("system\n") and not seg.startswith("system\n# Tools"):
                    sys_rows += 1
                    if seg.startswith("system\n[{"):
                        repr_rows += 1
        for b in range(len(rows) // 16):
            blk = rows[16 * b:16 * b + 16]
            gens = {tuple(t for t, k in zip(r["token_ids"], r["token_loss_mask"]) if k) for r in blk}
            mean_len = sum(sum(r["token_loss_mask"]) for r in blk) / 16
            distinct.append((len(gens), round(mean_len)))
    print("1. system segments: %d, rendered as Python list: %d -> %s" % (sys_rows, repr_rows, "OK" if repr_rows == 0 else "FAIL"))
    ok &= repr_rows == 0
    short = [n for n, L in distinct if L < 200]
    print("2. distinct generations per group (all):", [n for n, _ in distinct])
    print("   groups with mean length < 200 tokens: %s (v9 seed=0: mostly 2-6)" % short)
    if short:
        ok &= sum(short) / len(short) >= 10
    adv = []
    for line in open(os.path.join(a.run, "driver.log"), errors="replace"):
        mm = re.search(r"Advantages stats: min=(-?[\d.]+), max=(-?[\d.]+)", line)
        if mm:
            adv.append((float(mm.group(1)), float(mm.group(2))))
    print("3. pre-clip advantage min/max per step:", adv)
    if adv:
        ok &= max(abs(x) for p in adv for x in p) <= 5.5
    print("4. mismatch, this run:\n" + tb(a.run, r"^train/(token_mult_prob_error|gen_kl_error|reward)$"))
    if a.ref:
        print("   reference %s:\n%s" % (a.ref, tb(a.ref, r"^train/(token_mult_prob_error|gen_kl_error|reward)$")))
    print("ACCEPT" if ok else "REJECT")


if __name__ == "__main__":
    main()
