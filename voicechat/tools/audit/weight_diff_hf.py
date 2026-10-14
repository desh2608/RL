#!/usr/bin/env python3
"""Tensor-by-tensor diff of two HF checkpoints of the same architecture (e.g. RL start vs RL step N export).

Answers: which tensors changed at all, by how many bf16/fp32 ulps, whether frozen parts (vision/sound encoders, MoE router
weights and expert-bias) stayed bit-identical, and whether any tensor changed suspiciously much (export/conversion bug).
Per tensor: shape/dtype match, fraction of changed elements, ulp-distance histogram, relative L2 change. Aggregated per
category (regex on the tensor name). Reads safetensors lazily via the index; run in a CPU allocation (≈ 2x checkpoint size of
I/O). Usage: weight_diff_hf.py --a DIR_A --b DIR_B --out REPORT.json [--threads 8]
"""
import argparse
import collections
import json
import os
import re
import time

import torch
from safetensors import safe_open

CATEGORIES = [
    ("vision", r"(vision_model|vision_tower|radio|mlp1|vision_projection)"),
    ("sound", r"(sound|audio)"),
    ("router_bias", r"(e_score_correction_bias|expert_bias)"),
    ("router_weight", r"mixer\.gate\.weight$|router"),
    ("moe_shared_expert", r"shared_expert"),
    ("moe_experts", r"mixer\.experts\."),
    ("mamba_A_D_dtbias", r"mixer\.(A_log|D|dt_bias)$"),
    ("mamba_conv1d", r"mixer\.conv1d"),
    ("mamba_in_out_proj", r"mixer\.(in_proj|out_proj)"),
    ("mamba_norm", r"mixer\.norm"),
    ("attention", r"mixer\.(q_proj|k_proj|v_proj|o_proj)"),
    ("layer_norm", r"(norm\.weight|norm_f|final_norm|layernorm)"),
    ("embedding", r"embed"),
    ("lm_head", r"lm_head"),
]


def category(name):
    for cat, pat in CATEGORIES:
        if re.search(pat, name):
            return cat
    return "other"


def ordered_int(t):
    """Map float bits to integers whose difference is the ulp distance (sign-magnitude -> two's complement order)."""
    if t.dtype == torch.bfloat16 or t.dtype == torch.float16:
        i = t.view(torch.int16).to(torch.int32)
        mag = i & 0x7FFF
    elif t.dtype == torch.float32:
        i = t.view(torch.int32).to(torch.int64)
        mag = i & 0x7FFFFFFF
    else:
        return None
    return torch.where(i < 0, -mag, mag)


def index(d):
    p = os.path.join(d, "model.safetensors.index.json")
    if os.path.exists(p):
        return json.load(open(p))["weight_map"]
    files = [f for f in os.listdir(d) if f.endswith(".safetensors")]
    out = {}
    for f in files:
        with safe_open(os.path.join(d, f), "pt") as h:
            for k in h.keys():
                out[k] = f
    return out


class Reader:
    def __init__(self, d):
        self.d, self.map, self.open = d, index(d), {}

    def get(self, name):
        f = self.map[name]
        if f not in self.open:
            self.open[f] = safe_open(os.path.join(self.d, f), "pt")
        return self.open[f].get_tensor(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    A, B = Reader(a.a), Reader(a.b)
    names = sorted(set(A.map) | set(B.map))
    only_a = sorted(set(A.map) - set(B.map))
    only_b = sorted(set(B.map) - set(A.map))
    per, mismatched = [], []
    agg = collections.defaultdict(lambda: collections.Counter())
    bins = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 8), (9, 64), (65, 1 << 62)]
    t0 = time.time()
    for k, name in enumerate(n for n in names if n in A.map and n in B.map):
        x, y = A.get(name), B.get(name)
        cat = category(name)
        if x.shape != y.shape or x.dtype != y.dtype:
            mismatched.append({"name": name, "a": [str(x.dtype), list(x.shape)], "b": [str(y.dtype), list(y.shape)]})
            continue
        n = x.numel()
        neq = (x != y)
        nd = int(neq.sum())
        xf, yf = x.float(), y.float()
        rel = float((yf - xf).norm() / (xf.norm() + 1e-30))
        rec = {"name": name, "cat": cat, "dtype": str(x.dtype), "numel": n, "n_changed": nd, "frac_changed": nd / max(n, 1),
               "rel_l2": rel, "max_abs": float((yf - xf).abs().max()) if n else 0.0, "a_norm": float(xf.norm())}
        oi = ordered_int(x)
        if oi is not None and nd:
            u = (ordered_int(y) - oi).abs()
            rec["ulp_hist"] = {f"{lo}-{hi}" if hi < (1 << 62) else f">{lo - 1}": int(((u >= lo) & (u <= hi)).sum()) for lo, hi in bins}
            rec["max_ulp"] = int(u.max())
        per.append(rec)
        g = agg[cat]
        g["tensors"] += 1
        g["numel"] += n
        g["n_changed"] += nd
        g["tensors_changed"] += int(nd > 0)
        g["sq_diff"] += float(((yf - xf) ** 2).sum())
        g["sq_a"] += float((xf ** 2).sum())
        for b, c in rec.get("ulp_hist", {}).items():
            g["ulp_" + b] += c
        if (k + 1) % 500 == 0:
            print(f"[weight_diff] {k + 1}/{len(names)} tensors, {time.time() - t0:.0f}s", flush=True)
    summary = {}
    for cat, g in sorted(agg.items()):
        summary[cat] = {"tensors": g["tensors"], "tensors_changed": g["tensors_changed"], "numel": g["numel"],
                        "frac_elements_changed": g["n_changed"] / max(g["numel"], 1),
                        "rel_l2": (g["sq_diff"] / max(g["sq_a"], 1e-30)) ** 0.5,
                        "ulp_hist": {k[4:]: v for k, v in g.items() if k.startswith("ulp_")}}
    per.sort(key=lambda r: -r["rel_l2"])
    report = {"a": a.a, "b": a.b, "tensors_a": len(A.map), "tensors_b": len(B.map), "only_in_a": only_a, "only_in_b": only_b,
              "shape_or_dtype_mismatch": mismatched, "by_category": summary, "top_rel_l2": per[:40],
              "unchanged_by_category": {c: sum(1 for r in per if r["cat"] == c and r["n_changed"] == 0) for c in summary},
              "seconds": time.time() - t0}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(report, open(a.out, "w"), indent=1)
    with open(a.out.replace(".json", ".tensors.jsonl"), "w") as f:
        for r in per:
            f.write(json.dumps(r) + "\n")
    print(json.dumps({"by_category": summary, "only_in_a": len(only_a), "only_in_b": len(only_b),
                      "mismatch": len(mismatched)}, indent=1))


if __name__ == "__main__":
    main()
