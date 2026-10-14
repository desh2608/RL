#!/usr/bin/env python3
"""Gym RL bundle (gym-data-v1, input of scripts/make_run.py --bundle) from the start policy's pass-rate profile of cleaned
candidate rows (pivot_data/clean_candidates.py -> profile_passrate/profile_candidates.sbatch): keep 0 < pass < 1 (groups
with GRPO signal), cap each route (CAPS / --cap), write train.jsonl, val.jsonl, MANIFEST.json, BUNDLE_REPORT.md.

Rules (voicechat/docs/BUGS_FIXED.md):
* the per-repeat sampling seed that the profiler writes into responses_create_params.metadata.extra_body is removed
  (bug B4: a kept seed made all K training samples of a prompt identical up to numerics);
* every row is checked: no seed / extra_body, parallel_tool_calls true, no history reasoning item, no ENVIRONMENT
  REMINDER, prompt does not end with an assistant message or an unanswered function_call;
* a route may reuse an older profile (--reuse DIR:route,route) only if cleaning changed none of its rows (duplicates may
  be dropped) and no row has a system message (rendering of system messages changed with bug B3); duplicate prompts are
  dropped.
CAPS below are one example recipe (route keys of pivot_data/example_build_candidates.py); pass --cap for your routes.
Usage: build_bundle.py --profile PROFILE_DIR [--reuse OLD_PROFILE_DIR:route,route] --clean-manifest CLEAN_DIR/CLEAN_MANIFEST.json
       --out data/<bundle> --bundle-name NAME --start-policy HF_DIR [--cap route=N ...]
"""
import argparse
import collections
import glob
import hashlib
import json
import os

CAPS = {"bfcl_mt_call": 100000, "bfcl_mt_msg": 100000, "nocall": 100000, "nearmiss": 100000, "tau2_write": 100000,
        "single_step": 1200, "bfcl_mt_batch": 600, "telecom": 400, "parallel": 300, "tau_mms": 300, "ctu": 200}
VAL_PER_ROUTE = 8


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def finalize(r):
    """Strip profiler bookkeeping and check the row against the audit rules; returns the row."""
    r = {k: v for k, v in r.items() if not k.startswith("_ng_")}
    p = r["responses_create_params"]
    md = p.get("metadata")
    if isinstance(md, dict):
        md.pop("extra_body", None)
        if not md:
            p.pop("metadata")
    md = p.get("metadata") or {}
    assert "seed" not in p and "seed" not in md and "extra_body" not in md, "sampling seed in row"
    assert p.get("parallel_tool_calls") is True, "parallel_tool_calls must be true"
    items = p["input"]
    assert not any(it.get("type") == "reasoning" for it in items), "history reasoning item"
    assert "ENVIRONMENT REMINDER" not in json.dumps(items), "ENVIRONMENT REMINDER left"
    last = items[-1]
    assert not (last.get("role") == "assistant" or last.get("type") == "function_call"), "prompt ends with assistant/call"
    return r


def load_profile(d, routes=None):
    rate, rows = {}, collections.defaultdict(dict)
    for f in glob.glob(os.path.join(d, "g*/per_prompt.jsonl")):
        for line in open(f):
            p = json.loads(line)
            if routes is None or p["route_key"] in routes:
                rate[(p["route_key"], p["_ng_task_index"])] = p["pass_rate"]
    for rd in sorted(glob.glob(os.path.join(d, "g*/rollouts/*"))):
        key = os.path.basename(rd)
        if routes is not None and key not in routes:
            continue
        for line in open(os.path.join(rd, "rollouts_materialized_inputs.jsonl")):
            r = json.loads(line)
            t = r["_ng_task_index"]
            if t not in rows[key]:
                rows[key][t] = r
    return rate, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--reuse", action="append", default=[], help="DIR:route[,route] profiled earlier on identical rows")
    ap.add_argument("--clean-manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap", action="append", default=[])
    ap.add_argument("--seed", default=None, help="ordering salt (default: bundle name)")
    ap.add_argument("--bundle-name", required=True)
    ap.add_argument("--start-policy", required=True, help="HF dir of the policy that was profiled (recorded in MANIFEST)")
    a = ap.parse_args()
    a.seed = a.seed or a.bundle_name
    caps = dict(CAPS)
    for c in a.cap:
        k, v = c.split("=")
        caps[k] = int(v)
    assert not os.path.exists(a.out), a.out
    clean = json.load(open(a.clean_manifest))["routes"]
    rate, rows = load_profile(a.profile)
    source = {k: os.path.realpath(a.profile) for k in rows}
    reused = set()
    for spec in a.reuse:
        d, keys = spec.split(":")
        keys = set(keys.split(","))
        for k in keys:
            assert clean[k]["rows_unchanged"] == clean[k]["rows_out"], f"route {k} changed in cleaning; cannot reuse {d}"
            assert k not in rows, f"route {k} profiled twice"
        r2, rows2 = load_profile(d, keys)
        rate.update(r2)
        rows.update(rows2)
        reused |= set(rows2)
        source.update({k: os.path.realpath(d) for k in rows2})
    missing = sorted(set(clean) - set(rows))
    assert not missing, f"routes without a profile: {missing}"
    train, val, report = [], [], {}
    for key in sorted(rows):
        hist = collections.Counter()
        mixed, rest = [], []
        seen = set()
        for t, r in sorted(rows[key].items()):
            h = hashlib.sha256(json.dumps(r["responses_create_params"], sort_keys=True).encode()).hexdigest()
            if h in seen:
                hist["duplicate_dropped"] += 1
                continue
            seen.add(h)
            if key in reused:
                assert not any(it.get("role") in ("system", "developer") for it in r["responses_create_params"]["input"]), \
                    f"reused route {key} has a system message (rendering changed); re-profile it"
            p = rate.get((key, t))
            hist["missing" if p is None else "0" if p == 0 else "1" if p == 1 else "mixed"] += 1
            r = finalize(r)
            r.setdefault("info", {})
            if isinstance(r["info"], dict):
                r["info"] = dict(r["info"], start_pass_rate=p)
            (mixed if p is not None and 0 < p < 1 else rest).append(r)
        order = lambda r: hashlib.sha256((a.seed + key + json.dumps(r.get("uuid") or r["responses_create_params"], sort_keys=True)).encode()).hexdigest()
        mixed.sort(key=order)
        rest.sort(key=order)
        pick = mixed[:caps[key]]
        train += pick
        val += rest[:VAL_PER_ROUTE]
        report[key] = {"agent": pick[0]["agent_ref"]["name"] if pick else None, "profiled": len(rows[key]), "pass_hist": dict(hist),
                       "mixed": len(mixed), "train": len(pick), "cap": caps[key], "profile": source[key]}
    train.sort(key=lambda r: hashlib.sha256((a.seed + json.dumps(r.get("uuid") or r["responses_create_params"], sort_keys=True)).encode()).hexdigest())
    os.makedirs(a.out)
    for name, rs in (("train.jsonl", train), ("val.jsonl", val)):
        with open(os.path.join(a.out, name), "w") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    routes = collections.Counter(r["agent_ref"]["name"] for r in train)
    man = {"schema": "gym-data-v1", "bundle": a.bundle_name, "data_path": "train.jsonl", "rows": len(train),
           "data_sha256": sha(os.path.join(a.out, "train.jsonl")), "val_sha256": sha(os.path.join(a.out, "val.jsonl")),
           "val_rows": len(val), "route_counts": dict(routes), "selection": report,
           "rule": ("start policy profiled with K samples at T=1.0 and the training verifiers, system rendered as plain text "
                    "(chat_template_content_format=string), fp32 SSM cache; keep 0 < pass_rate < 1; per-route caps; "
                    "rows cleaned by pivot_data/clean_candidates.py, no sampling seed"),
           "clean_manifest": os.path.realpath(a.clean_manifest),
           "start_policy": os.path.realpath(a.start_policy)}
    json.dump(man, open(os.path.join(a.out, "MANIFEST.json"), "w"), indent=1)
    lines = ["# %s bundle (cleaned rows chosen by the start policy's pass rate under eval-style rendering)" % a.bundle_name, "",
             "| route key | agent | profiled | all-wrong | mixed | all-right | train rows |", "|---|---|---|---|---|---|---|"]
    for k, v in report.items():
        h = v["pass_hist"]
        lines.append("| %s | %s | %d | %d | %d | %d | %d |" % (k, (v["agent"] or "")[:48], v["profiled"], h.get("0", 0), v["mixed"], h.get("1", 0), v["train"]))
    lines += ["", "train rows %d (P512 -> %.1f steps), val rows %d" % (len(train), len(train) / 512.0, len(val))]
    open(os.path.join(a.out, "BUNDLE_REPORT.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
