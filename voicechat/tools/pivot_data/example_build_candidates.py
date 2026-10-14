#!/usr/bin/env python3
"""EXAMPLE data recipe (adapt it): candidate single-step (pivot) RL rows grouped by the behaviour they target, in the
samples layout of scripts/profile_passrate (routes.json + <route_key>/sample.jsonl), so the start policy's per-prompt pass rate can
be measured (K samples) before rows enter the bundle. Input pools default to read-only reference data on cs-oci
(NVC_REF_ROOT); run clean_candidates.py on the output before profiling.

Route keys (agent = Gym agent block in config/gym-services-static-v8.yaml):
  bfcl_mt_call   BFCL-style multi-turn, gold = one call            multiturn_single_call_... agent   (no-call turns)
  bfcl_mt_batch  BFCL-style multi-turn, gold = several calls       parallel_tool_calls_... agent
  bfcl_mt_msg    BFCL-style multi-turn, gold = text (turn done /   hard_no_call_... agent             (stop when done)
                 ask for a missing value / function missing)
  nocall         v8 hard_no_call rows                              hard_no_call_... agent             (irrelevance)
  nearmiss       v8 irrelevance_near_miss rows                     irrelevance_near_miss_... agent    (live_irrelevance)
  single_step    v8 single_step rows                               single_step_... agent              (live_multiple values)
  parallel       v8 parallel rows (anchor)                         parallel_tool_calls_... agent
  tau2_write     Qwen3.8 AReaL retail/airline pivots, gold = write single_step_... agent              (retail / airline)
  telecom        v8 telecom_teacher_pivot rows (anchor)            telecom_teacher_pivot_... agent
  tau_mms, ctu   v8 tau_mms / ctu pivots (helped retail in v7)     their agents
BFCL-style rows come from the token-checked SFT records sft/data/bfcl-mt-v1 (passing episodes, error-call steps already
removed; new tasks on the gorilla backends, contamination-filtered). v8-derived rows exclude data/difficulty/easy_uuids.txt.
Usage: example_build_candidates.py --out DIR [--seed SALT] [--cap key=N ...]
"""
import argparse
import collections
import hashlib
import json
import os
import random
import uuid as uuidlib

P = os.environ.get("NVC_REF_ROOT", "/lustre/fsw/portfolios/llmservice/users/cchen1/NemotronVC_rl")  # read-only reference pools on cs-oci
AG = {
    "single": "single_step_tool_use_with_argument_comparison_agent",
    "multiturn": "multiturn_single_call_single_step_tool_use_with_argument_comparison_agent",
    "parallel": "parallel_tool_calls_single_step_tool_use_with_argument_comparison_agent",
    "nocall": "hard_no_call_single_step_tool_use_with_argument_comparison_agent",
    "nearmiss": "irrelevance_near_miss_single_step_tool_use_with_argument_comparison_agent",
    "telecom": "telecom_teacher_pivot_single_step_tool_use_with_argument_comparison_agent",
    "tau_mms": "tau_mms_pivot_single_step_tool_use_with_argument_comparison_agent",
    "ctu": "ctu_pivot_single_step_tool_use_with_argument_comparison_agent",
}
CAPS = {"bfcl_mt_call": 5000, "bfcl_mt_batch": 2500, "bfcl_mt_msg": 1000, "nocall": 2500, "nearmiss": 1500, "single_step": 3000,
        "parallel": 1500, "tau2_write": 7000, "telecom": 2000, "tau_mms": 1500, "ctu": 1500}
BENCH = {"bfcl_mt_call": "BFCL multi-turn: act instead of ending the turn in text", "bfcl_mt_batch": "BFCL multi-turn: several calls",
         "bfcl_mt_msg": "BFCL multi-turn: stop / ask only when needed", "nocall": "BFCL irrelevance (hard no-call)", "nearmiss": "BFCL live_irrelevance (near-miss tools)",
         "single_step": "BFCL live_multiple / live_simple values", "parallel": "BFCL parallel (anchor)",
         "tau2_write": "tau2 retail / airline writes (incl. right after the user's yes)", "telecom": "tau2 telecom (anchor)",
         "tau_mms": "tau2-like domains", "ctu": "tau2-like domains"}
WRITE_SKIP = ("get_", "find_", "list_", "search_", "calculate", "think", "transfer")


def chat_tools_to_responses(tools):
    out = []
    for t in tools or []:
        f = t.get("function", t)
        out.append({"type": "function", "name": f["name"], "description": f.get("description", ""),
                    "parameters": f.get("parameters", {"type": "object", "properties": {}}), "strict": False})
    return out


def chat_to_input(msgs):
    items = []
    for m in msgs:
        r = m["role"]
        if r == "user":
            items.append({"role": "user", "content": m.get("content") or ""})
        elif r == "system":
            items.append({"role": "system", "content": m.get("content") or ""})
        elif r == "assistant":
            if m.get("content"):
                items.append({"role": "assistant", "content": m["content"]})
            for c in m.get("tool_calls") or []:
                a = c["function"]["arguments"]
                items.append({"type": "function_call", "call_id": c["id"], "name": c["function"]["name"],
                              "arguments": a if isinstance(a, str) else json.dumps(a, ensure_ascii=False)})
        elif r == "tool":
            items.append({"type": "function_call_output", "call_id": m.get("tool_call_id"), "output": str(m.get("content"))})
    return items


def target_action(m):
    calls = m.get("tool_calls") or []
    fc = [{"type": "function_call", "name": c["function"]["name"],
           "arguments": c["function"]["arguments"] if isinstance(c["function"]["arguments"], str)
           else json.dumps(c["function"]["arguments"], ensure_ascii=False)} for c in calls]
    if not fc:
        return {"type": "message", "content": m.get("content") or ""}
    return fc[0] if len(fc) == 1 else {"type": "function_call_batch", "calls": fc}


def stable(key, salt):
    return int(hashlib.sha256(("%s|%s" % (salt, key)).encode()).hexdigest()[:12], 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", default="pivot-candidates")
    ap.add_argument("--cap", action="append", default=[])
    ap.add_argument("--bfcl-mt", default=P + "/sft/data/bfcl-mt-v1")
    ap.add_argument("--v8", default=P + "/data/static-v8/train.jsonl")
    ap.add_argument("--areal", default=P + "/data/tau2-src/pools/areal_teacher_qwen38_k8.jsonl")
    ap.add_argument("--easy", default=P + "/data/difficulty/easy_uuids.txt")
    a = ap.parse_args()
    caps = dict(CAPS)
    for c in a.cap:
        k, v = c.split("=")
        caps[k] = int(v)
    easy = set(x.strip() for x in open(a.easy) if x.strip())
    pools = collections.defaultdict(list)
    st = collections.Counter()

    # BFCL-style multi-turn pivots from the checked SFT records
    with open(os.path.join(a.bfcl_mt, "train.jsonl")) as fd, open(os.path.join(a.bfcl_mt, "train.meta.jsonl")) as fm:
        for line, ml in zip(fd, fm):
            r, m = json.loads(line), json.loads(ml)
            tgt = r["messages"][-1]
            act = target_action(tgt)
            key = {"function_call": "bfcl_mt_call", "function_call_batch": "bfcl_mt_batch", "message": "bfcl_mt_msg"}[act["type"]]
            agent = {"bfcl_mt_call": AG["multiturn"], "bfcl_mt_batch": AG["parallel"], "bfcl_mt_msg": AG["nocall"]}[key]
            uid = str(uuidlib.uuid5(uuidlib.NAMESPACE_URL, "bfcl-mt-v1/%s/%s" % (m["conv_id"], m["target_index"])))
            row = {"responses_create_params": {"input": chat_to_input(r["messages"][:-1]), "tools": chat_tools_to_responses(r["tools"]),
                                               "parallel_tool_calls": True, "max_output_tokens": 2048},
                   "expected_action": act, "agent_ref": {"type": "responses_api_agents", "name": agent}, "uuid": uid,
                   "dataset": "bfcl_mt_gen_v1@" + m.get("route", ""),
                   "info": {"route": m.get("route"), "split_key": m.get("split_key"), "target_index": m.get("target_index"),
                            "turn_users": m.get("turn_users"), "prev_role": r["messages"][-2]["role"]},
                   "provenance": {"source": "sft/data/bfcl-mt-v1", "conv_id": m["conv_id"]}}
            pools[key].append(row)
            st["bfcl_mt|" + key] += 1

    # v8 routes
    route_of = {AG["single"]: "single_step", AG["parallel"]: "parallel", AG["nocall"]: "nocall", AG["nearmiss"]: "nearmiss",
                AG["telecom"]: "telecom", AG["tau_mms"]: "tau_mms", AG["ctu"]: "ctu"}
    with open(a.v8) as f:
        for line in f:
            r = json.loads(line)
            key = route_of.get(r["agent_ref"]["name"])
            if not key:
                continue
            if r.get("uuid") in easy:
                st["v8_easy_excluded|" + key] += 1
                continue
            if key == "single_step" and r["expected_action"].get("type") == "message":
                continue
            pools[key].append(r)
            st["v8|" + key] += 1

    # tau2 retail / airline write pivots (Qwen3.8 on AReaL tasks)
    with open(a.areal) as f:
        for line in f:
            r = json.loads(line)
            ea = r["expected_action"]
            if ea.get("type") != "function_call" or ea.get("name", "").startswith(WRITE_SKIP):
                continue
            names = {t.get("name") for t in r["responses_create_params"].get("tools") or []}
            dom = "retail" if "exchange_delivered_order_items" in names else "airline" if "book_reservation" in names else "other"
            r = dict(r, info=dict(r.get("info") or {}, domain=dom))
            pools["tau2_write"].append(r)
            st["areal_write|" + dom] += 1

    os.makedirs(a.out, exist_ok=True)
    routes = []
    for prio, key in enumerate(["bfcl_mt_call", "bfcl_mt_msg", "nocall", "nearmiss", "tau2_write", "single_step", "bfcl_mt_batch",
                                "telecom", "parallel", "tau_mms", "ctu"], 1):
        rows = pools[key]
        seen, uniq = set(), []
        for r in rows:
            u = r.get("uuid") or hashlib.sha256(json.dumps(r["responses_create_params"], sort_keys=True).encode()).hexdigest()
            if u not in seen:
                seen.add(u)
                uniq.append(r)
        uniq.sort(key=lambda r: stable(r.get("uuid"), a.seed + key))
        pick = uniq[:caps[key]]
        d = os.path.join(a.out, key)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "sample.jsonl"), "w") as f:
            for r in pick:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        agent = collections.Counter(r["agent_ref"]["name"] for r in pick).most_common(1)[0][0] if pick else ""
        routes.append({"route_key": key, "agent": agent, "route_file": agent, "sample_path": os.path.realpath(os.path.join(d, "sample.jsonl")),
                       "n_prompts": len(pick), "max_output_tokens": 2048, "concurrency": 384, "priority": prio,
                       "est_output_tokens_per_rollout": 500, "steps": "1", "benchmark": BENCH[key],
                       "pool_unique": len(uniq)})
        st["picked|" + key] = len(pick)
    json.dump(routes, open(os.path.join(a.out, "routes.json"), "w"), indent=1)
    json.dump({"stats": dict(sorted(st.items())), "caps": caps, "seed": a.seed}, open(os.path.join(a.out, "SAMPLE_MANIFEST.json"), "w"), indent=1)
    print(json.dumps(dict(sorted(st.items())), indent=1))


if __name__ == "__main__":
    main()
