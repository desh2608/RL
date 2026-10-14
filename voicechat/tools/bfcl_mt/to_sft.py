#!/usr/bin/env python3
"""Per-step SFT records from passing BFCL-style multi-turn episodes (rollout.py output).

Kept: episodes the official multi_turn_checker passed; one episode per entry (first passing k); episodes with any step cut
by max_tokens or with more than --max-error-calls tool results that are errors are dropped; a step whose own calls got
an error result is not a target (later steps that recover from it are). One record per assistant step: history = request messages (no system prompt, assistant
history without reasoning, exactly what the agent saw), target = that step's assistant message + its reasoning; tools =
the tool list of that request (held-out functions absent before they are added). Steps with empty reasoning are skipped.
Output (work-dir layout used by the SFT mix builder): OUT/perturn.jsonl, perturn.meta.jsonl, convert_stats.json.
Usage: to_sft.py --rollouts DIR/rollouts.jsonl --out DIR/sft [--source qwen38-bfcl-mt-gen-v1]
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402


def norm(m, reasoning=None):
    role = m.get("role")
    if role == "assistant":
        out = {"role": "assistant", "content": m.get("content") or ""}
        if m.get("tool_calls"):
            out["tool_calls"] = []
            for t in m["tool_calls"]:
                args = t["function"]["arguments"]
                try:
                    args = json.loads(args) if isinstance(args, str) else args
                except Exception:
                    return None
                out["tool_calls"].append({"id": t["id"], "type": "function",
                                          "function": {"name": t["function"]["name"], "arguments": args}})
        if reasoning is not None:
            out["reasoning_content"] = reasoning
        return out
    if role == "tool":
        return {"role": "tool", "content": str(m.get("content")), "tool_call_id": m.get("tool_call_id")}
    return {"role": role, "content": m.get("content") or ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", default="qwen38-bfcl-mt-gen-v1")
    ap.add_argument("--max-error-calls", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    st = collections.Counter()
    seen = set()
    recs, metas = [], []
    for line in open(a.rollouts):
        r = json.loads(line)
        st["episodes"] += 1
        if not r["valid"]:
            st["fail|" + r["cat"]] += 1
            continue
        if r["id"] in seen:
            st["dup_pass"] += 1
            continue
        if any(s["finish_reason"] == "length" for s in r["steps"]):
            st["drop_truncated|" + r["cat"]] += 1
            continue
        msgs = r["messages"]
        err_ids = {m.get("tool_call_id") for m in msgs if m.get("role") == "tool" and C.is_error(m.get("content"))}
        if len(err_ids) > a.max_error_calls:
            st["drop_error_calls|" + r["cat"]] += 1
            continue
        seen.add(r["id"])
        st["pass|" + r["cat"]] += 1
        tid = r["id"].rsplit("_", 1)[1]
        hist = []
        bad = False
        for m in msgs:
            n = norm(m)
            if n is None:
                bad = True
                break
            hist.append(n)
        if bad:
            st["drop_bad_args"] += 1
            continue
        for s in r["steps"]:
            i = s["n_prefix"]
            if i >= len(msgs) or msgs[i].get("role") != "assistant":
                st["skip_step_misaligned"] += 1
                continue
            if not (s.get("reasoning") or "").strip():
                st["skip_step_no_reasoning"] += 1
                continue
            if any(t["id"] in err_ids for t in msgs[i].get("tool_calls") or []):
                st["skip_step_error_call"] += 1
                continue
            tgt = norm(msgs[i], reasoning=s["reasoning"])
            calls = len(tgt.get("tool_calls") or [])
            if not calls and not tgt["content"].strip():
                st["skip_step_empty"] += 1
                continue
            kind = "message" if not calls else ("function_call" if calls == 1 else "function_call_batch")
            recs.append({"messages": hist[:i] + [tgt], "tools": r["tools_list"][s["tools_idx"]]})
            metas.append({"source": a.source, "route": r["cat"], "conv_id": r["id"], "split_key": tid,
                          "target_index": i, "target_kind": kind, "n_calls": calls,
                          "turn_users": sum(1 for x in hist[:i] if x["role"] == "user")})
            st["records|" + r["cat"] + "|" + kind] += 1
    C.write_jsonl(os.path.join(a.out, "perturn.jsonl"), recs)
    C.write_jsonl(os.path.join(a.out, "perturn.meta.jsonl"), metas)
    json.dump({"source": a.source, "rollouts": os.path.realpath(a.rollouts), "stats": dict(sorted(st.items())),
               "teacher": "Qwen/Qwen3.8-27B (agent and task writer; reasoning_effort medium)",
               "env": "gorilla BFCL v1.3 multi-turn backends (Apache-2.0), new tasks only, contamination-filtered",
               "license": "teacher apache-2.0; gorilla apache-2.0"},
              open(os.path.join(a.out, "convert_stats.json"), "w"), indent=1)
    print(json.dumps(dict(sorted(st.items())), indent=1))


if __name__ == "__main__":
    main()
