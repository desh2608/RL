#!/usr/bin/env python3
"""Clean single-step pivot candidate rows (route samples in the profile_passrate layout: routes.json + <route>/sample.jsonl)
so the trained prompt has the eval prompt shape (bugs B5, B6, B7 in voicechat/docs/BUGS_FIXED.md):

* history reasoning items (type "reasoning") removed: bfcl_v3 / tau2 clients never send reasoning back;
* "ENVIRONMENT REMINDER: You have N turns left ..." stripped from user messages and tool outputs (tau2 eval has none);
* trailing assistant text messages removed (Qwen3.8 said "let me process that now" in the same turn as the gold write
  call; rendered as a finished assistant turn they made the policy open a second assistant turn, which never happens in
  eval; tau2 policy: a turn is either a message or a tool call);
* rows whose prompt ends with a function_call without its output are dropped;
* parallel_tool_calls forced to true (with false, vLLM returns only the first call and the single-call verifier never
  sees the extra calls);
* no sampling seed anywhere (a metadata.extra_body seed makes all K samples of a prompt identical);
* duplicate prompts (after cleaning) dropped.
The system message is left as is; its rendering is fixed on the server side (chat_template_content_format=string).
Writes OUT/<route>/sample.jsonl, OUT/routes.json (sample paths rewritten) and OUT/CLEAN_MANIFEST.json, including for
every route whether cleaning left its rows unchanged (then an older profile of that route may be reused, see build_bundle.py).
Usage: clean_candidates.py --src SAMPLES_DIR --out CLEAN_SAMPLES_DIR
"""
import argparse
import collections
import copy
import hashlib
import json
import os
import re

REMINDER = re.compile(r"\s*ENVIRONMENT REMINDER: You have \d+ turns? left to complete the task\.\s*$")


def is_assistant_message(it):
    return it.get("role") == "assistant" and it.get("type") in (None, "message")


def strip_reminder(it, st):
    def fix(s):
        t = REMINDER.sub("", s)
        if t != s:
            st["reminders_stripped"] += 1
        return t
    if it.get("role") == "user":
        c = it.get("content")
        if isinstance(c, str):
            it["content"] = fix(c)
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    part["text"] = fix(part["text"])
    elif it.get("type") == "function_call_output" and isinstance(it.get("output"), str):
        it["output"] = fix(it["output"])


def clean(row, st):
    r = copy.deepcopy(row)
    p = r["responses_create_params"]
    items = []
    for it in p["input"]:
        if it.get("type") == "reasoning":
            st["reasoning_items_removed"] += 1
            continue
        strip_reminder(it, st)
        items.append(it)
    trailing = 0
    while items and is_assistant_message(items[-1]):
        items.pop()
        trailing += 1
    if trailing:
        st["rows_trailing_assistant_removed"] += 1
    if not items or items[-1].get("type") == "function_call" or items[-1].get("role") == "assistant":
        st["dropped_ends_with_call_or_empty"] += 1
        return None
    p["input"] = items
    if p.get("parallel_tool_calls") is not True:
        st["parallel_forced_true"] += 1
        p["parallel_tool_calls"] = True
    md = p.get("metadata")
    if isinstance(md, dict) and "extra_body" in md:
        md.pop("extra_body")
        st["extra_body_removed"] += 1
        if not md:
            p.pop("metadata")
    md = p.get("metadata") or {}
    assert "seed" not in p and "seed" not in md and "extra_body" not in md, "sampling seed left in row"
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert not os.path.exists(a.out), a.out
    routes = json.load(open(os.path.join(a.src, "routes.json")))
    out_routes, manifest = [], {}
    for rt in routes:
        key = rt["route_key"]
        st = collections.Counter()
        seen, kept = set(), []
        unchanged = 0
        for line in open(os.path.join(a.src, key, "sample.jsonl")):
            row = json.loads(line)
            st["rows_in"] += 1
            c = clean(row, st)
            if c is None:
                continue
            h = hashlib.sha256(json.dumps(c["responses_create_params"], sort_keys=True).encode()).hexdigest()
            if h in seen:
                st["duplicates_dropped"] += 1
                continue
            seen.add(h)
            unchanged += int(c == row)
            kept.append(c)
        d = os.path.join(a.out, key)
        os.makedirs(d)
        with open(os.path.join(d, "sample.jsonl"), "w") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        st["rows_out"] = len(kept)
        st["rows_unchanged"] = unchanged
        manifest[key] = dict(st, identical_to_src=(unchanged == len(kept) == st["rows_in"]))
        out_routes.append(dict(rt, sample_path=os.path.realpath(os.path.join(d, "sample.jsonl")), n_prompts=len(kept)))
    json.dump(out_routes, open(os.path.join(a.out, "routes.json"), "w"), indent=1)
    json.dump({"src": os.path.realpath(a.src), "routes": manifest}, open(os.path.join(a.out, "CLEAN_MANIFEST.json"), "w"), indent=1)
    for k, v in manifest.items():
        print(k, json.dumps(v))


if __name__ == "__main__":
    main()
