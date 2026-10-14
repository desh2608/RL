#!/usr/bin/env python3
"""Turn written tasks into BFCL-format entries + answers for the four multi-turn categories, after a contamination filter.

  base          as written
  long_context  same entry, id prefix multi_turn_long_context_ (gorilla backends then load extra filler state)
  miss_func     a turn t>=1 that introduces a function f no earlier turn used: f is held out of the tools,
                ground truth for turn t is [], an empty question turn t+1 re-adds f (gorilla's fixed user prompt) and
                carries turn t's calls (same construction as BFCL miss_func)
  miss_param    the writer's vague rewrite of turn t (ground truth []) + the user's follow-up as turn t+1 carrying the calls
Contamination: a task is dropped if any of its user messages has word-3-gram Jaccard >= --max-sim with any BFCL v3
multi-turn test user message (common.ContaminationIndex).
Usage: make_variants.py --tasks OUT/tasks.jsonl --out OUT [--max-sim 0.5]
Writes OUT/entries.jsonl, OUT/answers.jsonl, OUT/variants_stats.json.
"""
import argparse
import collections
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402


def entry(cat, task, question, gt, missed=None):
    e = {"id": "multi_turn_%s_%s" % (cat, task["tid"]), "question": question, "initial_config": task["initial_config"],
         "path": [], "involved_classes": task["involved_classes"]}
    if missed:
        e["missed_function"] = missed
    return e, {"id": e["id"], "ground_truth": gt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-sim", type=float, default=0.5)
    a = ap.parse_args()
    idx = C.ContaminationIndex()
    st = collections.Counter()
    ents, ans = [], []
    sims = []
    for task in C.read_jsonl(a.tasks):
        texts = [t["user"] for t in task["turns"]]
        if task.get("miss_param"):
            texts += [task["miss_param"]["vague"], task["miss_param"]["followup"]]
        sim = max(idx.max_sim(x) for x in texts)
        sims.append(sim)
        if sim >= a.max_sim:
            st["drop_contamination"] += 1
            continue
        st["tasks_kept"] += 1
        q = [[{"role": "user", "content": t["user"]}] for t in task["turns"]]
        gt = [list(t["calls"]) for t in task["turns"]]
        for cat in ("base", "long_context"):
            e, g = entry(cat, task, q, gt)
            ents.append(e), ans.append(g)
            st[cat] += 1
        # miss_func
        used = set()
        cands = []
        for ti, t in enumerate(task["turns"]):
            names = [C.call_name(c) for c in t["calls"]]
            fresh = [n for n in dict.fromkeys(names) if n not in used]
            if ti >= 1 and fresh:
                cands.append((ti, fresh[0]))  # first function this turn introduces; the whole turn waits for it
            used |= set(names)
        if cands:
            ti, f = cands[int(hashlib.sha256(task["tid"].encode()).hexdigest()[:8], 16) % len(cands)]
            q2 = q[:ti + 1] + [[]] + q[ti + 1:]
            gt2 = gt[:ti] + [[]] + [gt[ti]] + gt[ti + 1:]
            e, g = entry("miss_func", task, q2, gt2, missed={str(ti + 1): [f]})
            ents.append(e), ans.append(g)
            st["miss_func"] += 1
        else:
            st["miss_func_no_candidate"] += 1
        mp = task.get("miss_param")
        if mp:
            ti = mp["turn"]
            q3 = q[:ti] + [[{"role": "user", "content": mp["vague"]}], [{"role": "user", "content": mp["followup"]}]] + q[ti + 1:]
            gt3 = gt[:ti] + [[]] + [gt[ti]] + gt[ti + 1:]
            e, g = entry("miss_param", task, q3, gt3)
            ents.append(e), ans.append(g)
            st["miss_param"] += 1
    C.write_jsonl(os.path.join(a.out, "entries.jsonl"), ents)
    C.write_jsonl(os.path.join(a.out, "answers.jsonl"), ans)
    sims.sort()
    st["sim_p50"] = round(sims[len(sims) // 2], 3) if sims else 0
    st["sim_p99"] = round(sims[int(len(sims) * .99)], 3) if sims else 0
    json.dump(dict(st), open(os.path.join(a.out, "variants_stats.json"), "w"), indent=1)
    print(json.dumps(dict(st)))


if __name__ == "__main__":
    main()
