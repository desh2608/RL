#!/usr/bin/env python3
"""Write NEW BFCL-style multi-turn tasks with Qwen3.8 as the task writer, grounded turn by turn on the real gorilla
backends (data/bfcl-src/gorilla-bfcl-v1.3).

Per task: pick a class combo and turn count from the BFCL v3 multi_turn_base mix (common.COMBOS / TURNS); the writer
(1) invents a fresh initial_config from the class schema (DEFAULT_STATE + _load_scenario source, never test states),
(2) writes each turn as {"user": message, "calls": [python-call strings]}; the calls are executed on the backend and the
results are shown to the writer before the next turn (so ids created by earlier calls are real); a turn whose calls
error is retried (state rebuilt by replaying accepted turns), max 2 retries, else the task is dropped;
(3) for one turn whose call argument value appears verbatim in the user message, writes a vague version without the
value + the user's follow-up that supplies it (used for the miss_param variant).
Output: OUT/tasks.jsonl (append; resumable by tid) and OUT/gen_stats.json.
Usage: gen_tasks.py --out DIR --n 2000 [--base-url http://127.0.0.1:18200/v1] [--workers 64] [--seed bfclmt-v1]
"""
import argparse
import collections
import concurrent.futures as cf
import hashlib
import json
import os
import re
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

from openai import OpenAI  # noqa: E402

SYSTEM = """You write evaluation-style tasks for a function-calling assistant that works with stateful Python backends.
A task is a short multi-turn session: a user sends one message per turn; for each turn you also give the exact function
calls a careful assistant must make. Your calls are executed on the real backend and you see the results.

Rules for user messages:
- Sound like a real person typing to an assistant (vary tone and length; some terse, some chatty). Never mention function
  names, parameter names or code.
- Each turn needs 1-4 function calls. Later turns often build on earlier ones ("that file", "the booking you just made").
- Every argument value the calls need must be stated by the user, present in the initial state the assistant can look up
  with a function, or returned by an earlier call. Never require the assistant to guess.
- Only ask for things the listed functions can do.
Rules for calls:
- Python call strings with keyword arguments and literal values, e.g. "mv(source='a.txt', destination='docs')".
- Bare function names exactly as listed (no class prefix). Only listed functions.
- The minimal correct sequence: include a lookup/read call only when the user asks for that information or its output is
  needed for a later argument. Respect backend state (e.g. the file system has a current directory; use cd as needed).
Reply with JSON only, no markdown fences."""


def compact_docs(classes):
    out = []
    for c in classes:
        for d in C.func_docs(c):
            out.append({"class": c, "name": d["name"], "description": d.get("description", "")[:400],
                        "parameters": d.get("parameters", {})})
    return json.dumps(out, ensure_ascii=False)


def parse_json(text):
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j <= i:
        raise ValueError("no JSON object")
    body = t[i:j + 1]
    try:
        return json.loads(body)
    except ValueError:
        import ast  # writers sometimes emit Python literals (None / True / single quotes)
        v = ast.literal_eval(body)
        if not isinstance(v, dict):
            raise
        return json.loads(json.dumps(v, default=str))


def pick(seed, i, table):
    h = int(hashlib.sha256(("%s|%d|%s" % (seed, i, len(table))).encode()).hexdigest()[:12], 16)
    total = sum(w for _, w in table)
    r = h % total
    for v, w in table:
        if r < w:
            return v
        r -= w
    return table[-1][0]


class Writer:
    def __init__(self, a):
        self.a = a
        self.client = OpenAI(base_url=a.base_url, api_key="local", timeout=900, max_retries=2)
        self.stats = collections.Counter()
        self.lock = threading.Lock()

    def chat(self, msgs, max_tokens=8192):
        r = self.client.chat.completions.create(
            model=self.a.model, messages=msgs, max_tokens=max_tokens, temperature=self.a.temperature, top_p=0.95,
            extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": True,
                                                              "reasoning_effort": self.a.reasoning_effort}})
        return r.choices[0].message.content or ""

    def bump(self, k, n=1):
        with self.lock:
            self.stats[k] += n

    def replay(self, tag, tid, cfg, classes, turns):
        C.reset_instances(tag, tid, classes)
        C.run_calls([], cfg, classes, tag, tid)
        for t in turns:
            C.run_calls(t["calls"], cfg, classes, tag, tid)

    def task(self, i):
        a = self.a
        tid = "g%06d" % i
        classes = list(pick(a.seed, i, C.COMBOS))
        n_turns = pick(a.seed + "turns", i, C.TURNS)
        tag = "writer"
        schema = "\n\n".join("## %s state schema\n%s" % (c, C.class_schema_source(c)) for c in classes if c not in C.STATELESS)
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": (
                    "Backends: %s.\nFunctions (JSON):\n%s\n\n%s\n\nStep 1. Invent a fresh, realistic initial state for the "
                    "stateful backends (%s) that follows the schema exactly (same keys and value types). Use new names, "
                    "contents and numbers of your own; make it rich enough for a %d-turn session (several items, not just "
                    "one). Reply {\"initial_config\": {<ClassName>: {...}}}." % (
                        ", ".join(classes), compact_docs(classes), schema,
                        ", ".join(c for c in classes if c not in C.STATELESS) or "none", n_turns))}]
        try:
            cfg = parse_json(self.chat(msgs)).get("initial_config", {})
            cfg = {c: cfg[c] for c in classes if c in cfg and c not in C.STATELESS}
            if any(c not in cfg for c in classes if c not in C.STATELESS):
                self.bump("drop_config_missing_class")
                return None
            C.reset_instances(tag, tid, classes)
            C.run_calls([], cfg, classes, tag, tid)
        except Exception as e:
            self.bump("drop_config_" + type(e).__name__)
            return None
        msgs.append({"role": "assistant", "content": json.dumps({"initial_config": cfg}, ensure_ascii=False)})
        turns = []
        for k in range(n_turns):
            ask = ("Turn %d of %d. Write the user's message and the exact calls. Reply {\"user\": \"...\", \"calls\": "
                   "[\"...\"]}." % (k + 1, n_turns))
            msgs.append({"role": "user", "content": ask})
            ok = False
            for attempt in range(3):
                try:
                    t = parse_json(self.chat(msgs, max_tokens=6144))
                    user, calls = str(t["user"]).strip(), [str(x) for x in t["calls"]]
                    if not user or not (1 <= len(calls) <= 6) or any(C.call_name(x) is None for x in calls):
                        raise ValueError("bad turn shape")
                    self.replay(tag, tid, cfg, classes, turns)
                    results = C.run_calls(calls, cfg, classes, tag, tid)
                    bad = [(c, r) for c, r in zip(calls, results) if C.is_error(r)]
                    if bad:
                        self.bump("turn_retry_error")
                        msgs.append({"role": "assistant", "content": json.dumps(t, ensure_ascii=False)})
                        msgs.append({"role": "user", "content": "These calls failed: %s\nThe state was rolled back. Rewrite "
                                     "this turn (message and calls) so every call succeeds. Same JSON format." %
                                     json.dumps(bad, ensure_ascii=False)[:3000]})
                        continue
                    msgs.append({"role": "assistant", "content": json.dumps(t, ensure_ascii=False)})
                    msgs.append({"role": "user", "content": "Executed. Results:\n%s" % json.dumps(
                        list(zip(calls, results)), ensure_ascii=False)[:6000]})
                    turns.append({"user": user, "calls": calls, "results": results})
                    ok = True
                    break
                except Exception as e:
                    self.bump("turn_retry_" + type(e).__name__)
                    msgs.append({"role": "user", "content": "Invalid reply (%s). Reply with the JSON object only." % e})
            if not ok:
                self.bump("drop_turn_failed")
                C.reset_instances(tag, tid, classes)
                return None
        C.reset_instances(tag, tid, classes)
        task = {"tid": tid, "involved_classes": classes, "initial_config": cfg, "turns": turns}
        mp = self.miss_param(task)
        if mp:
            task["miss_param"] = mp
        self.bump("tasks")
        return task

    def miss_param(self, task):
        """Pick a turn whose call value is stated verbatim in the message; ask for a vague version + follow-up."""
        cands = []
        for ti, t in enumerate(task["turns"]):
            for call in t["calls"]:
                for k, v in C.call_kwargs(call).items():
                    sv = str(v)
                    if isinstance(v, (str, int, float)) and not isinstance(v, bool) and len(sv) >= 3 \
                            and sv.lower() in t["user"].lower():
                        cands.append((ti, k, sv))
        if not cands:
            self.bump("miss_param_no_candidate")
            return None
        ti, key, val = cands[int(hashlib.sha256(task["tid"].encode()).hexdigest()[:8], 16) % len(cands)]
        prompt = ("A user message in a tool-using session:\n%s\n\nThe assistant needs the value %r (for argument %r). "
                  "Rewrite the message so it no longer gives that value and the assistant cannot work it out from the "
                  "conversation or by looking things up (the request must be genuinely underspecified, so a careful "
                  "assistant asks). Keep everything else. Then write the user's short follow-up reply that gives the value "
                  "after the assistant asks. Reply {\"vague\": \"...\", \"followup\": \"...\"}." % (
                      task["turns"][ti]["user"], val, key))
        try:
            r = parse_json(self.chat([{"role": "user", "content": prompt}], max_tokens=4096))
            vague, fu = str(r["vague"]).strip(), str(r["followup"]).strip()
        except Exception:
            self.bump("miss_param_parse_fail")
            return None
        if not vague or val.lower() in vague.lower() or val.lower() not in fu.lower():
            self.bump("miss_param_invalid")
            return None
        self.bump("miss_param_ok")
        return {"turn": ti, "vague": vague, "followup": fu, "value": val, "arg": key}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--base-url", default="http://127.0.0.1:18200/v1")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--reasoning-effort", default="medium")
    ap.add_argument("--seed", default="bfclmt-v1")
    ap.add_argument("--deadline", type=float, default=0, help="unix time to stop submitting new tasks")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "tasks.jsonl")
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["tid"] for l in open(path) if l.strip()}
    todo = [i for i in range(a.start, a.start + a.n) if "g%06d" % i not in done]
    w = Writer(a)
    t0 = time.time()
    with open(path, "a") as f, cf.ThreadPoolExecutor(a.workers) as ex:
        futs = {}
        it = iter(todo)
        for i in it:
            futs[ex.submit(w.task, i)] = i
            if len(futs) >= a.workers * 2:
                break
        while futs:
            fin, _ = cf.wait(futs, return_when=cf.FIRST_COMPLETED)
            for fu in fin:
                futs.pop(fu)
                try:
                    task = fu.result()
                except Exception as e:
                    w.bump("crash_" + type(e).__name__)
                    task = None
                if task:
                    f.write(json.dumps(task, ensure_ascii=False) + "\n")
                    f.flush()
                if not a.deadline or time.time() < a.deadline:
                    nxt = next(it, None)
                    if nxt is not None:
                        futs[ex.submit(w.task, nxt)] = nxt
            if w.stats["tasks"] and w.stats["tasks"] % 50 == 0:
                print("[gen] %d tasks in %.0fs %s" % (w.stats["tasks"], time.time() - t0, dict(w.stats)), flush=True)
    json.dump(dict(w.stats), open(os.path.join(a.out, "gen_stats.json"), "w"), indent=1)
    print("[gen] done", dict(w.stats), flush=True)


if __name__ == "__main__":
    main()
