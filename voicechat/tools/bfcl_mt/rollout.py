#!/usr/bin/env python3
"""Run Qwen3.8 as the agent on generated BFCL-format multi-turn entries with the official gorilla v1.3 FC loop
(OpenAICompletionsHandler.inference_multi_turn_FC) and score each episode with the official multi_turn_checker.

Request semantics follow the eval (NemotronVC_eval bfcl runtime/generate.py LocalVLLMHandler): no system prompt, tools +
tool_choice auto, chat history without reasoning, one tool message per call id, max_tokens per step = 2048 (eval
budget, incl. reasoning), 20-step force-quit inside a turn. Sampling: --temperature (default 0.6), top_p 0.95, top_k 20.
Every episode (pass or fail) is written to OUT/rollouts.jsonl with the final message list, the tools lists, and per step
{prefix length, tools index, reasoning, finish_reason}; to_sft.py turns passing episodes into per-step records.
Usage: rollout.py --entries OUT/entries.jsonl --answers OUT/answers.jsonl --out OUT [--k 1] [--workers 128]
"""
import argparse
import collections
import concurrent.futures as cf
import copy
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

os.environ.setdefault("OPENAI_API_KEY", "local-vllm-not-a-credential")
from bfcl_eval.model_handler import base_handler as BH  # noqa: E402
from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler  # noqa: E402
from bfcl_eval.eval_checker.multi_turn_eval import multi_turn_checker as MTC  # noqa: E402
from openai import BadRequestError, OpenAI  # noqa: E402

C.install_safe_exec(BH, MTC)


def msg_dict(m):
    src = m if isinstance(m, dict) else m.model_dump(exclude_none=True)
    out = {"role": src.get("role", "assistant"), "content": src.get("content")}
    tcs = src.get("tool_calls")
    if tcs:
        out["tool_calls"] = [{"id": t["id"], "type": "function",
                              "function": {"name": t["function"]["name"], "arguments": t["function"]["arguments"]}}
                             for t in (x if isinstance(x, dict) else x.model_dump(exclude_none=True) for x in tcs)]
    return out


class Handler(OpenAICompletionsHandler):
    def __init__(self, a, k):
        super().__init__("%s-k%d-FC" % (a.model, k), a.temperature)
        self.is_fc_model = True
        self.a = a
        self.client = OpenAI(api_key="local", base_url=a.base_url, timeout=600, max_retries=2)
        self.episodes = {}
        self.lock = threading.Lock()

    def _pre_query_processing_FC(self, inference_data, test_entry):
        inference_data["message"] = []
        with self.lock:
            self.episodes[test_entry["id"]] = {"data": inference_data, "steps": [], "tools_list": []}
        inference_data["_eid"] = test_entry["id"]
        return inference_data

    def _query_FC(self, inference_data):
        ep = self.episodes[inference_data["_eid"]]
        messages = [m if isinstance(m, dict) else msg_dict(m) for m in inference_data["message"]]
        inference_data["message"] = messages
        tools = inference_data["tools"]
        if not ep["tools_list"] or ep["tools_list"][-1] != tools:
            ep["tools_list"].append(copy.deepcopy(tools))
        kw = {"messages": messages, "model": self.a.model, "temperature": self.a.temperature, "top_p": 0.95,
              "max_tokens": self.a.max_tokens,
              "extra_body": {"top_k": 20, "chat_template_kwargs": {"enable_thinking": True,
                                                                   "reasoning_effort": self.a.reasoning_effort}}}
        if tools:
            kw["tools"] = tools
            kw["tool_choice"] = "auto"
        t0 = time.monotonic()
        mt = self.a.max_tokens
        while True:
            kw["max_tokens"] = mt
            try:
                resp = self.client.chat.completions.create(**kw)
                break
            except BadRequestError as e:
                if "maximum context length" not in str(e).lower() or mt <= 1:
                    raise
                mt = max(1, mt // 2)
        ch = resp.choices[0]
        src = ch.message.model_dump(exclude_none=True)
        ep["steps"].append({"n_prefix": len(messages), "tools_idx": len(ep["tools_list"]) - 1,
                            "reasoning": src.get("reasoning_content") or src.get("reasoning") or "",
                            "finish_reason": ch.finish_reason, "max_tokens": mt,
                            "out_tokens": int(getattr(resp.usage, "completion_tokens", 0) or 0)})
        return resp, time.monotonic() - t0

    def _parse_query_response_FC(self, api_response):
        m = api_response.choices[0].message
        tcs = list(getattr(m, "tool_calls", None) or [])
        if tcs:
            responses = [{c.function.name: c.function.arguments} for c in tcs]
            ids = [c.id for c in tcs]
        else:
            responses, ids = m.content or "", []
        u = api_response.usage
        return {"model_responses": responses, "model_responses_message_for_chat_history": msg_dict(m),
                "tool_call_ids": ids, "reasoning_content": "",
                "input_token": int(getattr(u, "prompt_tokens", 0) or 0),
                "output_token": int(getattr(u, "completion_tokens", 0) or 0)}

    def _add_reasoning_content_if_available_FC(self, api_response, response_data):
        response_data["model_responses_message_for_chat_history"] = msg_dict(api_response.choices[0].message)


def prepare(entry):
    """gorilla process_multi_turn_test_case: attach function docs, move held-out ones into missed_function."""
    e = copy.deepcopy(entry)
    e["function"] = []
    for c in e["involved_classes"]:
        e["function"].extend(C.func_docs(c))
    if "missed_function" in e:
        for t, names in list(e["missed_function"].items()):
            e["missed_function"][t] = []
            for n in names:
                for i, d in enumerate(e["function"]):
                    if d["name"] == n:
                        e["missed_function"][t].append(d)
                        e["function"].pop(i)
                        break
    return e


def score(h, entry, gt, result, check_name):
    if len(result) != len(gt):
        return False, "multi_turn:force_terminated"
    decoded = []
    for turn in result:
        td = []
        for step in turn:
            try:
                d = h.decode_execute(step)
                if C.MTU.is_empty_execute_response(d):
                    continue
                td.append(d)
            except Exception:
                continue
        decoded.append(td)
    te = {k: v for k, v in entry.items() if k != "function"}
    r = MTC.multi_turn_checker(decoded, gt, te, entry["id"].rsplit("_", 1)[0], check_name)
    return bool(r.get("valid")), r.get("error_type")


def clear(eid):
    ns = vars(C.MTU)
    for name in [n for n in list(ns) if n.endswith("_instance") and ("_%s_" % eid) in n]:
        ns.pop(name, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entries", required=True)
    ap.add_argument("--answers", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--workers", type=int, default=128)
    ap.add_argument("--base-url", default="http://127.0.0.1:18200/v1")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--reasoning-effort", default="medium")
    ap.add_argument("--deadline", type=float, default=0)
    a = ap.parse_args()
    ents = C.read_jsonl(a.entries)
    gts = {x["id"]: x["ground_truth"] for x in C.read_jsonl(a.answers)}
    path = os.path.join(a.out, "rollouts.jsonl")
    done = set()
    if os.path.exists(path):
        done = {(r["id"], r["k"]) for r in map(json.loads, open(path)) if r}
    handlers = [Handler(a, k) for k in range(a.k)]
    jobs = [(e, k) for k in range(a.k) for e in ents if (e["id"], k) not in done]
    st = collections.Counter()
    lock = threading.Lock()

    def run(e, k):
        if a.deadline and time.time() > a.deadline:
            return None
        h = handlers[k]
        eid = "%sk%d" % (e["id"], k)  # unique per sample: gorilla keys backend instances by entry id
        entry = prepare(e)
        entry["id"] = eid
        clear(eid)
        try:
            result, meta = h.inference_multi_turn_FC(copy.deepcopy(entry), include_input_log=False, exclude_state_log=True)
            ok, err = score(h, prepare(dict(e, id=eid)), gts[e["id"]], result, "chk_k%d" % k)
        except Exception as ex:
            ok, err, result = False, "crash:%s:%s" % (type(ex).__name__, str(ex)[:200]), None
        ep = h.episodes.pop(eid, {"data": {"message": []}, "steps": [], "tools_list": []})
        clear(eid)
        return {"id": e["id"], "k": k, "cat": e["id"].rsplit("_", 1)[0].replace("multi_turn_", ""), "valid": ok,
                "error_type": err, "messages": ep["data"].get("message", []), "tools_list": ep["tools_list"],
                "steps": ep["steps"], "turns": len(e["question"])}

    t0 = time.time()
    with open(path, "a") as f, cf.ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(run, e, k) for e, k in jobs]
        for fu in cf.as_completed(futs):
            r = fu.result()
            if r is None:
                st["skipped_deadline"] += 1
                continue
            with lock:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                st["%s|%s" % (r["cat"], "pass" if r["valid"] else (r["error_type"] or "fail").split(":")[0] + ":" +
                              (r["error_type"] or "").split(":")[-1][:40])] += 1
                n = sum(v for kk, v in st.items() if "|" in kk)
                if n % 200 == 0:
                    print("[rollout] %d episodes %.0fs" % (n, time.time() - t0), flush=True)
    json.dump(dict(st), open(os.path.join(a.out, "rollout_stats.json"), "w"), indent=1)
    print(json.dumps(dict(sorted(st.items())), indent=1))


if __name__ == "__main__":
    main()
