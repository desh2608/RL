#!/usr/bin/env python3
"""Aggregate `gym eval run` rollouts into a per-route / per-prompt pass-rate profile.

Stdlib only; safe on the login node. Reads <out>/routes.json (copied from the sample dir by
launch_profile.sh; falls back to <out>/samples/routes.json) and, per route:
  <out>/rollouts/<route_key>/rollouts.jsonl                     one result per rollout
  <out>/rollouts/<route_key>/rollouts_failures.jsonl            Gym failure sidecar (transport / judge errors)
  <out>/rollouts/<route_key>/rollouts_materialized_inputs.jsonl the K-expanded inputs (for joins)
  <out>/timing.jsonl                                            wall time per route (launch_profile.sh)

Rollout result fields used (nemo_gym/rollout_collection.py:846-850, responses_converter.py:623-657):
  reward, response.usage.output_tokens, response.status ("incomplete" when finish_reason == length),
  response.incomplete_details.reason ("max_output_tokens"), response.output[*] (function_call items /
  message text), _ng_task_index, _ng_rollout_index, agent_ref.name, nvc_profile (echoed from the input row).

Writes <out>/per_prompt.jsonl, <out>/route_summary.json, <out>/keep_pass_0.2_0.8.jsonl, <out>/SUMMARY.md.
"""
import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

INVALID_TOOL_CALL_PATTERNS = ("<tool_call>", "</tool_call>", "<function_call>", "</function_call>")  # frozen config env.nemo_gym.invalid_tool_call_patterns
GLOBAL_CAP = 65536  # training max_new_tokens (runs/train-v1-20260928T023000Z/config.yaml policy.generation.max_new_tokens)


def read_jsonl(path: Path):
    if not path.is_file():
        return []
    rows = []
    with path.open("rb") as handle:
        for line_no, raw in enumerate(handle, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError as error:
                print(f"WARN {path}:{line_no}: {error}")
    return rows


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    idx = max(0, min(len(values) - 1, math.ceil(q * len(values)) - 1))
    return values[idx]


def text_of_output(response: dict) -> str:
    parts = []
    for item in (response or {}).get("output") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "message":
            content = item.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and isinstance(block.get("text"), str):
                        parts.append(block["text"])
    return "\n".join(parts)


def rollout_features(result: dict, cap: int) -> dict:
    response = result.get("response") if isinstance(result.get("response"), dict) else {}
    usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    out_tok = usage.get("output_tokens")
    out_tok = int(out_tok) if isinstance(out_tok, (int, float)) else None
    incomplete = response.get("status") == "incomplete"
    reason = (response.get("incomplete_details") or {}).get("reason") if isinstance(response.get("incomplete_details"), dict) else None
    output_items = response.get("output") or []
    n_function_calls = sum(1 for item in output_items if isinstance(item, dict) and item.get("type") == "function_call")
    text = text_of_output(response)
    invalid_tool_text = any(pattern in text for pattern in INVALID_TOOL_CALL_PATTERNS)
    reward = result.get("reward")
    reward = float(reward) if isinstance(reward, (int, float)) and not isinstance(reward, bool) else None
    return {
        "reward": reward,
        "output_tokens": out_tok,
        "cap_hit_route": bool(incomplete and reason == "max_output_tokens") or (out_tok is not None and out_tok >= cap),
        "cap_hit_65536": out_tok is not None and out_tok >= GLOBAL_CAP,
        "incomplete": incomplete,
        "incomplete_reason": reason,
        "n_function_calls": n_function_calls,
        "invalid_tool_call_text": invalid_tool_text,
        "empty_output": not text.strip() and n_function_calls == 0,
    }


def bin_of(pass_rate):
    if pass_rate is None:
        return "missing"
    if pass_rate <= 0:
        return "0"
    if pass_rate < 0.2:
        return "(0,0.2)"
    if pass_rate <= 0.8:
        return "[0.2,0.8]"
    if pass_rate < 1:
        return "(0.8,1)"
    return "1"


BINS = ["0", "(0,0.2)", "[0.2,0.8]", "(0.8,1)", "1", "missing"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="profile directory (contains rollouts/ and routes.json)")
    ap.add_argument("--keep-low", type=float, default=0.2)
    ap.add_argument("--keep-high", type=float, default=0.8)
    ap.add_argument("--pass-threshold", type=float, default=1.0 - 1e-9,
                    help="reward >= threshold counts as a pass (binary verifiers give exactly 0/1)")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    routes_path = out / "routes.json"
    if not routes_path.is_file():
        routes_path = out / "samples" / "routes.json"
    if not routes_path.is_file():
        raise SystemExit(f"routes.json not found under {out} (or {out}/samples)")
    routes = json.loads(routes_path.read_text())
    timing = {t["route_key"]: t for t in read_jsonl(out / "timing.jsonl") if "route_key" in t}
    runtime = json.loads((out / "RUNTIME.json").read_text()) if (out / "RUNTIME.json").is_file() else {}

    per_prompt_path = out / "per_prompt.jsonl"
    keep_path = out / f"keep_pass_{args.keep_low}_{args.keep_high}.jsonl"
    summaries = []
    with per_prompt_path.open("w") as per_prompt_file, keep_path.open("w") as keep_file:
        for route in sorted(routes, key=lambda r: r["priority"]):
            key = route["route_key"]
            cap = int(route["max_output_tokens"])
            route_dir = out / "rollouts" / key
            results = read_jsonl(route_dir / "rollouts.jsonl")
            failures = read_jsonl(route_dir / "rollouts_failures.jsonl")
            inputs = read_jsonl(route_dir / "rollouts_materialized_inputs.jsonl")
            if not results and not failures:
                summaries.append({"route_key": key, "agent": route["agent"], "status": "no rollouts on disk",
                                  "timing": timing.get(key)})
                continue
            task_meta = {}
            for row in inputs:
                task_idx = row.get("_ng_task_index")
                if task_idx is not None and task_idx not in task_meta:
                    task_meta[task_idx] = row.get("nvc_profile") or {}
            groups = defaultdict(list)
            failure_groups = Counter()
            for result in results:
                groups[result.get("_ng_task_index")].append(rollout_features(result, cap))
                if result.get("_ng_task_index") not in task_meta and isinstance(result.get("nvc_profile"), dict):
                    task_meta[result["_ng_task_index"]] = result["nvc_profile"]
            for failure in failures:
                failure_groups[failure.get("_ng_task_index")] += 1

            hist = Counter()
            hist_vs_new = Counter()
            zero_var = 0
            all_rewards, all_out, cap_route, cap_global, invalid_text, multi_call, no_call, empty, incomplete = [], [], 0, 0, 0, 0, 0, 0, 0
            n_rollouts = 0
            keep = 0
            for task_idx, feats in sorted(groups.items(), key=lambda kv: (kv[0] is None, kv[0])):
                rewards = [f["reward"] for f in feats if f["reward"] is not None]
                outs = [f["output_tokens"] for f in feats if f["output_tokens"] is not None]
                n = len(feats)
                n_rollouts += n
                pass_rate = (sum(1 for r in rewards if r >= args.pass_threshold) / len(rewards)) if rewards else None
                mean_reward = statistics.fmean(rewards) if rewards else None
                zero_variance = bool(rewards) and (max(rewards) - min(rewards) < 1e-9)
                zero_var += int(zero_variance)
                meta = task_meta.get(task_idx, {})
                expected_type = meta.get("expected_action_type")
                record = {
                    "route_key": key,
                    "agent": route["agent"],
                    "_ng_task_index": task_idx,
                    "route_file": meta.get("route_file"),
                    "source_ordinal_1based": meta.get("source_ordinal_1based"),
                    "raw_line_sha256": meta.get("raw_line_sha256"),
                    "n_rollouts": n,
                    "n_failures": failure_groups.get(task_idx, 0),
                    "pass_rate": pass_rate,
                    "mean_reward": mean_reward,
                    "rewards": [f["reward"] for f in feats],
                    "zero_variance": zero_variance,
                    "output_tokens_mean": statistics.fmean(outs) if outs else None,
                    "output_tokens_max": max(outs) if outs else None,
                    "cap_hits_route": sum(1 for f in feats if f["cap_hit_route"]),
                    "cap_hits_65536": sum(1 for f in feats if f["cap_hit_65536"]),
                    "invalid_tool_call_text": sum(1 for f in feats if f["invalid_tool_call_text"]),
                    "multi_function_call": sum(1 for f in feats if f["n_function_calls"] > 1),
                    "no_function_call": sum(1 for f in feats if f["n_function_calls"] == 0) if expected_type == "function_call" else None,
                    "historical_pass_rate": meta.get("historical_pass_rate"),
                    "expected_action_type": expected_type,
                }
                per_prompt_file.write(json.dumps(record) + "\n")
                hist[bin_of(pass_rate)] += 1
                hist_vs_new[(bin_of(meta.get("historical_pass_rate")), bin_of(pass_rate))] += 1
                if pass_rate is not None and args.keep_low <= pass_rate <= args.keep_high:
                    keep += 1
                    keep_file.write(json.dumps({k: record[k] for k in (
                        "route_key", "agent", "route_file", "source_ordinal_1based", "raw_line_sha256",
                        "pass_rate", "mean_reward", "n_rollouts", "output_tokens_mean")}) + "\n")
                all_rewards.extend(rewards)
                all_out.extend(outs)
                cap_route += record["cap_hits_route"]
                cap_global += record["cap_hits_65536"]
                invalid_text += record["invalid_tool_call_text"]
                multi_call += record["multi_function_call"]
                no_call += record["no_function_call"] or 0
                empty += sum(1 for f in feats if f["empty_output"])
                incomplete += sum(1 for f in feats if f["incomplete"])

            t = timing.get(key) or {}
            seconds = t.get("seconds")
            total_out = sum(all_out) if all_out else 0
            summaries.append({
                "route_key": key,
                "agent": route["agent"],
                "benchmark": route.get("benchmark"),
                "n_prompts_sampled": route.get("n_prompts"),
                "n_prompts_with_results": len(groups),
                "n_rollouts": n_rollouts,
                "expected_rollouts": t.get("expected_rollouts"),
                "n_failure_rows": len(failures),
                "mean_reward": statistics.fmean(all_rewards) if all_rewards else None,
                "pass_rate_mean": (sum(1 for r in all_rewards if r >= args.pass_threshold) / len(all_rewards)) if all_rewards else None,
                "pass_rate_hist": {b: hist.get(b, 0) for b in BINS},
                "zero_variance_prompts": zero_var,
                "zero_variance_fraction": (zero_var / len(groups)) if groups else None,
                "keep_0.2_0.8_prompts": keep,
                "keep_fraction": (keep / len(groups)) if groups else None,
                "output_tokens": {
                    "mean": statistics.fmean(all_out) if all_out else None,
                    "p50": percentile(all_out, 0.5), "p90": percentile(all_out, 0.9), "max": max(all_out) if all_out else None,
                    "total": total_out,
                },
                "cap_hit_fraction_route_cap": (cap_route / n_rollouts) if n_rollouts else None,
                "route_cap": cap,
                "cap_hit_fraction_65536": (cap_global / n_rollouts) if n_rollouts else None,
                "incomplete_fraction": (incomplete / n_rollouts) if n_rollouts else None,
                "invalid_tool_call_text_fraction": (invalid_text / n_rollouts) if n_rollouts else None,
                "multi_function_call_fraction": (multi_call / n_rollouts) if n_rollouts else None,
                "no_function_call_when_expected": no_call,
                "empty_output_fraction": (empty / n_rollouts) if n_rollouts else None,
                "wall_seconds": seconds,
                "output_tokens_per_second": (total_out / seconds) if (seconds and total_out) else None,
                "historical_vs_new_bins": {f"{h}->{n_}": c for (h, n_), c in sorted(hist_vs_new.items())},
                "exit_code": t.get("exit_code"),
                "skipped": t.get("skipped", False),
            })

    (out / "route_summary.json").write_text(json.dumps({"runtime": runtime, "routes": summaries}, indent=2) + "\n")

    lines = ["# HF399 pass-rate profile summary", ""]
    if runtime:
        lines.append(f"Runtime: job {runtime.get('job_id')} on {runtime.get('host')}, K={runtime.get('k')}, "
                     f"vLLM max_num_seqs={runtime.get('max_num_seqs')} x DP8, max_model_len={runtime.get('max_model_len')}, "
                     f"mamba_ssm_cache_dtype={runtime.get('mamba_ssm_cache_dtype')}, temperature=1.0, top_p=1.0, thinking on.")
        lines.append("")
    lines.append("| route | prompts | rollouts (fail) | mean reward | pass=1 rate | zero-var % | keep [0.2,0.8] % | out tok mean/p90/max | cap-hit route / 65536 | invalid-tool text % | multi-call % | wall s | out tok/s |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")

    def pct(x):
        return "-" if x is None else f"{100 * x:.1f}"

    def num(x, fmt="{:.3f}"):
        return "-" if x is None else fmt.format(x)

    for s in summaries:
        if s.get("status"):
            lines.append(f"| {s['route_key']} | - | {s['status']} | | | | | | | | | | |")
            continue
        ot = s["output_tokens"]
        lines.append(
            f"| {s['route_key']} | {s['n_prompts_with_results']}/{s['n_prompts_sampled']} | {s['n_rollouts']} ({s['n_failure_rows']}) | "
            f"{num(s['mean_reward'])} | {pct(s['pass_rate_mean'])} | {pct(s['zero_variance_fraction'])} | {pct(s['keep_fraction'])} | "
            f"{num(ot['mean'], '{:.0f}')}/{ot['p90'] if ot['p90'] is not None else '-'}/{ot['max'] if ot['max'] is not None else '-'} | "
            f"{pct(s['cap_hit_fraction_route_cap'])} (cap {s['route_cap']}) / {pct(s['cap_hit_fraction_65536'])} | "
            f"{pct(s['invalid_tool_call_text_fraction'])} | {pct(s['multi_function_call_fraction'])} | "
            f"{s['wall_seconds'] if s['wall_seconds'] is not None else '-'} | {num(s['output_tokens_per_second'], '{:.0f}')} |")
    lines.append("")
    lines.append("## Per-route pass-rate histogram (per prompt, K rollouts)")
    lines.append("")
    lines.append("| route | " + " | ".join(BINS) + " |")
    lines.append("|---|" + "---|" * len(BINS))
    for s in summaries:
        if s.get("status"):
            continue
        lines.append(f"| {s['route_key']} | " + " | ".join(str(s["pass_rate_hist"][b]) for b in BINS) + " |")
    lines.append("")
    lines.append("Notes: `pass=1 rate` counts reward >= 1 as a pass (binary verifiers); `mean reward` is the "
                 "fractional signal for multichallenge (rubric mean) and conversational_tool_use (0/1 after judge). "
                 "`zero-var %` is the fraction of prompts whose K rewards are identical: those contribute no GRPO advantage. "
                 "`cap-hit route` uses each route's --max-output-tokens; `65536` is the training cap. "
                 "`invalid-tool text` scans message text for `<tool_call>`/`<function_call>` tags (the NeMo-RL invalid_tool_call_patterns); "
                 "`multi-call` counts responses with >1 function_call items (the single-step verifier scores these 0).")
    lines.append("")
    lines.append(f"Files: per_prompt.jsonl, route_summary.json, {keep_path.name} (prompts to keep for training).")
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out / 'SUMMARY.md'}, {out / 'route_summary.json'}, {per_prompt_path}, {keep_path}")


if __name__ == "__main__":
    main()
