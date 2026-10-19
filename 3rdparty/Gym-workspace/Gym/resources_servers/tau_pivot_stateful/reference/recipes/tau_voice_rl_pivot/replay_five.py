# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Restore one frozen audit example, replay its prefix, and execute pivot controls.

Run each family in a fresh process, with its archived code/tau sources first on
PYTHONPATH. Network is disabled before importing any native framework module.
Failing a gate preserves its evidence and stops certification for that example.
"""

import argparse
import copy
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import re
import socket
import sys
import time
import traceback
import types
from pathlib import Path


def no_network(*args, **kwargs):
    raise RuntimeError("Network disabled: deterministic Pivot replay has zero model/API calls")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def call_output(call):
    return {
        "type": "function_call",
        "call_id": "positive-control",
        "name": call["name"],
        "arguments": json.dumps(call["arguments"], ensure_ascii=False),
    }


def state_of(env):
    return {
        "agent": env.tools.db.model_dump(mode="json"),
        "user": env.user_tools.db.model_dump(mode="json") if env.user_tools else None,
    }


def policy_messages(messages):
    converted = []
    for m in messages:
        if m.get("type") == "text":
            converted.append(
                {"role": "assistant" if m.get("source") == "agent" else "user", "content": m.get("text", "")}
            )
        elif m.get("type") == "tool_call":
            converted.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": m["tool_call_id"], "function": {"name": m["tool_name"], "arguments": m["arguments"]}}
                    ],
                }
            )
        elif m.get("type") == "tool_execution":
            converted.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["execution_result"]})
    return converted


def get_native_result_content(result):
    return result.content if hasattr(result, "content") else str(result)


def run_areal(raw, root, report, pivot, override=None):
    from tau2.data_model.message import ToolCall
    from tau2.data_model.tasks import InitialState, Task

    from recipes.tau_voice_sdg import areal_policy_checks as policy

    telecom_policy = None
    if raw["domain_name"] == "telecom":
        from recipes.tau_voice_sdg import areal_telecom_policy as telecom_policy
    from recipes.tau_voice_sdg.areal_gold_replay import constructor, replay
    from recipes.tau_voice_sdg.areal_step_oracle import render_consent
    from recipes.tau_voice_sdg.areal_step_verifier import NativeStepVerifier

    source = raw["source_artifacts"]
    task = copy.deepcopy(raw["tau_task"])
    task["db_path"] = source["db_path"]
    asset = root / "frozen/assets" / ("areal-telecom" if raw["domain_name"] == "telecom" else "areal")
    db = asset / task["db_path"]
    if sha(db) != source["db_sha256"]:
        raise ValueError("Initial DB SHA does not match recorded source")
    expected = next(
        json.loads(line) for line in (asset / "gates/gold_replay.jsonl").open() if json.loads(line)["id"] == task["id"]
    )
    oracle = replay(task, asset)
    oracle_pass = oracle["gold_db_hash"] == expected["gold_db_hash"] and oracle.get(
        "gold_user_db_hash"
    ) == expected.get("gold_user_db_hash")
    report["oracle"] = {"observed": oracle, "expected": expected, "passed": oracle_pass, "db_sha256": sha(db)}
    if not oracle_pass:
        raise ValueError("Gold oracle final hash mismatch")
    report["gates"]["G_B"] = "pass"
    report["active_gate"] = "G_C"
    env = constructor(raw["domain_name"], str(db))()
    initial = Task.model_validate(task).initial_state or InitialState()
    env.set_state(initial.initialization_data, initial.initialization_actions, initial.message_history or [])
    verifier = NativeStepVerifier(env, task, raw["policy"], closure_entities=policy.mapped_entities(asset))
    native = raw["result"]["source_artifacts"]["native_environment"]
    messages = raw["result"]["trajectory"]["messages"]
    events = [e for e in native["f3s"]["events"] if e.get("actor") == "assistant" and e.get("rejected_candidate")]
    boundary = min(events, key=lambda e: e["trajectory_index"])
    index = boundary["trajectory_index"]
    report["boundary"] = {"kind": "raw_trajectory_index", "index": index, "recorded_reason": boundary["reason"]}
    user_events = native.get("user_tool_events") or []
    replayed = []
    interventions = []
    for i in range(index + 1):
        for e in user_events:
            if e.get("before_trajectory_index") != i:
                continue
            c = e["call"]
            args = c.get("arguments") or {}
            args = json.loads(args) if isinstance(args, str) else args
            verdict = verifier.preview(env, policy_messages(messages[:i]), c["name"], args, "user")
            if not verdict["accepted"]:
                raise ValueError("Recorded accepted caller action fails native preview")
            result = env.get_response(
                ToolCall(id=c.get("call_id", "user-replay"), name=c["name"], arguments=args, requestor="user")
            )
            recorded = e.get("output")
            recorded = recorded.get("content") if isinstance(recorded, dict) else recorded
            if get_native_result_content(result) != recorded:
                report["replay_mismatch"] = {
                    "actor": "user",
                    "trajectory_index": i,
                    "expected": recorded,
                    "actual": result.content,
                }
                raise ValueError("Caller tool result byte mismatch")
            if not result.error:
                verifier.commit(verdict)
            replayed.append(
                {
                    "actor": "user",
                    "before_trajectory_index": i,
                    "name": c["name"],
                    "arguments": args,
                    "recorded_result": recorded,
                    "passed": True,
                }
            )
        if i == index:
            break
        m = messages[i]
        if m.get("type") != "tool_call" or m.get("source") != "agent":
            continue
        args = json.loads(m["arguments"]) if isinstance(m["arguments"], str) else m["arguments"]
        verdict = verifier.preview(env, policy_messages(messages[:i]), m["tool_name"], args)
        if not verdict["accepted"]:
            interventions.append({"trajectory_index": i, "verdict": verdict})
            report["assistant_interventions"] = interventions
            raise ValueError("Accepted prefix action rejected by frozen step checker")
        result = env.get_response(ToolCall(id=m["tool_call_id"], name=m["tool_name"], arguments=args))
        recorded = next(
            x["execution_result"]
            for x in messages[i + 1 : index]
            if x.get("type") == "tool_execution" and x.get("tool_call_id") == m["tool_call_id"]
        )
        if result.content != recorded:
            report["replay_mismatch"] = {
                "actor": "assistant",
                "trajectory_index": i,
                "tool": m["tool_name"],
                "expected": recorded,
                "actual": result.content,
            }
            raise ValueError("Agent tool result byte mismatch")
        if not result.error:
            verifier.commit(verdict)
        replayed.append(
            {
                "actor": "assistant",
                "trajectory_index": i,
                "name": m["tool_name"],
                "arguments": args,
                "recorded_result": recorded,
                "passed": True,
            }
        )
    history = policy_messages(messages[:index])
    saved = [x for x in boundary["rejected_candidate"] if x.get("type") != "reasoning"]
    report["prefix_replay"] = {"actions": replayed, "assistant_interventions": interventions, "normalization": []}
    report["state_hash_at_boundary"] = digest({"environment": state_of(env), "ledger": verifier.gold})
    report["remaining_gold"] = [x for x in verifier.gold if not x["done"]]
    report["gates"]["G_C"] = "pass"
    report["active_gate"] = "G_D"

    def native_check(call):
        args = call.get("arguments", {})
        args = json.loads(args) if isinstance(args, str) else args
        verdict = verifier.preview(copy.deepcopy(env), history, call["name"], args)
        return {**verdict, "matched_gold": verdict.get("match") is not None}

    negative_native = native_check(next(x for x in saved if x.get("type") == "function_call"))
    report["negative_native"] = negative_native
    if negative_native["accepted"] or negative_native.get("reason") != boundary["reason"]:
        raise ValueError("Negative control did not reproduce recorded rejection reason")
    report["gates"]["G_D"] = "pass"
    report["active_gate"] = "G_E"
    if override is not None:
        return override_controls(
            override, report, pivot, native_check, lambda: native_check({"name": "end_call", "arguments": {}})
        )
    remaining = [g for g in verifier.gold if not g["done"] and g["requestor"] == "assistant"]
    rejected = next(x for x in saved if x.get("type") == "function_call")
    rejected_args = json.loads(rejected["arguments"])
    proposed = next(
        (g for g in remaining if g["name"] == rejected["name"] and g["arguments"] == rejected_args),
        remaining[0] if remaining else None,
    )
    if proposed is None:
        raise ValueError("No remaining assistant action for a positive control")
    probe = native_check(call_output(proposed))
    is_telecom = raw["domain_name"] == "telecom"
    unmet = [k for k, v in (probe.get("private_diagnostics") or {}).get("gates", {}).items() if v is False]
    report["derived_at_boundary"] = {"gold_probe": probe, "unmet_gates": unmet, "proposed": proposed}
    if probe["accepted"]:
        kind = "write"
        positive = [call_output(proposed)]
        consent_check = None
    elif unmet and set(unmet) <= ({"C3"} if is_telecom else {"C1", "C2", "C3", "C4"}):
        kind = "consent_message"
        shadow = copy.deepcopy(env)
        posting = shadow.get_response(ToolCall(id="control", name=proposed["name"], arguments=proposed["arguments"]))
        try:
            result = json.loads(posting.content)
        except ValueError:
            result = None
        before = state_of(env)["agent"]
        text = render_consent(proposed["name"], proposed["arguments"], before, result)
        if is_telecom:
            amount = telecom_policy.refuel_charge(before, proposed["arguments"])
            text = f"Confirm adding {proposed['arguments']['gb_amount']} GB for ${amount:.2f} total. Do you confirm?"
        text = pivot.cosmetic_control_text(text)
        positive = [{"type": "message", "role": "assistant", "content": text}]
        consent_check = pivot.consent_predicates(
            policy,
            {"before": before, "result": result, "messages": history},
            proposed,
            unmet,
            telecom=is_telecom,
            telecom_policy=telecom_policy,
        )
    else:
        report["verifier_suspect"] = True
        report["order_sensitive_candidate"] = boundary["reason"] == "gold_delta" and any(
            g["name"] == rejected["name"] and g["arguments"] == rejected_args for g in remaining
        )
        raise ValueError("Gold-consistent positive cannot pass native checker; no registered consent repair applies")
    spec = pivot.spec_for(kind, report["verifier_version"], unmet=unmet)
    spec["kind"] = "areal_native_step"
    negative = pivot.evaluate(spec, saved, native_check=native_check, consent_check=consent_check)
    positive_result = pivot.evaluate(spec, positive, native_check=native_check, consent_check=consent_check)
    report.update(
        expected_turn_kind=kind,
        verification=spec,
        controls={
            "negative": {"input_sha": digest(saved), **negative},
            "positive": {
                "input_sha": digest(positive),
                "construction": "deterministic remaining-gold template; control only",
                "output": positive,
                **positive_result,
            },
        },
    )
    if negative["reward"] != 0 or positive_result["reward"] != 1:
        report["verifier_suspect"] = True
        raise ValueError("Binary pivot controls failed")
    report["gates"]["G_E"] = "pass"


def public_visibility_text(items):
    projected = []
    for item in items:
        keys = {
            "message": ("role", "content"),
            "function_call": ("name", "arguments"),
            "function_call_output": ("output",),
        }.get(item.get("type", "message"))
        if keys:
            projected.append({key: item[key] for key in keys if key in item})
    return json.dumps(projected, ensure_ascii=False)


def replay_gold_action(env, action, mapping, *, native_call, direct_agent_call, direct_user_call, grants=False):
    """Exact original gold wrappers; grants use the frozen public runtime route.

    Legacy mode preserves turn-2 behavior for the matched I13 comparison.
    No native source, checker, or authorization policy is changed.
    """
    name, args = action["name"], action["arguments"]
    actor = action.get("requestor", "assistant")
    if grants:
        if name == "give_discoverable_user_tool":
            return direct_agent_call(
                env,
                "enable_caller_action",
                {
                    "action_name": args["discoverable_tool_name"],
                    "arguments": args.get("arguments", {}),
                },
                mapping,
            ), "frozen direct_agent_call.enable_caller_action"
        # The gold plan explicitly carries unlock/call pairs. Execute each
        # once rather than re-unlocking inside the direct assistant adapter.
        return native_call(
            env, name, args, requestor="user" if name == "call_discoverable_user_tool" else actor
        ), "frozen native_call.original_gold_action"
    if name in {"unlock_discoverable_agent_tool", "give_discoverable_user_tool"}:
        return None, "legacy_dropped_wrapper"
    if name == "call_discoverable_agent_tool":
        original = args["agent_tool_name"]
        public = next(k for k, v in mapping.items() if v == original)
        actual = json.loads(args["arguments"]) if isinstance(args["arguments"], str) else args["arguments"]
        return direct_agent_call(env, public, actual, mapping), "legacy_direct_agent_call"
    if name == "call_discoverable_user_tool":
        original = args["discoverable_tool_name"]
        actual = json.loads(args["arguments"]) if isinstance(args["arguments"], str) else args["arguments"]
        return direct_user_call(env, original, actual, {original: original}), "legacy_direct_user_call"
    return native_call(env, name, args, requestor=actor), "frozen native_call"


def run_afterquery(raw, root, report, pivot, gold_path, override=None, gold_grants=False, ordered_callers=False):
    from recipes.tau_voice_sdg.afterquery_step_verifier import F3sStepVerifier
    from tau2.data_model.tasks import Task

    from recipes.tau_voice_sdg.afterquery_adapter import (
        direct_agent_call,
        direct_user_call,
        native_call,
    )

    def load_environment(domain, task, *, retrieval_variant):
        # Identical constructor/set_state arguments to the archived adapter;
        # avoid the registry's eager model/voice runner imports, not its state semantics.
        module = __import__("tau2.domains." + domain + ".environment", fromlist=["get_environment"])
        parsed = Task.model_validate(task)
        env = module.get_environment(retrieval_variant=retrieval_variant, task=parsed)
        initial = parsed.initial_state
        env.set_state(
            initialization_data=initial.initialization_data if initial else None,
            initialization_actions=initial.initialization_actions if initial else None,
            message_history=list(initial.message_history or []) if initial else [],
            strict=False,
        )
        return env

    task = raw["tau_task"]
    domain = raw["domain_name"]
    source = raw.get("source_artifacts") or {}
    mapping = source.get("k1_tool_map") or {}
    user_map = {}
    expected = next(
        r for r in json.loads(gold_path.read_text())["tasks"] if r["domain"] == domain and r["task_id"] == task["id"]
    )
    oracle = load_environment(domain, task, retrieval_variant="bm25")
    initial_db_hash = oracle.get_db_hash()
    outputs = []
    for action in task["evaluation_criteria"].get("actions") or []:
        result, route = replay_gold_action(
            oracle,
            action,
            mapping,
            native_call=native_call,
            direct_agent_call=direct_agent_call,
            direct_user_call=direct_user_call,
            grants=gold_grants,
        )
        if result is None:
            continue
        if result.error:
            raise ValueError("Gold oracle tool error: " + str(result))
        outputs.append(
            {
                "name": action["name"],
                "requestor": action.get("requestor", "assistant"),
                "arguments": action["arguments"],
                "content": result.content,
                "route": route,
                "content_error": str(result.content).casefold().startswith("error"),
            }
        )
    observed = {"agent_db_hash": oracle.get_db_hash(), "user_db_hash": oracle.get_user_db_hash()}
    passed = observed["agent_db_hash"] == expected["agent_db_hash"] and observed["user_db_hash"] == expected.get(
        "user_db_hash"
    )
    report["oracle"] = {
        "observed": observed,
        "expected": expected,
        "passed": passed,
        "actions": outputs,
        "initial_db_hash": initial_db_hash,
        "grant_wrappers_replayed": gold_grants,
        "wrapper_actions": sum(
            item["name"] in {"unlock_discoverable_agent_tool", "give_discoverable_user_tool"} for item in outputs
        ),
    }
    if not passed:
        raise ValueError("AfterQuery gold oracle DB hash mismatch")
    report["gates"]["G_B"] = "pass"
    report["active_gate"] = "G_C"
    env = load_environment(domain, task, retrieval_variant="bm25")
    if ordered_callers:
        from recipes.tau_voice_sdg.afterquery_adapter import k1_user_specs

        _, user_map = k1_user_specs(domain, env, task)
    verifier = F3sStepVerifier(task, direct_agent_map=mapping, direct_user_map=user_map)
    output = raw["response"]["output"]
    calls = {x.get("call_id"): (i, x) for i, x in enumerate(output) if x.get("type") == "function_call"}
    rejection = []
    for i, item in enumerate(output):
        if item.get("type") != "function_call_output":
            continue
        try:
            value = json.loads(item.get("output", ""))
        except (ValueError, TypeError):
            continue
        if isinstance(value, dict) and (value.get("private_f3s_note") or value.get("abandoned")):
            call = calls[item["call_id"]]
            rejection.append((call[0], call[1], value))
    boundary, saved_call, recorded = min(rejection, key=lambda e: e[0])
    report["boundary"] = {"kind": "response_output_index", "index": boundary, "recorded_rejection": recorded}
    native = raw["result"]["source_artifacts"]["native_environment"]
    caller_events = native.get("user_tool_events") or []
    if caller_events and not ordered_callers:
        raise ValueError(
            "AfterQuery private caller actions require an ordered trace mapping; not implemented for this example"
        )
    replayed = []
    if ordered_callers:
        helper_spec = importlib.util.spec_from_file_location(
            "ordered_afterquery", Path(__file__).with_name("ordered_afterquery.py")
        )
        helper = importlib.util.module_from_spec(helper_spec)
        helper_spec.loader.exec_module(helper)
        replayed, report["ordered_caller_replay"] = helper.replay_prefix(
            raw,
            boundary,
            env,
            verifier,
            mapping,
            user_map,
            direct_agent_call=direct_agent_call,
            direct_user_call=direct_user_call,
        )
    for i, item in enumerate([] if ordered_callers else output[:boundary]):
        if item.get("type") != "function_call":
            continue
        args = json.loads(item["arguments"])
        decision = verifier.observe_agent_tool(item["name"], args)
        if not decision["accepted"]:
            report["assistant_interventions"] = [{"response_output_index": i, "verdict": decision}]
            raise ValueError("Earlier AfterQuery assistant intervention before selected boundary")
        result = direct_agent_call(env, item["name"], args, mapping)
        recorded_result = next(
            x["output"]
            for x in output[i + 1 : boundary]
            if x.get("type") == "function_call_output" and x.get("call_id") == item["call_id"]
        )
        if result.content != recorded_result:
            report["replay_mismatch"] = {
                "response_output_index": i,
                "tool": item["name"],
                "expected": recorded_result,
                "actual": result.content,
            }
            raise ValueError("AfterQuery tool result byte mismatch")
        replayed.append(
            {
                "response_output_index": i,
                "actor": "assistant",
                "name": item["name"],
                "arguments": args,
                "recorded_result": recorded_result,
                "passed": True,
            }
        )
    report["prefix_replay"] = {"actions": replayed, "assistant_interventions": [], "normalization": []}
    boundary_state = {"environment": state_of(env), "ledger": verifier.remaining}
    if ordered_callers:
        boundary_state["caller_grants"] = copy.deepcopy(
            getattr(env.tools, "_user_discoverable_tools_state", None) or {}
        )
        report["caller_grants_at_boundary"] = boundary_state["caller_grants"]
    report["state_hash_at_boundary"] = digest(boundary_state)
    report["remaining_gold"] = verifier.remaining
    report["gates"]["G_C"] = "pass"
    report["active_gate"] = "G_D"

    def native_check(call):
        v = copy.deepcopy(verifier)
        args = call.get("arguments") or {}
        args = json.loads(args) if isinstance(args, str) else args
        return v.observe_agent_tool(call["name"], args)

    negative_native = native_check(saved_call)
    report["negative_native"] = negative_native
    if negative_native["accepted"] or negative_native.get("reason") != recorded.get("private_f3s_note"):
        raise ValueError("AfterQuery negative reason mismatch")
    report["gates"]["G_D"] = "pass"
    report["active_gate"] = "G_E"
    if override is not None:
        return override_controls(override, report, pivot, native_check, lambda: copy.deepcopy(verifier).observe_end())
    remaining = [g for g in verifier.remaining if g[0] == "assistant" and g[1] != "end_call"]
    if not remaining:
        raise ValueError("No remaining gold action for positive control")
    _, name, args = next(
        (
            g
            for g in remaining
            if g[1].removeprefix("@discoverable:") == mapping.get(saved_call["name"], saved_call["name"])
        ),
        remaining[0],
    )
    public = next((k for k, v in mapping.items() if "@discoverable:" + v == name), name)
    positive = [call_output({"name": public, "arguments": args})]

    # Missing public values are a separate elicitation tier; spelling variants
    # are normalized only for visibility, never for byte-identical tool replay.
    public_text = public_visibility_text([*raw["responses_create_params"]["input"], *output[:boundary]])
    visible = re.sub(r"[^a-z0-9]", "", public_text.casefold())
    missing = []
    for key, value in args.items():
        values = list(value.values()) if isinstance(value, dict) else [value]
        if any(
            isinstance(v, str) and len(v) > 2 and re.sub(r"[^a-z0-9]", "", v.casefold()) not in visible for v in values
        ):
            missing.append(key)
    if missing:
        schema = next(t["params"] for t in raw["tools"] if t["name"] == public)
        elicit = {
            key: set(
                re.findall(
                    r"[a-z]+",
                    key.replace("_", " ")
                    + " "
                    + schema.get("properties", {}).get(key, {}).get("description", "").casefold(),
                )
            )
            - {"the", "a", "an", "of", "to", "for", "and", "or", "is", "in", "with"}
            for key in missing
        }
        if any(not words for words in elicit.values()):
            raise ValueError("Missing argument has no schema-derived elicitation tokens")
        kind = "elicit_message"
        positive = [
            {
                "type": "message",
                "role": "assistant",
                "content": "Please provide " + ", ".join(key.replace("_", " ") for key in missing) + ".",
            }
        ]
        spec = pivot.spec_for(kind, report["verifier_version"], elicit=elicit)
    else:
        kind = "write"
        spec = pivot.spec_for(kind, report["verifier_version"])
    spec["kind"] = "afterquery_f3s_step"
    report["missing_public_arguments"] = missing
    negative = pivot.evaluate(spec, [saved_call], native_check=native_check)
    positive_result = pivot.evaluate(spec, positive, native_check=native_check)
    report.update(
        expected_turn_kind=kind,
        verification=spec,
        controls={
            "negative": {"input_sha": digest([saved_call]), **negative},
            "positive": {
                "input_sha": digest(positive),
                "construction": "deterministic remaining-gold/schema control; never target",
                "output": positive,
                **positive_result,
            },
        },
    )
    if negative["reward"] != 0 or positive_result["reward"] != 1:
        raise ValueError("AfterQuery binary pivot controls failed")
    report["gates"]["G_E"] = "pass"


def override_controls(override, report, pivot, native_check, end_check):
    spec = importlib.util.spec_from_file_location(
        "certification_v12", Path(__file__).with_name("certification_v12.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    row = copy.deepcopy(override["row"])
    if row.get("state_hash_at_boundary") and row["state_hash_at_boundary"] != report["state_hash_at_boundary"]:
        raise ValueError("Changed-kind replay boundary state differs from saved receipt")
    row.setdefault("restore", {})["verifier_version"] = report["verifier_version"]
    result = module.apply_kind_controls(row, override["rule"], pivot, native_check=native_check, end_check=end_check)
    report.update(result)
    report["expected_turn_kind"] = override["rule"]["expected_turn_kind"]
    if result["controls"]["negative"]["reward"] != 0 or result["controls"]["positive"]["reward"] != 1:
        raise ValueError("v1.1 changed-kind controls failed")
    report["gates"]["G_E"] = "pass"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pivot-module", type=Path, required=True)
    parser.add_argument("--afterquery-gold", type=Path)
    parser.add_argument("--afterquery-package", type=Path)
    parser.add_argument("--rule-json", type=Path)
    parser.add_argument("--gold-grants", action="store_true")
    parser.add_argument("--ordered-callers", action="store_true")
    args = parser.parse_args()
    started = time.time()
    root = args.root
    family = root.name
    socket.socket.connect = no_network
    socket.create_connection = no_network
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    os.environ["HF_HUB_OFFLINE"] = "1"
    report = {
        "family": family,
        "gates": {"G_" + k: "fail" for k in "ABCDEFGH"},
        "active_gate": "G_A",
        "training_ready": False,
        "model_calls": 0,
        "network_disabled": True,
        "normalization": [],
    }
    try:
        pin = json.loads((root / "PIN_RECEIPT.json").read_text())
        raw = json.loads((root / "raw.json").read_text())
        if pin["status"] != "PASS":
            raise ValueError("Archive pin failed")
        for name, expected in pin["frozen_file_sha256"].items():
            if sha(root / "frozen" / name) != expected:
                raise ValueError("Frozen source/asset hash changed: " + name)
        modules = pin.get("verifier_module_set") or (
            ["areal_step_verifier.py", "areal_policy_checks.py", "areal_telecom_policy.py"]
            if family.startswith("areal")
            else ["afterquery_step_verifier.py", "afterquery_adapter.py"]
        )
        version = {name: pin["frozen_file_sha256"]["code/recipes/tau_voice_sdg/" + name] for name in modules}
        launch = json.loads((root / "launch.json").read_text())
        report["verifier_version"] = family + ":" + str(launch.get("code_commit")) + ":" + digest(version)
        report["pinned_modules"] = version
        report["module_set_derivation"] = pin.get("module_set_derivation", "turn-2 fixed module set")
        report["id"] = raw["id"]
        report["gates"]["G_A"] = "pass"
        report["active_gate"] = "G_B"
        spec = importlib.util.spec_from_file_location("pivot_reference", args.pivot_module)
        pivot = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pivot)
        override = json.loads(args.rule_json.read_text()) if args.rule_json else None
        if family.startswith("areal"):
            run_areal(raw, root, report, pivot, override=override)
        else:
            base_tau = root / "frozen/site-packages/tau-company/tau2"
            if not base_tau.is_dir() or args.afterquery_package is None:
                raise ValueError("Pinned base Tau source and AfterQuery package are required")
            # Namespace-only bootstrap leaves all imported environment/checker
            # modules byte-identical to the frozen archives. __init__ only exports
            # runner APIs, which are outside this deterministic backend harness.
            parent = types.ModuleType("tau2")
            parent.__path__ = [str(base_tau)]
            parent.__spec__ = importlib.machinery.ModuleSpec("tau2", loader=None, is_package=True)
            domains = types.ModuleType("tau2.domains")
            domains.__path__ = [str(args.afterquery_package / "src/tau2/domains"), str(base_tau / "domains")]
            domains.__spec__ = importlib.machinery.ModuleSpec("tau2.domains", loader=None, is_package=True)
            parent.domains = domains
            sys.modules["tau2"] = parent
            sys.modules["tau2.domains"] = domains
            report["import_bootstrap"] = {
                "namespace_only": ["tau2", "tau2.domains"],
                "base": str(base_tau),
                "domains": domains.__path__,
                "constructor": "pinned domain.get_environment; identical adapter set_state",
            }
            run_afterquery(
                raw,
                root,
                report,
                pivot,
                args.afterquery_gold,
                override=override,
                gold_grants=args.gold_grants,
                ordered_callers=args.ordered_callers,
            )
        report["harness_pass"] = all(report["gates"]["G_" + k] == "pass" for k in "ABCDE")
        # G-F/G-G/G-H are intentionally not certified by the five-example harness.
        report["first_failed_gate"] = "G_F"
        report["status"] = "controls_pass_prefix_certification_pending"
    except Exception as exc:
        report["harness_pass"] = False
        report["status"] = "FAIL"
        report["first_failed_gate"] = report["active_gate"]
        report["error"] = type(exc).__name__ + ": " + str(exc)
        report["traceback"] = traceback.format_exc()
    if os.environ.get("PIVOT_ENFORCE_IMPORT_GUARD") == "1":
        guard_spec = importlib.util.spec_from_file_location(
            "pivot_import_guard", Path(__file__).with_name("import_guard.py")
        )
        guard = importlib.util.module_from_spec(guard_spec)
        guard_spec.loader.exec_module(guard)
        report["lustre_import_guard"] = (
            "lustre-import-guard: ok (" + str(guard.assert_no_lustre_imports()) + " modules checked)"
        )
    report["wall_seconds"] = time.time() - started
    report["loaded_native_modules"] = {
        k: getattr(v, "__file__", None)
        for k, v in sys.modules.items()
        if k.startswith(("tau2.", "recipes.tau_voice_sdg."))
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(family, report["status"], report.get("error", ""), report["gates"])


if __name__ == "__main__":
    main()
