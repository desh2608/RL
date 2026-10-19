# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strict Responses-to-trajectory binding and recorded K1 caller replay.

No coordinate shifting, speculative authorization, or checker changes. Future
caller events are counted but never applied to the earlier rejection boundary.
"""

import json


def content_text(value):
    if isinstance(value, str):
        return value
    return " ".join(p.get("text", "") for p in value if isinstance(p, dict))


def args(value):
    return json.loads(value) if isinstance(value, str) else value


def matches(item, message):
    kind = item.get("type", "message")
    if kind == "message":
        return (
            message.get("type") == "text"
            and message.get("source") == {"assistant": "agent", "user": "user"}.get(item.get("role"))
            and content_text(item.get("content", "")) == message.get("text", "")
        )
    if kind == "function_call":
        return (
            message.get("type") == "tool_call"
            and message.get("source") == "agent"
            and item["call_id"] == message.get("tool_call_id")
            and item["name"] == message.get("tool_name")
            and args(item["arguments"]) == args(message["arguments"])
        )
    if kind == "function_call_output":
        return (
            message.get("type") == "tool_execution"
            and item["call_id"] == message.get("tool_call_id")
            and item["output"] == message.get("execution_result")
        )
    return False


def bind_prefix(raw, boundary):
    messages = raw["result"]["trajectory"]["messages"]
    initial = [x for x in raw["responses_create_params"]["input"] if x.get("role") != "system"]
    position = 0
    for item in initial:
        if not matches(item, messages[position]):
            raise ValueError("Initial public input does not bind to raw trajectory")
        position += 1
    bound = []
    for i, item in enumerate(raw["response"]["output"][:boundary]):
        if item.get("type") == "reasoning":
            continue
        if position >= len(messages) or not matches(item, messages[position]):
            raise ValueError(f"Ambiguous Responses/raw trajectory binding at output {i}, trajectory {position}")
        bound.append((i, position, item))
        position += 1
    return bound, position


def replay_prefix(raw, boundary, env, verifier, mapping, user_map, *, direct_agent_call, direct_user_call):
    bound, end = bind_prefix(raw, boundary)
    native = raw["result"]["source_artifacts"]["native_environment"]
    events = native.get("user_tool_events") or []
    if any(e["before_trajectory_index"] == end for e in events):
        raise ValueError("Caller event exactly at rejected/unrecorded boundary is chronologically ambiguous")
    event_positions = [e["before_trajectory_index"] for e in events]
    if event_positions != sorted(event_positions):
        raise ValueError("Caller event order is not monotonic")
    due = {}
    actions = []
    for e in events:
        if e["before_trajectory_index"] < end:
            due.setdefault(e["before_trajectory_index"], []).append(e)
    direct = native.get("afterquery_direct_events") or []
    replayed_users = 0
    output = raw["response"]["output"]
    for i, position, item in bound:
        for e in due.pop(position, []):
            call = e["call"]
            arguments = args(call["arguments"])
            original = call["name"]
            public = next((k for k, v in user_map.items() if v == original), None)
            if public is None:
                raise ValueError("Recorded caller tool is absent from frozen K1 user map")
            verdict = verifier.observe_user_tool(public, arguments)
            if not verdict["accepted"]:
                raise ValueError("Recorded accepted caller action rejected by frozen checker")
            grants = [d for d in direct if d.get("call_id") == call["call_id"] + "-k1-give"]
            enabled = getattr(env.tools, "_user_discoverable_tools_state", None) or {}
            authorize = False
            if original not in enabled:
                if len(grants) != 1:
                    raise ValueError("Missing or ambiguous recorded K1 grant; implicit authorization forbidden")
                grant = grants[0]
                if (
                    grant.get("name") != "give_discoverable_user_tool"
                    or grant.get("requestor") != "assistant"
                    or grant["arguments"].get("discoverable_tool_name") != original
                    or args(grant["arguments"]["arguments"]) != arguments
                ):
                    raise ValueError("Recorded K1 grant arguments do not bind to caller event")
                authorize = True
            elif grants:
                raise ValueError("Recorded K1 grant conflicts with restored enabled state")
            # The archived app calls this exact route. Its adapter creates the
            # empty-argument enable wrapper internally, then executes the call.
            result = direct_user_call(env, public, arguments, user_map, authorize_if_needed=authorize)
            if result.content != e["output"] or bool(result.error) != bool(e.get("error")):
                raise ValueError("Caller result/error differs byte-for-byte from recorded event")
            actions.append(
                {
                    "actor": "user",
                    "before_trajectory_index": position,
                    "call_id": call["call_id"],
                    "name": original,
                    "arguments": arguments,
                    "recorded_result": e["output"],
                    "grant_receipt": grants[0] if grants else None,
                    "route": "frozen direct_user_call",
                    "authorize_if_needed": authorize,
                    "passed": True,
                }
            )
            replayed_users += 1
        if item.get("type") != "function_call":
            continue
        arguments = args(item["arguments"])
        verdict = verifier.observe_agent_tool(item["name"], arguments)
        if not verdict["accepted"]:
            raise ValueError("Earlier assistant intervention in ordered prefix replay")
        result = direct_agent_call(env, item["name"], arguments, mapping)
        recorded = next(
            x["output"]
            for x in output[i + 1 : boundary]
            if x.get("type") == "function_call_output" and x.get("call_id") == item["call_id"]
        )
        if result.content != recorded:
            raise ValueError("Assistant result differs byte-for-byte in ordered prefix replay")
        actions.append(
            {
                "actor": "assistant",
                "response_output_index": i,
                "trajectory_index": position,
                "name": item["name"],
                "arguments": arguments,
                "recorded_result": recorded,
                "passed": True,
            }
        )
    if due:
        raise ValueError("Caller event has no exact public trajectory binding")
    return actions, {
        "boundary_trajectory_position": end,
        "bound_public_items": len(bound),
        "caller_events_replayed": replayed_users,
        "future_caller_events_excluded": len(events) - replayed_users,
        "normalization": [],
    }
