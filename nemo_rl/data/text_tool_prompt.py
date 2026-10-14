# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate static text/tool history without rewriting it or executing tools."""
import copy
import json
from typing import Any


def validate_text_tool_messages(messages: Any, *, owner: str) -> list[dict[str, Any]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{owner}: messages must be a nonempty list")
    pending: set[str] = set()
    for i, message in enumerate(messages):
        if not isinstance(message, dict):
            raise TypeError(f"{owner}: message {i} must be a mapping")
        role = message.get("role")
        allowed = {"role", "content"}
        if role == "assistant":
            allowed |= {"tool_calls", "name"}
        elif role == "tool":
            allowed |= {"tool_call_id", "name"}
        elif role == "user":
            allowed |= {"name"}
        elif role != "system":
            raise ValueError(f"{owner}: unsupported message role {role!r}")
        if set(message) - allowed:
            raise ValueError(f"{owner}: unsupported {role} fields {sorted(set(message)-allowed)}")
        if not isinstance(message.get("content"), str):
            raise TypeError(f"{owner}: only string message content is supported")
        if "name" in message and (not isinstance(message["name"], str) or not message["name"]):
            raise ValueError(f"{owner}: name must be nonempty text")
        if role == "system" and i != 0:
            raise ValueError(f"{owner}: only one leading system message is supported")
        # Some static conversation pivots include the assistant's opening
        # greeting before the first user turn. Preserve that supplied history.
        if role != "tool" and pending:
            raise ValueError(f"{owner}: tool call results missing before next turn")
        if role == "assistant":
            calls = message.get("tool_calls", [])
            if not isinstance(calls, list):
                raise TypeError(f"{owner}: tool_calls must be a list")
            if not message["content"].strip() and not calls:
                raise ValueError(f"{owner}: empty assistant without tool calls")
            for call in calls:
                if not isinstance(call, dict) or set(call) != {"id", "type", "function"}:
                    raise ValueError(f"{owner}: tool call must have id, type, function")
                if call["type"] != "function" or not isinstance(call["id"], str) or not call["id"]:
                    raise ValueError(f"{owner}: invalid function call identity")
                if call["id"] in pending:
                    raise ValueError(f"{owner}: duplicate tool call ID in one assistant turn")
                function = call["function"]
                if not isinstance(function, dict) or set(function) != {"name", "arguments"}:
                    raise ValueError(f"{owner}: function needs name and arguments")
                if not isinstance(function["name"], str) or not function["name"]:
                    raise ValueError(f"{owner}: function name is empty")
                arguments = function["arguments"]
                if not isinstance(arguments, str) or not isinstance(json.loads(arguments), dict):
                    raise ValueError(f"{owner}: function arguments must encode a JSON object")
                pending.add(call["id"])
        elif role == "tool":
            tool_id = message.get("tool_call_id")
            if not isinstance(tool_id, str) or tool_id not in pending:
                raise ValueError(f"{owner}: orphan or repeated tool result")
            pending.remove(tool_id)
        elif not message["content"].strip():
            raise ValueError(f"{owner}: empty {role} content")
    if messages[-1]["role"] not in ("user", "tool") or pending:
        raise ValueError(f"{owner}: generation prompt must end after user or completed tool results")
    return copy.deepcopy(messages)


def validate_text_tool_schemas(tools: Any, *, owner: str) -> list[dict[str, Any]]:
    if not isinstance(tools, list):
        raise TypeError(f"{owner}: tools must be a list")
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise ValueError(f"{owner}: only function tool schemas are supported")
        fn = tool.get("function")
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str) or not fn["name"]:
            raise ValueError(f"{owner}: tool schema requires a function name")
        if "parameters" in fn and not isinstance(fn["parameters"], dict):
            raise TypeError(f"{owner}: tool parameters schema must be an object")
    return copy.deepcopy(tools)


def text_tool_render_messages(messages: Any, *, owner: str) -> list[dict[str, Any]]:
    """Adapt OAI JSON-string arguments to the model Jinja mapping contract.

    Authoritative source messages retain the original strings. Both original
    student and Lightning templates serialize argument mappings themselves.
    """
    rendered = validate_text_tool_messages(messages, owner=owner)
    for message in rendered:
        for call in message.get("tool_calls", []):
            call["function"]["arguments"] = json.loads(call["function"]["arguments"])
    return rendered
