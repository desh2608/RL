# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""INBOX62 F3 projection onto fields declared by each public tool schema.

Structured union branches take precedence over a generic dict fallback. A
genuinely open map with no declared properties retains its keys. Neither
values, list order, omitted fields, nor JSON-encoded strings are rewritten.
"""

import copy
import json


CONTRACT = "INBOX62-schema-fields-v1"


def resolve(node: dict, root: dict) -> dict:
    """Resolve local references and collect declared union properties."""
    if "$ref" in node:
        target = root
        reference = node["$ref"]
        if not reference.startswith("#/"):
            raise ValueError("Only local tool-schema references are supported")
        for part in reference[2:].split("/"):
            target = target[part.replace("~1", "/").replace("~0", "~")]
        node = {**target, **{key: value for key, value in node.items() if key != "$ref"}}
    result = dict(node)
    properties = dict(node.get("properties", {}))
    for keyword in ("allOf", "anyOf", "oneOf"):
        for branch in node.get(keyword, []):
            branch = resolve(branch, root)
            properties.update(branch.get("properties", {}))
            for key in ("items", "additionalProperties"):
                if key not in result and key in branch:
                    result[key] = branch[key]
    if properties:
        result["properties"] = properties
    return result


def project(value: object, schema: dict, root: dict) -> object:
    """Drop undeclared object keys while preserving the supplied typed values."""
    schema = resolve(schema, root)
    if isinstance(value, dict):
        properties = schema.get("properties")
        if properties is not None:
            return {key: project(child, properties[key], root) for key, child in value.items() if key in properties}
        additional = schema.get("additionalProperties")
        if isinstance(additional, dict):
            return {key: project(child, additional, root) for key, child in value.items()}
        return copy.deepcopy(value) if additional is True else {}
    if isinstance(value, list):
        item = schema.get("items")
        return [project(child, item, root) for child in value] if isinstance(item, dict) else copy.deepcopy(value)
    return value


def schemas(tools: list[dict]) -> dict[str, dict]:
    return {
        (tool.get("function") or tool)["name"]: (tool.get("function") or tool).get("parameters", {}) for tool in tools
    }


def call(value: dict, tools: list[dict]) -> dict:
    """Normalize a known tool call; unknown names remain unknown to evaluators."""
    result = copy.deepcopy(value)
    schema = schemas(tools).get(value.get("name"))
    if schema is None:
        return result
    arguments = value.get("arguments")
    encoded = isinstance(arguments, str)
    if encoded:
        try:
            arguments = json.loads(arguments)
        except (ValueError, TypeError):
            return result
    if not isinstance(arguments, dict):
        return result
    normalized = project(arguments, schema, schema)
    if normalized != arguments:
        result["arguments"] = json.dumps(normalized, ensure_ascii=False) if encoded else normalized
    return result


def tree(value: object, tools: list[dict]) -> object:
    """Project stored call witnesses and generated calls with the same schemas."""
    if isinstance(value, list):
        return [tree(child, tools) for child in value]
    if isinstance(value, dict):
        if isinstance(value.get("name"), str) and "arguments" in value:
            return call(value, tools)
        return {key: tree(child, tools) for key, child in value.items()}
    return value
