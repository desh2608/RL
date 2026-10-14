# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed terminal verifiers for actual teacher-solved text/tool metadata.

The legacy strict tool_action schema and qwen3_coder XML coercions come from
NemotronVC_warmup/tools/verify/verify_rollouts_v2.py. Weak message-only and
non-strict first-call checks are deliberately unsupported, never zero rewards.
This module has no model, Ray, or torch dependency and can preflight CPU data.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from decimal import Decimal
from functools import lru_cache
from typing import Any


class RewardContractError(ValueError):
    """The task cannot be scored by this adapter; no reward may be fabricated."""


class RewardInfrastructureError(RuntimeError):
    """The evaluator failed; this is not a model failure or zero reward."""


@dataclass(frozen=True)
class RewardSpec:
    """Validated private task data; never added to the actor prompt."""

    verifier_type: str
    payload: dict[str, Any]
    tools: list[dict[str, Any]]
    verifier_sha256: str


@dataclass(frozen=True)
class RewardResult:
    """A finite valid outcome. Infrastructure/unsupported outcomes raise."""

    schema_version: int
    status: str
    score: float
    outcome: str
    source_id: str
    domain: str
    verifier_type: str
    verifier_sha256: str
    response_sha256: str
    response_truncated: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise ValueError(f"nonfinite JSON number: {value}")


def _finite_decimal(value: str) -> Decimal:
    result = Decimal(value)
    if not result.is_finite() or not math.isfinite(float(result)):
        raise ValueError("nonfinite JSON number")
    return result


def _strict_json(text: str) -> Any:
    return json.loads(
        text,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
        parse_float=_finite_decimal,
    )


def _exact_equal(expected: Any, actual: Any) -> bool:
    if type(expected) is not type(actual):
        return False
    if isinstance(expected, dict):
        return expected.keys() == actual.keys() and all(
            _exact_equal(value, actual[key]) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(expected) == len(actual) and all(
            _exact_equal(a, b) for a, b in zip(expected, actual)
        )
    return expected == actual


def _expected_call(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("type") != "function_call":
        raise RewardContractError(
            "Only explicit function_call references are supported"
        )
    name = value.get("name")
    if not isinstance(name, str) or not name.strip() or name != name.strip():
        raise RewardContractError(
            "Reference function name must be nonempty canonical text"
        )
    arguments = value.get("arguments")
    if arguments is None:
        arguments = {}
    try:
        encoded = (
            arguments
            if isinstance(arguments, str)
            else json.dumps(arguments, allow_nan=False)
        )
        decoded = _strict_json(encoded if encoded.strip() else "{}")
    except (TypeError, ValueError, RecursionError) as error:
        raise RewardContractError("Reference arguments are not strict JSON") from error
    if not isinstance(decoded, dict):
        raise RewardContractError("Reference arguments must be an object")
    return {"name": name, "arguments": decoded}


@lru_cache(maxsize=1)
def _instruction_registry() -> Mapping[str, Any]:
    # These optional dependencies are only needed when selecting IF data.
    try:
        detector = importlib.import_module("langdetect")
        registry_module = importlib.import_module(
            "verifiable_instructions.instructions_registry"
        )
    except ModuleNotFoundError as error:
        raise RewardInfrastructureError(
            "IF requires the pinned verifiable_instructions and langdetect packages"
        ) from error
    detector.DetectorFactory.seed = 0
    registry = registry_module.INSTRUCTION_DICT
    if not isinstance(registry, Mapping):
        raise RewardInfrastructureError("IF registry INSTRUCTION_DICT is not a mapping")
    return registry


def _if_arguments(payload: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Validate portable IF structure without importing scoring dependencies."""
    if payload.get("grading_mode") not in (None, "binary"):
        raise RewardContractError("Only binary all-constraint IF is supported")
    ids, kwargs = payload.get("instruction_id_list"), payload.get("kwargs")
    if (
        not isinstance(ids, list)
        or not ids
        or not isinstance(kwargs, list)
        or len(ids) != len(kwargs)
    ):
        raise RewardContractError(
            "IF instruction_id_list and kwargs must be nonempty and aligned"
        )
    entries = []
    for instruction_id, arguments in zip(ids, kwargs):
        if not isinstance(instruction_id, str) or not instruction_id.strip():
            raise RewardContractError("IF instruction IDs must be nonempty strings")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, Mapping):
            raise RewardContractError("IF kwargs must be mappings or null")
        canonical = {
            key: value for key, value in arguments.items() if value is not None
        }
        if instruction_id == "count:count_increment_word":
            for key in ("keyword1", "keyword2"):
                if isinstance(canonical.get(key), list):
                    values = canonical[key]
                    if len(values) != 1 or not isinstance(values[0], str):
                        raise RewardContractError(
                            "count_increment_word requires singleton endpoints"
                        )
                    canonical[key] = values[0]
        entries.append((instruction_id, canonical))
    return entries


def _if_instructions(payload: Mapping[str, Any]) -> list[Any]:
    entries = _if_arguments(payload)
    registry = _instruction_registry()
    instructions = []
    for instruction_id, canonical in entries:
        if instruction_id not in registry:
            raise RewardContractError(f"Unsupported IF instruction: {instruction_id!r}")
        instruction = registry[instruction_id](instruction_id)
        try:
            instruction.build_description(**canonical)
        except (TypeError, ValueError, AssertionError) as error:
            raise RewardContractError(
                f"IF kwargs cannot build {instruction_id}"
            ) from error
        instructions.append(instruction)
    return instructions


def validate_reward_spec(
    metadata: Mapping[str, Any],
    *,
    tools: list[dict[str, Any]],
    validate_if_runtime: bool = True,
) -> RewardSpec:
    """Validate frozen data-v1 original_verifier or explicit reward verifier.

    Call this in CPU materialization before spending GPU time. IF dependencies
    and task kwargs are exercised by default. Only DataLoader preprocessing
    disables runtime validation: the collector uses a separate interpreter,
    while CPU input sealing and the reward actor always validate the full IF
    runtime. Unsupported rows never become negative training examples.
    """
    verifier = metadata.get("original_verifier")
    if verifier is None:
        verifier = metadata.get("verifier")
    if not isinstance(verifier, Mapping):
        raise RewardContractError("Missing verifier mapping")
    version = verifier.get("schema_version")
    if isinstance(version, bool) or version not in (1, 2):
        raise RewardContractError(
            "Only known verifier schema versions1/2 are supported"
        )
    kind, payload = verifier.get("type"), verifier.get("payload")
    if not isinstance(payload, dict) or not isinstance(tools, list):
        raise RewardContractError("Verifier payload/tools have invalid types")
    if kind == "tool_action":
        if payload.get("strict") is not True:
            raise RewardContractError(
                "Weak/non-strict first-call tool_action is unsupported"
            )
        expected = payload.get("expected")
        alternative = payload.get("expected_action")
        if expected is not None and alternative is not None and expected != alternative:
            raise RewardContractError(
                "Conflicting expected and expected_action aliases"
            )
        call = _expected_call(expected if expected is not None else alternative)
        schemas = []
        for tool in tools:
            if not isinstance(tool, dict) or not isinstance(tool.get("function"), dict):
                raise RewardContractError("Tool schema must contain a function mapping")
            function = tool["function"]
            if function.get("name") == call["name"]:
                parameters = function.get("parameters")
                if not isinstance(parameters, dict):
                    raise RewardContractError("Tool parameters must be an object")
                if "properties" in parameters and not isinstance(
                    parameters["properties"], dict
                ):
                    raise RewardContractError("Tool properties must be an object")
                schemas.append(parameters)
        if len(schemas) != 1:
            raise RewardContractError("Expected function has no supplied tool schema")
    elif kind == "instruction_following":
        if validate_if_runtime:
            _if_instructions(payload)
        else:
            _if_arguments(payload)
    else:
        raise RewardContractError(f"Unsupported reward verifier: {kind!r}")
    try:
        canonical = json.dumps(
            {"verifier": verifier, "tools": tools},
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise RewardContractError("Verifier metadata is not finite JSON") from error
    return RewardSpec(
        str(kind),
        copy.deepcopy(payload),
        copy.deepcopy(tools),
        hashlib.sha256(canonical.encode()).hexdigest(),
    )


def final_response(response: str) -> str | None:
    """Extract one final segment; malformed/empty model output is a failure."""
    text = response.strip()
    if text.endswith("<|im_end|>"):
        text = text[: -len("<|im_end|>")].rstrip()
    if "<|im_end|>" in text or text.count("</think>") != 1 or text.count("<think>") > 1:
        return None
    close = text.index("</think>")
    if "<think>" in text:
        opening = text.index("<think>")
        if text[:opening].strip() or opening >= close:
            return None
    final = text[close + len("</think>") :].strip()
    return final if final and "<think>" not in final else None


_CALL = re.compile(
    r"<tool_call>\s*<function=\s*([^>]+?)\s*>(.*?)</function>\s*</tool_call>", re.DOTALL
)
_PARAM = re.compile(r"<parameter=\s*([^>]+?)\s*>(.*?)</parameter>", re.DOTALL)


def _strip_one_newline(value: str) -> str:
    if value.startswith("\r\n"):
        value = value[2:]
    elif value.startswith("\n"):
        value = value[1:]
    if value.endswith("\r\n"):
        value = value[:-2]
    elif value.endswith("\n"):
        value = value[:-1]
    return value


def _convert_parameter(raw: str, prop: Any) -> Any:
    if raw.lower() == "null":
        return None
    if prop is None:
        return raw
    kind = (
        str(prop["type"]).strip().lower()
        if isinstance(prop, dict) and "type" in prop
        else ("object" if isinstance(prop, dict) and "anyOf" in prop else "string")
    )
    if kind in ("string", "str", "text", "varchar", "char", "enum"):
        return raw
    if kind.startswith(("int", "uint", "long", "short", "unsigned")):
        try:
            return int(raw)
        except ValueError:
            return raw
    if kind.startswith(("num", "float")):
        try:
            value = float(raw)
            return int(value) if value == int(value) else value
        except (ValueError, OverflowError):
            return raw
    if kind in ("boolean", "bool", "binary"):
        # The historical parser coerced every non-"true" string to False.
        # That accepts garbage for an expected false argument; keep it a
        # string so strict equality rejects it instead.
        return raw.lower() == "true" if raw.lower() in ("true", "false") else raw
    if kind in ("object", "array", "arr") or kind.startswith(("dict", "list")):
        try:
            return json.loads(
                raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant
            )
        except (ValueError, TypeError):
            pass
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return raw


def _score_tool(final: str, spec: RewardSpec) -> bool:
    expected = spec.payload.get("expected")
    if expected is None:
        expected = spec.payload["expected_action"]
    reference = _expected_call(expected)
    calls = list(_CALL.finditer(final))
    # The actual strict source specifies a complete singleton call set. Never
    # grade only calls[0] while silently accepting extra calls or non-call text.
    if (
        len(calls) != 1
        or _CALL.sub("", final).strip()
        or final.count("<tool_call>") != 1
    ):
        return False
    name, body = calls[0].group(1).strip(), calls[0].group(2)
    if name != reference["name"] or _PARAM.sub("", body).strip():
        return False
    schemas = [
        t["function"]["parameters"]
        for t in spec.tools
        if t.get("function", {}).get("name") == name
    ]
    if len(schemas) != 1:
        raise RewardContractError("Tool names must have one unambiguous schema")
    parameters = schemas[0]
    props = parameters["properties"] if "properties" in parameters else parameters
    arguments = {}
    for match in _PARAM.finditer(body):
        key = match.group(1).strip()
        if key in arguments:
            return False
        arguments[key] = _convert_parameter(
            _strip_one_newline(match.group(2)), props.get(key)
        )
    try:
        parsed = _strict_json(
            json.dumps(arguments, ensure_ascii=False, allow_nan=False)
        )
        return _exact_equal(reference["arguments"], parsed)
    except (TypeError, ValueError, RecursionError):
        return False


def score_terminal_response(
    response: str,
    *,
    spec: RewardSpec,
    source_id: str,
    domain: str,
    response_truncated: bool,
) -> RewardResult:
    """Score only the current assistant response; never retry a model failure."""
    if not isinstance(response, str) or type(response_truncated) is not bool:
        raise RewardContractError(
            "Response text and trusted truncation flag are required"
        )
    if (
        not isinstance(source_id, str)
        or not source_id
        or not isinstance(domain, str)
        or not domain
    ):
        raise RewardContractError("Source identity and domain are required")
    final = final_response(response)
    if response_truncated:
        score, outcome = 0.0, "budget_exhausted"
    elif final is None:
        score, outcome = 0.0, "malformed"
    else:
        if spec.verifier_type == "tool_action":
            correct = _score_tool(final, spec)
        elif spec.verifier_type == "instruction_following":
            instructions = _if_instructions(spec.payload)
            # Unexpected checker exceptions deliberately propagate as infra.
            correct = all(
                bool(instruction.check_following(final)) for instruction in instructions
            )
        else:
            raise RewardContractError("Unknown validated verifier type")
        score, outcome = float(correct), "success" if correct else "task_failure"
    return RewardResult(
        1,
        "valid",
        score,
        outcome,
        source_id,
        domain,
        spec.verifier_type,
        spec.verifier_sha256,
        hashlib.sha256(response.encode()).hexdigest(),
        response_truncated,
    )
