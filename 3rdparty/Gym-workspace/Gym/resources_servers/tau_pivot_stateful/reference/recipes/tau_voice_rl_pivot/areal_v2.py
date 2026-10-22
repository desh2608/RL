# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""G5 kind evaluator. Eligibility requires independently replayed boundary evidence.

An online SDG rejection is never eligibility evidence. This module deliberately
requires callbacks for native feasibility and the version-pinned consent check;
it cannot certify a record from a reference-action mismatch alone.
"""

import json
import re
from collections.abc import Callable
from typing import Any

from recipes.tau_voice_rl_pivot.consent_given import detect


G_TOOL_KINDS = frozenset({"K1", "K2", "K5", "K6", "K7"})
G_CLOSE_TOOLS = frozenset({"end_call", "transfer_to_human_agents", "transfer_to_human_agent"})
G_COMPLETION = re.compile(
    r"\b(?:I(?:'ve| have)|we(?:'ve| have))\s+(?:successfully\s+)?"
    r"(?:cancelled|canceled|changed|updated|modified|booked|returned|refunded|exchanged|processed|sent|"
    r"voided|removed|deleted|credited|debited|issued|rebooked|completed|confirmed)\b|"
    r"\b(?:has been|have been|is(?: now)?|are(?: now)?|was|were)\s+"
    r"(?:successfully\s+)?(?:cancelled|canceled|changed|updated|modified|booked|refunded|processed|sent|"
    r"voided|removed|deleted|credited|debited|issued|rebooked|completed|confirmed)\b|"
    r"\b(?:done|all set|completed|successfully processed)\b",
    re.I,
)
G_REFUSAL = re.compile(r"\b(?:cannot|can't|unable|not allowed|not eligible|ineligible|may not|must not)\b", re.I)


def typed_equal(actual: Any, gold: Any) -> bool:
    """Object order is irrelevant; list order and scalar types are significant.

    Booleans never equal numeric 0/1. JSON integer and float representations of
    the same finite number are equivalent; strings never coerce to numbers.
    """
    if isinstance(gold, bool) or isinstance(actual, bool):
        return type(actual) is type(gold) and actual == gold
    if isinstance(gold, (int, float)) and isinstance(actual, (int, float)):
        return actual == gold and abs(actual) != float("inf")
    if type(actual) is not type(gold):
        return False
    if isinstance(gold, dict):
        return actual.keys() == gold.keys() and all(typed_equal(actual[k], v) for k, v in gold.items())
    if isinstance(gold, list):
        return len(actual) == len(gold) and all(typed_equal(a, b) for a, b in zip(actual, gold))
    return actual == gold


def one_turn(output: list[dict]) -> tuple[str, dict | str | None]:
    """Accept exactly one assistant message or one typed function call."""
    if not isinstance(output, list):
        return "invalid", None
    if any(not isinstance(x, dict) for x in output):
        return "invalid", None
    items = [x for x in output if x.get("type") != "reasoning"]
    if len(items) != 1:
        return "invalid", None
    item = items[0]
    if item.get("type") == "function_call":
        args = item.get("arguments")
        try:
            args = json.loads(args) if isinstance(args, str) else args
        except (ValueError, TypeError):
            return "invalid", None
        if not isinstance(args, dict) or not isinstance(item.get("name"), str):
            return "invalid", None
        return "call", {"name": item["name"], "arguments": args}
    if item.get("type") == "message" and item.get("role") == "assistant":
        content = item.get("content")
        if isinstance(content, list):
            if any(not isinstance(x, dict) or x.get("type") not in {"output_text", "text"} for x in content):
                return "invalid", None
            if any(not isinstance(x.get("text", ""), str) for x in content):
                return "invalid", None
            content = "".join(x.get("text", "") for x in content)
        if (
            isinstance(content, str)
            and content.strip()
            and not re.search(r"<\s*(?:tool_call|function)\b", content, re.I)
        ):
            return "message", content
    return "invalid", None


def lookup_arguments_public(context: list[dict] | None, call: dict) -> bool:
    """Do not reward a gold lookup whose literal arguments are unavailable."""
    if not isinstance(context, list):
        return False
    visible = json.dumps(context, ensure_ascii=False)

    def values(value):
        if isinstance(value, dict):
            return [leaf for child in value.values() for leaf in values(child)]
        if isinstance(value, list):
            return [leaf for child in value for leaf in values(child)]
        return [str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)]

    return all(
        re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", visible, re.I)
        for value in values(call.get("arguments", {}))
    )


def evaluate(
    spec: dict,
    output: list[dict],
    *,
    native_check: Callable[[dict], dict] | None = None,
    consent_check: Callable[[str], dict] | None = None,
) -> dict:
    """Evaluate one turn against a pinned, replay-certified G5 boundary.

    native_check reports feasibility/state transitions without old online consent
    gates. consent_check checks a proposed question, never grants actual consent.
    The eligibility proof and actual prior consent are fixed when mining.
    """
    if spec.get("contract") == "G5-INBOX36-R1":
        from recipes.tau_voice_rl_pivot.areal_r1 import evaluate

        return evaluate(spec, output, native_check=native_check, consent_check=consent_check)
    kind = spec.get("error_kind")
    shape, value = one_turn(output)
    proof = spec.get("eligibility", {})
    predicates = {
        "version": str(spec.get("verifier_version", "")).startswith("areal-v2:"),
        "boundary_verified": proof.get("prefix_replayed") is True and bool(proof.get("state_hash_at_boundary")),
        "kind_witness": proof.get("kind") == kind and proof.get("proven") is True,
        "single_turn": shape != "invalid",
    }
    native = None
    context = proof.get("consent_context")
    target = spec.get("gold")
    consent = None
    uncertainty = None
    k4_eligible = False
    k5_eligible = False
    if isinstance(context, list) and isinstance(target, dict) and isinstance(target.get("name"), str):
        result = detect(context, target, native_amount=proof.get("native_amount"))
        consent = result["authorized_prior_consent"]
        k4_eligible = result["k4_eligible"]
        k5_eligible = result["k5_eligible"]
        uncertainty = result.get("uncertain_prior_consent")
    # Recompute from the fixed public boundary. A stale miner boolean cannot
    # override an actual confirmation and yes, even with perfect content gates.
    consent_agrees = consent is not None and consent is proof.get("valid_prior_consent")
    if kind in G_TOOL_KINDS:
        target = spec.get("gold")
        predicates["gold_call"] = shape == "call" and bool(target) and typed_equal(value, target)
        predicates["not_close"] = shape == "call" and value["name"] not in G_CLOSE_TOOLS
        predicates["correct_kind"] = kind != "K6" or proof.get("gold_tool_type") in {"read", "auth"}
        predicates["public_read_arguments"] = proof.get("gold_tool_type") not in {
            "read",
            "auth",
        } or lookup_arguments_public(context, target)
        predicates["valid_consent"] = kind != "K5" or (proof.get("valid_prior_consent") is True and k5_eligible)
        predicates["write_authorized"] = proof.get("gold_tool_type") in {"read", "auth"} or (
            proof.get("gold_tool_type") == "write"
            and ((consent_agrees and consent is True) or proof.get("consent_required") is False)
        )
        if predicates["gold_call"] and predicates["public_read_arguments"] and native_check is not None:
            native = native_check(value)
        predicates["native_feasible"] = bool(native and native.get("feasible") is True)
        predicates["native_step_due"] = bool(native and native.get("step_due") is True)
    elif kind == "K4":
        predicates["no_valid_prior_consent"] = (
            consent_agrees and consent is False and uncertainty is None and k4_eligible
        )
        predicates["no_segment_assent"] = k4_eligible
        predicates["public_write_arguments"] = isinstance(target, dict) and lookup_arguments_public(context, target)
        predicates["message_only"] = shape == "message"
        if predicates["no_valid_prior_consent"] and isinstance(target, dict) and native_check is not None:
            native = native_check(target)
        predicates["write_due"] = bool(native and native.get("feasible") is True and native.get("write_due") is True)
        checked = (
            consent_check(value)
            if predicates["no_valid_prior_consent"]
            and predicates["write_due"]
            and predicates["public_write_arguments"]
            and shape == "message"
            and consent_check is not None
            else {}
        )
        predicates["relaxed_consent_content"] = all(checked.get(f"C{i}") is True for i in range(1, 5))
        predicates["checker_pinned"] = bool(spec.get("consent_checker_sha256"))
    elif kind == "K3":
        # INBOX33: lexical refusals failed adversarial qualification. Preserve
        # source contexts as holds; never serve a K3-positive v2 record.
        predicates["K3_deferred_INBOX33"] = False
    else:
        predicates["known_kind"] = False
    return {"reward": int(all(predicates.values())), "predicates": predicates, "native": native}
