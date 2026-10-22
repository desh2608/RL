# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""INBOX36 recovery: remaining writes are a set, not a prescribed trace order."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from recipes.tau_voice_rl_pivot.areal_native import Boundary

from recipes.tau_voice_rl_pivot import areal_v2
from recipes.tau_voice_rl_pivot.consent_given import detect, is_reask


def call_of(gold: dict) -> dict:
    return {key: gold[key] for key in ("name", "arguments")}


def same_target_offer(offer: str, gold: dict) -> bool:
    """A K5 question must identify this order/reservation or the profile address."""
    if gold["name"] == "modify_user_address":
        return bool(re.search(r"\b(?:profile|default)(?: shipping)? address\b", offer, re.I))
    args = gold["arguments"]
    ids = [args[key] for key in ("order_id", "reservation_id") if key in args]
    return bool(ids) and all(re.search(r"(?<!\w)" + re.escape(str(value)) + r"(?!\w)", offer) for value in ids)


def target(call: dict) -> tuple | None:
    for key in ("order_id", "reservation_id", "user_id"):
        if key in call["arguments"]:
            return key, call["arguments"][key]
    return None


def writes(boundary: Boundary) -> list[dict]:
    return [g for g in boundary.remaining() if g["requestor"] == "assistant" and boundary.tool_type(g) == "write"]


def matching(boundary: Boundary, call: dict) -> list[dict]:
    return [
        g for g in boundary.remaining() if g["requestor"] == "assistant" and areal_v2.typed_equal(call_of(g), call)
    ]


def needed_lookup(boundary: Boundary, candidate: dict, prefix: list[dict]) -> dict | None:
    """Require missing public values and a read that supplies the intended target.

    An unrelated next read never proves a write erroneous. Private caller choices
    that no native read supplies remain holds, not manufactured lookup targets.
    """
    if matching(boundary, candidate) or areal_v2.lookup_arguments_public(prefix, candidate):
        return None
    intended = [g for g in writes(boundary) if g["name"] == candidate["name"] and target(g) == target(candidate)]
    if len(intended) != 1:
        return None
    wanted = call_of(intended[0])
    if areal_v2.lookup_arguments_public(prefix, wanted):
        return None
    for read in boundary.remaining():
        if read["requestor"] != "assistant" or boundary.tool_type(read) not in {"read", "auth"}:
            continue
        lookup = call_of(read)
        if not areal_v2.lookup_arguments_public(prefix, lookup):
            continue
        native = boundary.native_check(lookup)
        result = [{"role": "tool", "content": str(native.get("native_result", ""))}]
        if native["feasible"] and areal_v2.lookup_arguments_public(result, wanted):
            return {
                "status": "candidate",
                "error_kind": "K6",
                "gold": lookup,
                "native": native,
                "consent": detect(prefix, lookup),
                "witness": "unmatched_ungrounded_write_and_needed_native_read",
            }
    return None


def inspect_write(
    boundary: Boundary, output: list[dict], prefix: list[dict], candidate: dict, reference: dict | None
) -> dict:
    exact = [g for g in writes(boundary) if areal_v2.typed_equal(call_of(g), candidate)]
    if exact:
        native = boundary.native_check(candidate)
        consent = detect(prefix, candidate, native_amount=boundary.charge(candidate, native.get("native_result")))
        if not native["feasible"]:
            return {"status": "hold", "reason": "R1_matching_gold_not_native_feasible", "native": native}
        if not boundary.consent_required(candidate) or consent["authorized_prior_consent"]:
            return {"status": "continue", "reason": "R1_any_remaining_gold_write_authorized"}
        if consent["k4_eligible"] and areal_v2.lookup_arguments_public(prefix, candidate):
            return {
                "status": "candidate",
                "error_kind": "K4",
                "gold": candidate,
                "native": native,
                "consent": consent,
                "witness": "R1_ready_gold_write_without_assent",
            }
        return {"status": "hold", "reason": "R1_matching_gold_consent_uncertainty", "consent": consent}
    missing = needed_lookup(boundary, candidate, prefix)
    if missing:
        return missing
    same_tool = [g for g in writes(boundary) if g["name"] == candidate["name"]]
    if same_tool:
        possibilities = [g for g in same_tool if target(g) is not None and target(g) == target(candidate)]
    elif reference is not None and boundary.tool_type(reference) == "write":
        possibilities = [g for g in writes(boundary) if areal_v2.typed_equal(call_of(g), reference)]
    else:
        possibilities = writes(boundary)
    ready = []
    for gold in possibilities:
        call = call_of(gold)
        native = boundary.native_check(call)
        consent = detect(prefix, call, native_amount=boundary.charge(call, native.get("native_result")))
        if (
            native["feasible"]
            and areal_v2.lookup_arguments_public(prefix, call)
            and (not boundary.consent_required(call) or consent["authorized_prior_consent"])
        ):
            ready.append(call)
    if len(ready) != 1:
        return {"status": "hold", "reason": "R1_write_target_or_authorization_unproven"}
    # Preserve native consequence checks; R1 does not impose the old scalar-only
    # filter on public, same-target wrong arguments.
    # The wrapper has established K1=no matching tool, or K2=same tool+target.
    return boundary._inspect_student_legacy(output, prefix, ready[0])


def inspect(boundary: Boundary, output: list[dict], prefix: list[dict]) -> dict:
    shape, value = areal_v2.one_turn(output)
    if shape == "invalid":
        return {"status": "hold", "reason": "unsupported_turn_shape"}
    if shape == "message":
        hold = None
        for gold in writes(boundary):
            call = call_of(gold)
            native = boundary.native_check(call)
            amount = boundary.charge(call, native.get("native_result"))
            consent = detect(prefix, call, native_amount=amount)
            if not is_reask(value, call, prefix=prefix, native_amount=amount):
                continue
            witness = consent.get("witness") or {}
            if (
                consent["k5_eligible"]
                and same_target_offer(witness.get("offer", ""), call)
                and areal_v2.lookup_arguments_public(prefix, call)
                and native["feasible"]
            ):
                return {
                    "status": "candidate",
                    "error_kind": "K5",
                    "gold": call,
                    "native": native,
                    "consent": consent,
                    "witness": "R1_same_target_public_arguments_reask",
                }
            if consent["valid_prior_consent"] or consent["consent_label"] == "hold":
                hold = {"status": "hold", "reason": "R1_K5_target_or_arguments_unproven", "consent": consent}
        return hold or {"status": "continue", "reason": "speech_facts_outside_K1_K7"}
    if value["name"] in areal_v2.G_CLOSE_TOOLS:
        for gold in boundary.remaining():
            if gold["requestor"] != "assistant":
                continue
            call = call_of(gold)
            native = boundary.native_check(call)
            if native["feasible"] and areal_v2.lookup_arguments_public(prefix, call):
                return {
                    "status": "candidate",
                    "error_kind": "K7",
                    "gold": call,
                    "native": native,
                    "consent": detect(prefix, call),
                    "witness": "R1_in_scope_remaining_native_step",
                }
        return {"status": "hold", "reason": "R1_K7_no_executable_public_continuation"}
    kind = boundary.tool_type(value)
    if kind == "write":
        return inspect_write(boundary, output, prefix, value, None)
    if kind in {"read", "auth"}:
        return {"status": "continue", "reason": "R1_read_order_not_an_error"}
    return {"status": "hold", "reason": "unknown_tool_type"}


def inspect_student(boundary: Boundary, output: list[dict], prefix: list[dict], reference: dict | None) -> dict:
    shape, value = areal_v2.one_turn(output)
    if shape != "call":
        return inspect(boundary, output, prefix)
    if value["name"] in areal_v2.G_CLOSE_TOOLS:
        return inspect(boundary, output, prefix)
    kind = boundary.tool_type(value)
    if kind == "write":
        return inspect_write(boundary, output, prefix, value, reference)
    if matching(boundary, value):
        return {"status": "continue", "reason": "R1_any_remaining_gold_read"}
    if reference is not None and kind in {"read", "auth"}:
        return boundary._inspect_student_legacy(output, prefix, reference)
    return inspect(boundary, output, prefix)


def evaluate(spec: dict, output: list[dict], *, native_check=None, consent_check=None) -> dict:
    """R1 scoring is versioned; all earlier contracts retain their evaluator."""
    if spec.get("error_kind") == "K7":
        shape, value = areal_v2.one_turn(output)
        proof = spec.get("eligibility", {})
        remaining = [
            call_of(g)
            for g in proof.get("ordered_ledger", [])
            if not g.get("done") and g.get("requestor", "assistant") == "assistant"
        ]
        predicates = {
            "version": str(spec.get("verifier_version", "")).startswith("areal-v2:"),
            "boundary_verified": proof.get("prefix_replayed") is True and bool(proof.get("state_hash_at_boundary")),
            "kind_witness": proof.get("kind") == "K7" and proof.get("proven") is True,
            "single_turn": shape != "invalid",
            "no_transfer_or_end": shape == "message"
            or (shape == "call" and value["name"] not in areal_v2.G_CLOSE_TOOLS),
            "spoken_or_remaining_gold": shape == "message"
            or (shape == "call" and any(areal_v2.typed_equal(value, gold) for gold in remaining)),
        }
        return {"reward": int(all(predicates.values())), "predicates": predicates, "native": None}
    base = {**spec, "contract": "legacy-G5"}
    result = areal_v2.evaluate(base, output, native_check=native_check, consent_check=consent_check)
    gold = spec["gold"]
    prefix = spec["eligibility"].get("consent_context")
    if spec.get("error_kind") in {"K1", "K2", "K5"}:
        result["predicates"]["R1_public_write_arguments"] = areal_v2.lookup_arguments_public(prefix, gold)
    if spec.get("error_kind") == "K5":
        consent = detect(prefix, gold, native_amount=spec["eligibility"].get("native_amount"))
        result["predicates"]["R1_same_target_offer"] = same_target_offer(
            (consent.get("witness") or {}).get("offer", ""), gold
        )
    result["reward"] = int(all(result["predicates"].values()))
    return result
