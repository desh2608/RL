# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""INBOX37 scoring: caller-confirmed writes, discussion targets, guarded K7."""

from recipes.tau_voice_rl_pivot import areal_r1, areal_v2
from recipes.tau_voice_rl_pivot.confirmed_write import confirmed_write, discussion_target, known_ids, target_ids


def qualify(spec: dict) -> dict:
    kind = spec["error_kind"]
    prefix = spec["eligibility"]["consent_context"]
    gold = spec["gold"]
    if kind in {"K1", "K2", "K5"}:
        witness = confirmed_write(prefix, gold)
        if not witness["ok"]:
            return dict(status="hold", reason="upstream_offer_divergence", rule="confirmed_write", witness=witness)
        return dict(status="candidate", rule="confirmed_write", witness=witness)
    if kind == "K4":
        target = discussion_target(prefix, gold)
        if not target["ok"]:
            return dict(status="hold", reason="K4_target_mismatch", rule="discussion_target", witness=target)
    return dict(status="candidate", rule="unchanged_eligibility")


def evaluate(spec: dict, output: list[dict], *, native_check=None, consent_check=None) -> dict:
    kind = spec["error_kind"]
    proof = spec["eligibility"]
    prefix = proof.get("consent_context", [])
    shape, value = areal_v2.one_turn(output)
    if kind == "K7":
        allowed = False
        confirmation = None
        native = None
        if shape == "message":
            allowed = True
        elif shape == "call" and value["name"] not in areal_v2.G_CLOSE_TOOLS:
            for step in proof.get("r2_remaining_steps", []):
                if not areal_v2.typed_equal(value, areal_r1.call_of(step)):
                    continue
                if step["tool_type"] in {"read", "auth"}:
                    allowed = True
                elif step["tool_type"] == "write":
                    confirmation = confirmed_write(prefix, value)
                    if confirmation["ok"] and native_check is not None:
                        native = native_check(value)
                        allowed = native.get("feasible") is True
                break
        predicates = dict(
            version=str(spec.get("verifier_version", "")).startswith("areal-v2:"),
            boundary_verified=proof.get("prefix_replayed") is True and bool(proof.get("state_hash_at_boundary")),
            kind_witness=proof.get("kind") == "K7" and proof.get("proven") is True,
            single_turn=shape != "invalid",
            no_transfer_or_end=shape == "message" or (shape == "call" and value["name"] not in areal_v2.G_CLOSE_TOOLS),
            spoken_read_or_confirmed_write=allowed,
        )
        return dict(
            reward=int(all(predicates.values())), predicates=predicates, native=native, confirmation=confirmation
        )
    base = {**spec, "contract": "G5-INBOX36-R1"}
    result = areal_r1.evaluate(base, output, native_check=native_check, consent_check=consent_check)
    qualification = qualify(spec)
    result["r2_qualification"] = qualification
    result["predicates"]["R2_qualification"] = qualification["status"] == "candidate"
    if kind == "K4" and shape == "message":
        gold = spec["gold"]
        key = next((k for k in ["order_id", "reservation_id"] if k in gold["arguments"]), None)
        ids = target_ids(value, key, known_ids(prefix, key)) if key else set()
        result["predicates"]["R2_question_target"] = not ids or (str(gold["arguments"][key]) in ids and len(ids) == 1)
    result["reward"] = int(all(result["predicates"].values()))
    return result
