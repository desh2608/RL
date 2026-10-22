# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mutation checks for G5, including the valid-prior-consent regression."""

import copy
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).parent / "reference"))
from recipes.tau_voice_rl_pivot import areal_v2 as verifier


def spec(kind):
    return {
        "verifier_version": "areal-v2:test",
        "error_kind": kind,
        "gold": {
            "name": "get_order_details" if kind == "K6" else "cancel_pending_order",
            "arguments": {"order_id": "#A1"},
        },
        "consent_checker_sha256": "test-checker-pin",
        "eligibility": {
            "prefix_replayed": True,
            "state_hash_at_boundary": "test-boundary-pin",
            "kind": kind,
            "proven": True,
            "gold_tool_type": "read" if kind == "K6" else "write",
            "valid_prior_consent": kind not in {"K3", "K4"},
            "consent_context": []
            if kind in {"K3", "K4"}
            else [
                {"type": "message", "role": "assistant", "content": "Shall I cancel order #A1?"},
                {"type": "message", "role": "user", "content": "Yes"},
            ],
            "policy_quote": "Basic economy cannot be modified.",
            "policy_predicate": "basic_economy_modification",
            "policy_required_tokens": ["basic economy"],
        },
    }


def message(text):
    return [{"type": "message", "role": "assistant", "content": text}]


def call(name, args):
    return [{"type": "function_call", "name": name, "arguments": args}]


@pytest.mark.parametrize("kind", ["K1", "K2", "K5", "K6", "K7"])
def test_tool_kinds_reject_mutations_and_reasking(kind):
    boundary = spec(kind)
    target = boundary["gold"]
    native_calls = []

    def native(candidate):
        native_calls.append(candidate)
        return {"feasible": True}

    good = call(target["name"], target["arguments"])
    assert verifier.evaluate(boundary, good, native_check=native)["reward"] == 1
    assert len(native_calls) == 1
    for negative in [
        call(target["name"], {"order_id": "#WRONG"}),
        call(target["name"], {}),
        call("end_call", {}),
        call("transfer_to_human_agents", {}),
        message("Should I cancel order #A1?"),
        good + message("Done."),
    ]:
        assert verifier.evaluate(boundary, negative, native_check=native)["reward"] == 0
    assert len(native_calls) == 1  # Invalid candidates never execute in the environment.
    assert verifier.evaluate(boundary, good, native_check=lambda _: {"feasible": False})["reward"] == 0
    assert verifier.evaluate(boundary, good)["reward"] == 0
    if kind != "K6":
        boundary["eligibility"]["valid_prior_consent"] = False
        assert verifier.evaluate(boundary, good, native_check=native)["reward"] == 0


def test_k4_hard_gate_cannot_be_overridden_by_question_checker():
    boundary = spec("K4")
    question = message("Shall I cancel order #A1?")
    check = lambda _: {"C1": True, "C2": True, "C3": True, "C4": True}
    assert verifier.evaluate(boundary, question, consent_check=check)["reward"] == 1
    boundary["eligibility"]["valid_prior_consent"] = True
    assert verifier.evaluate(boundary, question, consent_check=check)["reward"] == 0
    boundary["eligibility"]["valid_prior_consent"] = None
    assert verifier.evaluate(boundary, question, consent_check=check)["reward"] == 0
    boundary["eligibility"]["valid_prior_consent"] = False
    assert verifier.evaluate(boundary, question, consent_check=lambda _: {})["reward"] == 0
    assert verifier.evaluate(boundary, question, consent_check=lambda _: {"C1": True})["reward"] == 0
    for gate in ["C1", "C2", "C3", "C4"]:
        assert verifier.evaluate(boundary, question, consent_check=lambda _, gate=gate: {gate: False})["reward"] == 0
    assert (
        verifier.evaluate(boundary, call("cancel_pending_order", {"order_id": "#A1"}), consent_check=check)["reward"]
        == 0
    )


def test_k3_policy_grounding_no_calls_and_no_completion():
    boundary = spec("K3")
    assert verifier.evaluate(boundary, message("I cannot modify basic economy reservations."))["reward"] == 1
    for negative in [
        message("I have updated the basic economy reservation."),
        message("I cannot modify basic economy, but I have cancelled your reservation."),
        message("I cannot modify basic economy. Your booking is cancelled."),
        message("I cannot modify basic economy. Your ticket is voided."),
        message("I cannot modify basic economy. Your refund has been issued."),
        message("I cannot modify basic economy. Your booking was removed."),
        message("I cannot help with your address."),
        message("Done."),
        call("update_reservation_flights", {}),
    ]:
        assert verifier.evaluate(boundary, negative)["reward"] == 0


@pytest.mark.parametrize("kind", ["K1", "K2", "K3", "K4", "K5", "K6", "K7"])
def test_unproven_or_unreplayed_records_never_pass(kind):
    boundary = spec(kind)
    output = call(boundary["gold"]["name"], boundary["gold"]["arguments"])
    for key in ["prefix_replayed", "proven"]:
        broken = copy.deepcopy(boundary)
        broken["eligibility"][key] = False
        assert verifier.evaluate(broken, output, native_check=lambda _: {"feasible": True})["reward"] == 0


def test_typed_comparison_and_malformed_outputs():
    assert not verifier.typed_equal(True, 1)
    assert not verifier.typed_equal("1", 1)
    assert not verifier.typed_equal([1, 2], [2, 1])
    assert not verifier.typed_equal(float("nan"), float("nan"))
    assert not verifier.typed_equal(float("inf"), float("inf"))
    assert verifier.typed_equal({"x": 1, "y": [2]}, {"y": [2.0], "x": 1.0})
    for output in [
        None,
        [None],
        [{"type": "function_call", "name": "x", "arguments": "{"}],
        message("<tool_call>x</tool_call>"),
    ]:
        assert verifier.evaluate(spec("K1"), output)["reward"] == 0


def test_cached_proof_must_bind_to_the_loaded_state_and_version(monkeypatch):
    sys.path.insert(0, str(Path(__file__).parent))
    import native_runtime

    boundary = spec("K1")
    ref = {
        "record_id": "binding-test",
        "dataset": "v2-areal",
        "verifier_version": boundary["verifier_version"],
        "expected_turn_kind": "write",
    }
    entry = {
        **ref,
        "verification": boundary,
        "state_hash_at_boundary": boundary["eligibility"]["state_hash_at_boundary"],
    }
    callbacks = {"native_check": lambda _: {"feasible": True}, "consent_check": None}
    monkeypatch.setattr(native_runtime, "load_cached", lambda *args: (entry, callbacks))
    item = {"entry": "unused", "entry_sha256": "unused"}
    output = call(boundary["gold"]["name"], boundary["gold"]["arguments"])
    assert native_runtime.verify(Path("/unused"), item, ref, output)["reward"] == 1
    boundary["eligibility"]["state_hash_at_boundary"] = "another-state"
    with pytest.raises(ValueError, match="different boundary"):
        native_runtime.verify(Path("/unused"), item, ref, output)
    boundary["eligibility"]["state_hash_at_boundary"] = entry["state_hash_at_boundary"]
    boundary["verifier_version"] = "areal-v2:another-version"
    with pytest.raises(ValueError, match="specification version"):
        native_runtime.verify(Path("/unused"), item, ref, output)


def test_k4_recomputes_given_consent_despite_stale_false_label():
    boundary = spec("K4")
    boundary["eligibility"]["consent_context"] = spec("K5")["eligibility"]["consent_context"]
    check = lambda _: {"C1": True, "C2": True, "C3": True, "C4": True}
    assert verifier.evaluate(boundary, message("Shall I cancel order #A1?"), consent_check=check)["reward"] == 0
    del boundary["eligibility"]["consent_context"]
    assert verifier.evaluate(boundary, message("Shall I cancel order #A1?"), consent_check=check)["reward"] == 0


def test_k4_native_amount_cue_blocks_a_stale_missing_consent_label():
    boundary = spec("K4")
    boundary["eligibility"]["native_amount"] = "13.36"
    boundary["eligibility"]["consent_context"] = [
        {"type": "message", "role": "assistant", "content": "The $13.36 refund. Shall I proceed?"},
        {"type": "message", "role": "user", "content": "Yes"},
    ]

    def must_not_grade(_):
        raise AssertionError("A prior yes must stop K4 before content grading")

    assert verifier.evaluate(boundary, message("Do you confirm?"), consent_check=must_not_grade)["reward"] == 0
