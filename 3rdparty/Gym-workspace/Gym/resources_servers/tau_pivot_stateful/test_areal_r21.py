# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""INBOX38 live/cache equivalence, spoken negatives, required-info regressions."""

import copy
import json
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).parent / "reference"))
from recipes.tau_voice_rl_pivot import areal_r2, areal_r21, areal_v2


CASES = json.loads(Path(__file__).with_name("r21_audit_cases.json").read_text())["cases"]


def k7_spec():
    return dict(
        contract="G5-INBOX38-R21",
        verifier_version="areal-v2:r21-test",
        error_kind="K7",
        gold=dict(name="get_user_details", arguments={"user_id": "specialist_123"}),
        eligibility=dict(
            kind="K7",
            proven=True,
            prefix_replayed=True,
            state_hash_at_boundary="hash",
            consent_context=[],
            r2_remaining_steps=[
                dict(name="get_user_details", arguments={"user_id": "specialist_123"}, tool_type="read")
            ],
        ),
    )


@pytest.mark.parametrize(
    "text",
    [
        "Let me transfer you to a human agent.",
        "I'll transfer you.",
        "A specialist can help.",
        "Our representative will answer.",
        "Your supervisor can explain.",
        "I will escalate this.",
        "Escalating this now.",
        "Hold on.",
        "Goodbye!",
        "Have a great day.",
        "Anything else today?",
        "Is there anything else I can do for you today?",
        "A HUMAN-AGENT will help.",
        "I am not going to transfer you.",
    ],
)
def test_spoken_handoff_language_cannot_pass_k7(text):
    spec = k7_spec()
    output = [dict(type="message", role="assistant", content=text)]
    prior = areal_r2.evaluate(spec, output)
    assert prior["reward"] == 1
    live = areal_v2.evaluate(spec, output)
    assert live["reward"] == 0 and live["r21_handoff_matches"]
    assert live == areal_r21.postprocess(spec, output, prior)


def test_k7_continue_and_remaining_read_are_unchanged():
    spec = k7_spec()
    for output in [
        [dict(type="message", role="assistant", content="I can continue helping with your request.")],
        [dict(type="function_call", **spec["gold"])],
    ]:
        prior = areal_r2.evaluate(spec, output)
        assert areal_v2.evaluate(spec, output) == areal_r21.postprocess(spec, output, prior)
        assert areal_v2.evaluate(spec, output)["reward"] == 1


@pytest.mark.parametrize("case", CASES, ids=lambda c: str(c["index"]))
def test_fable58_required_refund_method_is_not_a_reask(case):
    qualification = areal_r21.qualify(case["spec"])
    assert (qualification["status"] == "hold") == (case["index"] == 0)
    if case["index"] == 0:
        assert qualification["reason"] == "K5_required_information_not_in_offer"
        assert qualification["witness"]["fields"] == ["payment_method_id"]


@pytest.mark.parametrize(
    "key,offer,question",
    [
        ("reason", "Shall I cancel order #A1?", "What is the reason for cancellation?"),
        ("payment_method_id", "Shall I return order #A1?", "Which payment method should receive the refund?"),
    ],
)
def test_requested_required_parameter_must_be_in_confirmed_offer(key, offer, question):
    args = {key: "no longer needed" if key == "reason" else "credit_card_123"}
    assert areal_r21.requested_fields(question, args) == [key]
    assert not areal_r21.offer_states_value(offer, key, args[key])


def test_stated_refund_method_and_plain_confirmation_are_not_missing_information():
    key = "payment_method_id"
    assert areal_r21.offer_states_value("Return order #A1, refund to original card. Confirm?", key, "credit_card_123")
    assert (
        areal_r21.requested_fields("Return order #A1, refund to credit card. Confirm?", {key: "credit_card_123"}) == []
    )
    spec = copy.deepcopy(CASES[0]["spec"])
    spec["eligibility"].pop("r21_failing_output")
    assert areal_r21.qualify(spec)["reason"] == "K5_missing_failing_turn_witness"
