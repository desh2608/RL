# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Consent given is independent of detailed SDG readback quality."""

import json
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).parent / "reference"))
from recipes.tau_voice_rl_pivot.consent_given import detect


G_GOLD = {"name": "update_reservation_flights", "arguments": {"reservation_id": "ABC123", "cabin": "business"}}


def msg(role, content):
    return {"type": "message", "role": role, "content": content}


@pytest.mark.parametrize(
    "offer",
    [
        "Shall I update your flight?",
        "Reservation ABC123, business cabin. Confirm?",
        "May I change this?",
        "ABC123: does that sound okay?",
    ],
)
def test_action_or_detail_is_enough(offer):
    result = detect([msg("assistant", offer), msg("user", "Yes, proceed.")], G_GOLD)
    assert result["valid_prior_consent"]
    assert result["witness"]["reply_index"] == 1


@pytest.mark.parametrize(
    "reply",
    [
        "No",
        "Yes, but don't change it.",
        "Yes, wait.",
        "Yes, use my other card instead.",
        "Yes, change the date to May 20.",
        "Yes, add another bag.",
        "Not yet",
        "Yes, before that check the price.",
    ],
)
def test_negation_and_change_are_not_consent(reply):
    assert not detect([msg("assistant", "May I update ABC123?"), msg("user", reply)], G_GOLD)["valid_prior_consent"]


@pytest.mark.parametrize(
    "offer",
    [
        "Which cabin would you like, business or economy?",
        "May I cancel ABC123?",
        "May I return order ABC123?",
        "Is your email correct?",
        "ABC123 has business cabin.",
        "Your reservation ABC123 can be updated. May I explain?",
        "May I update reservation DEF456?",
    ],
)
def test_choice_different_action_and_unrelated_yes(offer):
    assert not detect([msg("assistant", offer), msg("user", "Yes")], G_GOLD)["valid_prior_consent"]


def test_change_after_yes_and_segment_write_reset():
    initial = [msg("assistant", "May I update ABC123?"), msg("user", "Yes")]
    assert not detect(initial + [msg("user", "Actually, use economy instead.")], G_GOLD)["valid_prior_consent"]
    committed = {"type": "function_call", **G_GOLD}
    assert not detect(initial + [committed], G_GOLD)["valid_prior_consent"]
    assert detect(initial + [msg("assistant", "Let me check the fare.")], G_GOLD)["valid_prior_consent"]
    renewed = initial + [msg("user", "Wait"), msg("assistant", "Shall I update ABC123?"), msg("user", "Yes")]
    assert detect(renewed, G_GOLD)["valid_prior_consent"]


def test_only_next_caller_reply_counts():
    messages = [msg("assistant", "May I update ABC123?"), msg("user", "How much?"), msg("user", "Yes")]
    assert not detect(messages, G_GOLD)["valid_prior_consent"]
    messages = [
        msg("assistant", "May I update ABC123?"),
        msg("assistant", "Is your email correct?"),
        msg("user", "Yes"),
    ]
    assert not detect(messages, G_GOLD)["valid_prior_consent"]


def test_batch_permission_survives_a_different_write():
    initial = [msg("assistant", "Shall I update ABC123 and DEF456?"), msg("user", "Yes")]
    other = {
        "type": "function_call",
        "name": G_GOLD["name"],
        "arguments": {"reservation_id": "DEF456", "cabin": "business"},
    }
    assert detect(initial + [other], G_GOLD)["valid_prior_consent"]


@pytest.mark.parametrize(
    "offer,reply",
    [
        ("I'll cancel this order. Shall I cancel it?", "Yes, please cancel the whole order."),
        ("Shall I change the flight for $130 refunded?", "Yes, I confirm that change."),
        ("May I update this to business class?", "Yes, proceed with the upgrade."),
    ],
)
def test_restatement_is_assent_not_a_new_request(offer, reply):
    gold = {"name": "cancel_pending_order", "arguments": {}} if "cancel" in offer else G_GOLD
    assert detect([msg("assistant", offer), msg("user", reply)], gold)["valid_prior_consent"]


def test_k5_requires_an_actual_reask():
    from recipes.tau_voice_rl_pivot.consent_given import is_reask

    assert is_reask("Shall I update ABC123?", G_GOLD)
    assert is_reask("Please confirm the change to ABC123.", G_GOLD)
    assert not is_reask("To confirm: ABC123 will change to business class.", G_GOLD)


def test_native_amount_or_public_item_name_identifies_the_write():
    offer = [msg("assistant", "The $13.36 refund to your card. Shall I proceed?"), msg("user", "Yes")]
    assert not detect(offer, G_GOLD)["valid_prior_consent"]
    assert detect(offer, G_GOLD, native_amount="13.36")["valid_prior_consent"]
    assert not detect(offer, G_GOLD, native_amount="19.36")["valid_prior_consent"]
    assert not detect(offer, G_GOLD, native_amount="13")["valid_prior_consent"]
    assert detect([msg("assistant", "$5. Confirm?"), msg("user", "Yes")], G_GOLD, native_amount="5")[
        "valid_prior_consent"
    ]
    gold = {"name": "exchange_delivered_order_items", "arguments": {"item_ids": ["12345"]}}
    prefix = [{"type": "function_call_output", "output": '{"items":[{"item_id":"12345","name":"Coffee Maker"}]}'}]
    prefix += [msg("assistant", "The Coffee Maker. Confirm?"), msg("user", "Yes")]
    assert detect(prefix, gold)["valid_prior_consent"]


@pytest.mark.parametrize(
    "case",
    json.loads((Path(__file__).parent / "consent_audit_turn48.json").read_text()),
    ids=lambda c: f"{c['audit_side']}-{c['audit_index']}",
)
def test_fable_turn48_audit(case):
    actual = detect(case["messages"], case["gold"], native_amount=case["native_amount"])
    assert actual["valid_prior_consent"] == case["expected"], actual
