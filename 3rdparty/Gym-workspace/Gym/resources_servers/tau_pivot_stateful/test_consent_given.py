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


def test_addon_has_its_own_consent_scope():
    flight = {"name": "update_reservation_flights", "arguments": {"reservation_id": "ABC123"}}
    bags = {"name": "update_reservation_baggages", "arguments": {"reservation_id": "ABC123", "total_baggages": 2}}
    prefix = [
        msg("assistant", "May I change the flight on ABC123?"),
        msg("user", "Yes, and please add the checked bags."),
    ]
    assert detect(prefix, flight)["valid_prior_consent"]
    assert not detect(prefix, bags)["valid_prior_consent"]
    prefix += [msg("assistant", "Shall I add two checked bags to ABC123?"), msg("user", "Yes, please.")]
    assert detect(prefix, bags)["valid_prior_consent"]


@pytest.mark.parametrize(
    "case",
    json.loads((Path(__file__).parent / "consent_audit_turn50.json").read_text()),
    ids=lambda c: f"{c['audit_side']}-{c['audit_index']}",
)
def test_fable_turn50_asymmetric_rule(case):
    actual = detect(case["messages"], case["gold"], native_amount=case["native_amount"])
    if case["expected"] == "not_K4":
        assert not actual["k4_eligible"], actual
    else:
        assert actual["consent_label"] == case["expected"], actual
    assert not (actual["k4_eligible"] and actual["write_segment"]["assents"])


@pytest.mark.parametrize(
    "assent",
    [
        "yes",
        "yeah",
        "yep",
        "sure",
        "okay",
        "ok",
        "sounds good",
        "that works",
        "that's fine",
        "fine",
        "go ahead",
        "proceed",
        "please do",
        "please update it",
        "let's go with it",
        "I'll take it",
        "I confirm",
        "confirm",
        "correct",
        "that's right",
    ],
)
def test_any_assent_blocks_k4_without_an_offer(assent):
    result = detect([msg("user", assent)], G_GOLD)
    assert result["consent_label"] == "hold"
    assert not result["k4_eligible"]
    assert not result["authorized_prior_consent"]


def test_question_after_assent_is_hold_until_clean_assent():
    prefix = [msg("assistant", "Shall I update ABC123?"), msg("user", "Yes. What's the fare difference?")]
    result = detect(prefix, G_GOLD)
    assert result["consent_label"] == "hold"
    assert not result["authorized_prior_consent"]
    assert not result["k4_eligible"]
    prefix += [msg("assistant", "It costs $50 more. Shall I update ABC123?"), msg("user", "Yes, proceed.")]
    assert detect(prefix, G_GOLD)["authorized_prior_consent"]


def test_information_and_other_item_do_not_supply_assent():
    prefix = [msg("user", "Change reservation ABC123."), msg("assistant", "Which card for ABC123?")]
    assert detect(prefix + [msg("user", "The card ending 1952.")], G_GOLD)["k4_eligible"]
    assert detect(prefix + [msg("user", "Okay, now for reservation DEF456.")], G_GOLD)["k4_eligible"]
    assert detect(prefix + [msg("user", "Do not proceed.")], G_GOLD)["k4_eligible"]
    assert not detect(prefix + [msg("user", "Not 6952. But yes, proceed with ABC123.")], G_GOLD)["k4_eligible"]


def test_clarification_never_erases_segment_assent():
    prefix = [
        msg("user", "Change ABC123."),
        msg("assistant", "Business on ABC123, $50 more."),
        msg("user", "I'll take it."),
        msg("assistant", "Which card?"),
        msg("user", "Ending 1952."),
    ]
    result = detect(prefix, G_GOLD)
    assert result["consent_label"] == "given"
    assert not result["k4_eligible"]
    assert not result["authorized_prior_consent"]


def test_clearly_different_public_item_is_ignored():
    gold = {"name": "return_delivered_order_items", "arguments": {"item_ids": ["123"]}}
    prefix = [
        {
            "type": "function_call_output",
            "output": json.dumps(
                {"items": [{"item_id": "123", "name": "Coffee Maker"}, {"item_id": "456", "name": "Fleece Jacket"}]}
            ),
        },
        msg("user", "Return the Coffee Maker."),
        msg("assistant", "Which payment method?"),
        msg("user", "Okay, now for the Fleece Jacket."),
    ]
    result = detect(prefix, gold)
    assert result["k4_eligible"]
    assert result["write_segment"]["ignored"][0]["reason"] == "different_named_write"


def test_correction_of_card_number_does_not_hide_yes():
    prefix = [
        msg("user", "Change ABC123."),
        msg("assistant", "Business for ABC123."),
        msg("user", "It is not 6952 but yes proceed."),
    ]
    assert not detect(prefix, G_GOLD)["k4_eligible"]


def test_negation_of_plan_change_does_not_hide_refuel_assent():
    gold = {"name": "refuel_data", "arguments": {"line_id": "L1", "gb_amount": 2}}
    prefix = [
        msg("assistant", "Want to add data at $2 per GB?"),
        msg("user", "Yes, I'll add 2 GB of data, but I don't want to change my plan."),
    ]
    result = detect(prefix, gold)
    assert not result["k4_eligible"]
    assert result["consent_label"] == "given"


@pytest.mark.parametrize(
    "offer,reply,expected",
    [
        ("Shall I update ABC123?", "Yes, please proceed.", True),
        ("Business on ABC123 would cost $50 more.", "Please proceed.", False),
        ("Shall I update ABC123?", "Yes. What is the fee?", False),
        ("Shall I update ABC123?", "Yes, but use another card instead.", False),
    ],
)
def test_inbox35_k5_requires_question_offer(offer, reply, expected):
    assert detect([msg("assistant", offer), msg("user", reply)], G_GOLD)["k5_eligible"] is expected


def test_k5_never_uses_numeric_substring_to_accept_a_changed_date():
    gold = {"name": "update_reservation_flights", "arguments": {"reservation_id": "ABC123", "date": "2024-05-18"}}
    prefix = [msg("assistant", "Shall I change ABC123 to May 18, 2024?"), msg("user", "Yes, change it to May 20.")]
    assert not detect(prefix, gold)["k5_eligible"]


def test_k5_does_not_traverse_a_later_caller_change():
    prefix = [
        msg("assistant", "Shall I update ABC123?"),
        msg("user", "Yes, please proceed."),
        msg("assistant", "Let me check that."),
        msg("user", "Move it to May 20."),
    ]
    assert not detect(prefix, G_GOLD)["k5_eligible"]


def test_identity_question_is_not_k5_write_permission():
    prefix = [msg("assistant", "Is reservation ABC123 correct?"), msg("user", "Yes")]
    assert not detect(prefix, G_GOLD)["k5_eligible"]


def test_canonical_affirmative_cannot_switch_the_action_noun():
    prefix = [msg("assistant", "Shall I update ABC123?"), msg("user", "Yes, proceed with the cancellation.")]
    assert not detect(prefix, G_GOLD)["authorized_prior_consent"]
