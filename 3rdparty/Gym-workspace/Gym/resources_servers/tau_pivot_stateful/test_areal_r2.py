# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""R2 binds actual audited caller confirmations; earlier specs stay unchanged."""

import copy
import json
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).parent / "reference"))
from recipes.tau_voice_rl_pivot import areal_v2
from recipes.tau_voice_rl_pivot.confirmed_write import confirmed_write, conflicts, discussion_target


CASES = json.loads(Path(__file__).with_name("r2_audit_cases.json").read_text())["cases"]
BAD = {("K1", 1), ("K1", 2), ("K1", 4), ("K1", 5), ("K2", 0), ("K2", 2), ("K2", 3), ("K2", 4), ("K2", 5)}


@pytest.mark.parametrize(
    "case", [x for x in CASES if (x["kind"], x["index"]) in BAD], ids=lambda c: c["kind"] + "-" + str(c["index"])
)
def test_fable56_harmful_gold_write_is_held(case):
    r = confirmed_write(case["prefix"], case["gold"])
    assert r["ok"] is False and r["reason"] == "upstream_offer_divergence"


@pytest.mark.parametrize(
    "case", [x for x in CASES if x["kind"] == "K1" and x["index"] in [0, 3]], ids=lambda c: str(c["index"])
)
def test_same_flights_confirmation_has_a_real_public_binding(case):
    assert confirmed_write(case["prefix"], case["gold"])["ok"] is True


def test_latest_order_question_controls_k4_target():
    c = next(x for x in CASES if x["kind"] == "K4" and x["index"] == 6)
    r = discussion_target(c["prefix"], c["gold"])
    assert not r["ok"] and r["targets"] == ["#H3972275"]


@pytest.mark.parametrize(
    "case", [x for x in CASES if x["kind"] == "K5" and x["index"] in [0, 1, 5]], ids=lambda c: str(c["index"])
)
def test_profile_address_offer_cannot_authorize_older_profile_values(case):
    r = confirmed_write(case["prefix"], case["gold"])
    assert not r["ok"] and any(x["field"] in {"address1", "state", "zip"} for x in r["conflicts"])


def test_confirmation_for_two_explicit_actions_can_authorize_the_remaining_one():
    c = next(x for x in CASES if x["kind"] == "K5" and x["index"] == 2)
    assert confirmed_write(c["prefix"], c["gold"])["ok"]


def make_spec(kind, gold, prefix):
    return dict(
        contract="G5-INBOX37-R2",
        verifier_version="areal-v2:r2-test",
        error_kind=kind,
        gold=gold,
        eligibility=dict(
            kind=kind,
            proven=True,
            prefix_replayed=True,
            state_hash_at_boundary="bound",
            consent_context=prefix,
            r2_remaining_steps=[dict(**gold, tool_type="write")],
        ),
    )


@pytest.mark.parametrize("case", [x for x in CASES if x["kind"] == "K7"], ids=lambda c: str(c["index"]))
def test_k7_unconfirmed_write_never_becomes_positive(case):
    s = make_spec("K7", case["gold"], case["prefix"])
    out = [dict(type="function_call", **case["gold"])]
    assert areal_v2.evaluate(s, out, native_check=lambda c: {"feasible": True})["reward"] == 0
    assert (
        areal_v2.evaluate(
            s, [dict(type="message", role="assistant", content="I can keep helping with this request.")]
        )["reward"]
        == 1
    )
    for name in ["end_call", "transfer_to_human_agents"]:
        assert areal_v2.evaluate(s, [dict(type="function_call", name=name, arguments={})])["reward"] == 0


def test_k7_read_and_confirmed_write_are_positive_but_unrelated_calls_are_not():
    gold = dict(name="cancel_pending_order", arguments=dict(order_id="#A1", reason="no longer needed"))
    prefix = [
        dict(type="message", role="assistant", content="Shall I cancel order #A1 for no longer needed?"),
        dict(type="message", role="user", content="Yes"),
    ]
    s = make_spec("K7", gold, prefix)
    read = dict(name="get_order_details", arguments=dict(order_id="#B2"))
    s["eligibility"]["r2_remaining_steps"].append(dict(**read, tool_type="read"))
    assert (
        areal_v2.evaluate(s, [dict(type="function_call", **gold)], native_check=lambda c: {"feasible": True})["reward"]
        == 1
    )
    assert areal_v2.evaluate(s, [dict(type="function_call", **read)])["reward"] == 1
    assert (
        areal_v2.evaluate(s, [dict(type="function_call", name="get_order_details", arguments={"order_id": "#OTHER"})])[
            "reward"
        ]
        == 0
    )
    stale = copy.deepcopy(s)
    stale["eligibility"]["prefix_replayed"] = False
    assert areal_v2.evaluate(stale, [dict(type="message", role="assistant", content="I can help.")])["reward"] == 0


def test_explicit_old_to_new_flight_only_binds_new_value():
    prefix = [
        dict(
            type="function_call_output",
            output=json.dumps([dict(flight_number="OLD123"), dict(flight_number="NEW456")]),
        )
    ]
    gold = dict(
        name="update_reservation_flights", arguments=dict(flights=[dict(flight_number="NEW456", date="2024-05-20")])
    )
    assert not conflicts(prefix, "Change flight from OLD123 to NEW456 on May 20.", gold)
    assert conflicts(prefix, "Change flight to OLD123 on May 20.", gold)


def test_telecom_write_cannot_use_a_different_confirmed_line():
    gold = dict(name="refuel_data", arguments=dict(line_id="L1855", gb_amount=2))
    prefix = [
        dict(type="function_call_output", output=json.dumps([dict(line_id="L1855"), dict(line_id="L9999")])),
        dict(type="message", role="assistant", content="Shall I add 2 GB of data to line L9999?"),
        dict(type="message", role="user", content="Yes"),
    ]
    r = confirmed_write(prefix, gold)
    assert not r["ok"] and r["detail"] == "confirmed_target_mismatch"
    prefix[-2]["content"] = "Shall I add 2 GB of data to line L1855?"
    assert confirmed_write(prefix, gold)["ok"]


def test_unresolved_line_identity_is_a_hold_but_unique_public_identity_can_bind():
    gold = dict(name="refuel_data", arguments=dict(line_id="L1855", gb_amount=2))
    prefix = [
        dict(type="function_call_output", output=json.dumps([dict(line_id="L1855"), dict(line_id="L9999")])),
        dict(type="message", role="assistant", content="Shall I add 2 GB of data?"),
        dict(type="message", role="user", content="Yes"),
    ]
    assert not confirmed_write(prefix, gold)["ok"]
    prefix[0]["output"] = json.dumps([dict(line_id="L1855")])
    assert confirmed_write(prefix, gold)["ok"]
