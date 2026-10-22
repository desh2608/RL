# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""R1 commuting-write, consent-target and lenient-K7 behavior."""

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


sys.path.insert(0, str(Path(__file__).parent / "reference"))
from recipes.tau_voice_rl_pivot import areal_r1, areal_v2
from recipes.tau_voice_rl_pivot.areal_native import Boundary


def call(name, **args):
    return {"name": name, "arguments": args}


def output(action):
    return [{"type": "function_call", **action}]


def msg(text):
    return [{"type": "message", "role": "assistant", "content": text}]


def boundary():
    b = Boundary.__new__(Boundary)
    b.recovery_r1 = True
    b.gold = [
        dict(**call("get_order_details", order_id="#OTHER"), done=False, requestor="assistant", gold_index=0),
        dict(
            **call("cancel_pending_order", order_id="#A1", reason="no longer needed"),
            done=False,
            requestor="assistant",
            gold_index=1,
        ),
    ]
    b.tool_type = lambda c: "read" if c["name"].startswith("get_") else "write"
    b.native_check = lambda c: {"feasible": True, "state_ready": True, "step_due": True, "write_due": True}
    b.charge = lambda c, r: None
    b.domain = "retail"
    return b


def prefix(order="#A1", reason="no longer needed"):
    return [
        {"type": "message", "role": "assistant", "content": f"Shall I cancel order {order} for {reason}?"},
        {"type": "message", "role": "user", "content": "Yes"},
    ]


def test_confirmed_gold_write_is_not_k6_before_an_unrelated_lookup():
    b = boundary()
    action = areal_r1.call_of(b.gold[1])
    assert b.inspect(output(action), prefix())["status"] == "continue"
    assert (
        b.inspect_student(output(action), prefix(), call("get_order_details", order_id="#OTHER"))["status"]
        == "continue"
    )
    assert areal_r1.needed_lookup(b, action, prefix()) is None


def test_remaining_alternative_read_is_not_an_error():
    b = boundary()
    b.gold.append(
        dict(**call("get_order_details", order_id="#SECOND"), done=False, requestor="assistant", gold_index=2)
    )
    assert (
        b.inspect_student(
            output(call("get_order_details", order_id="#SECOND")), [], call("get_order_details", order_id="#OTHER")
        )["status"]
        == "continue"
    )


def test_native_ready_uses_state_not_trace_position():
    b = boundary()
    b._native_cache = {}
    b._native_check_uncached = lambda c: {"feasible": True, "matched_gold_index": 1}
    # Restore the real method, replacing only native execution in this unit.
    del b.native_check
    r = b.native_check(areal_r1.call_of(b.gold[1]))
    assert r["state_ready"] and r["write_due"]
    assert r["trace_step_due"] is False


@pytest.mark.parametrize(
    "offer,accepted",
    [("Shall I cancel order #A1?", True), ("Shall I cancel order #A2?", False), ("Shall I cancel the order?", False)],
)
def test_same_target_offer(offer, accepted):
    assert areal_r1.same_target_offer(offer, call("cancel_pending_order", order_id="#A1")) is accepted


def test_profile_address_needs_explicit_profile_target():
    gold = call("modify_user_address", user_id="u1", city="Boston")
    assert areal_r1.same_target_offer("Shall I update your default address?", gold)
    assert not areal_r1.same_target_offer("Shall I update order #A1's shipping address?", gold)


def test_k5_requires_same_target_and_every_argument_public():
    b = boundary()
    reask = msg("Shall I cancel order #A1?")
    assert b.inspect(reask, prefix())["error_kind"] == "K5"
    assert b.inspect(reask, prefix("#A2"))["status"] != "candidate"
    assert b.inspect(reask, prefix(reason="unspecified"))["status"] != "candidate"


def test_grounded_non_gold_write_cannot_be_k6():
    b = boundary()
    wrong = call("cancel_pending_order", order_id="#A1", reason="different")
    assert areal_r1.needed_lookup(b, wrong, prefix() + msg("different")) is None


def k7_spec():
    return dict(
        contract="G5-INBOX36-R1",
        verifier_version="areal-v2:r1-test",
        error_kind="K7",
        gold=call("get_order_details", order_id="#A1"),
        eligibility=dict(
            kind="K7",
            proven=True,
            prefix_replayed=True,
            state_hash_at_boundary="bound",
            ordered_ledger=[
                dict(**call("get_order_details", order_id="#A1"), done=False, requestor="assistant"),
                dict(**call("cancel_pending_order", order_id="#B2"), done=False, requestor="assistant"),
            ],
        ),
    )


def test_k7_accepts_spoken_explanation_and_any_remaining_gold():
    s = k7_spec()
    assert areal_v2.evaluate(s, msg("Basic economy cannot be changed."))["reward"] == 1
    assert areal_v2.evaluate(s, output(call("cancel_pending_order", order_id="#B2")))["reward"] == 1
    for action in [call("end_call"), call("transfer_to_human_agents"), call("get_order_details", order_id="#WRONG")]:
        assert areal_v2.evaluate(s, output(action))["reward"] == 0
    assert areal_v2.evaluate(s, msg("Explanation") + output(call("end_call")))["reward"] == 0
    stale = copy.deepcopy(s)
    stale["eligibility"]["prefix_replayed"] = False
    assert areal_v2.evaluate(stale, msg("Explanation"))["reward"] == 0


def test_structural_wrong_write_tool_can_recover_k1():
    b = boundary()
    b.env = {}
    b.execute = lambda env, c: SimpleNamespace(error=True, content="Error: invalid write")
    result = b.inspect_student(
        output(call("modify_pending_order_payment", order_id="#A1", payment_method_id="p2")),
        prefix(),
        areal_r1.call_of(b.gold[1]),
    )
    assert result["error_kind"] == "K1"
    assert result["witness"] == "structural_write_mismatch_and_native_error"


def test_k2_needs_matching_tool_and_target_and_structural_argument():
    b = boundary()
    gold = call("update_reservation_baggages", reservation_id="ABC123", total_baggages=1, nonfree_baggages=0)
    b.gold = [dict(**gold, done=False, requestor="assistant", gold_index=0)]
    b.env = {}
    b.execute = lambda env, c: SimpleNamespace(error=True, content="Error: wrong bags")
    context = [
        {
            "type": "message",
            "role": "assistant",
            "content": "Shall I update reservation ABC123 to 1 total bag and 0 nonfree bags?",
        },
        {"type": "message", "role": "user", "content": "Yes"},
    ]
    wrong = call("update_reservation_baggages", reservation_id="ABC123", total_baggages=2, nonfree_baggages=0)
    assert b.inspect_student(output(wrong), context, gold)["error_kind"] == "K2"
    other = call("update_reservation_baggages", reservation_id="OTHER", total_baggages=2, nonfree_baggages=0)
    assert b.inspect_student(output(other), context, gold)["status"] == "hold"


def test_k6_requires_a_read_supplying_missing_target_arguments():
    b = boundary()
    gold = call("modify_pending_order_payment", order_id="#A1", payment_method_id="p2")
    b.gold[1].update(gold)
    b.gold[0].update(call("get_order_details", order_id="#A1"))
    b.native_check = lambda c: {
        "feasible": True,
        "step_due": True,
        "native_result": '{"order_id":"#A1","payment_method_id":"p2"}',
    }
    wrong = call("modify_pending_order_payment", order_id="#A1", payment_method_id="p3")
    result = areal_r1.needed_lookup(b, wrong, [{"role": "user", "content": "Order #A1"}])
    assert result["error_kind"] == "K6"
    b.native_check = lambda c: {"feasible": True, "step_due": True, "native_result": '{"order_id":"#OTHER"}'}
    assert areal_r1.needed_lookup(b, wrong, [{"role": "user", "content": "Order #A1"}]) is None


def test_r1_public_same_target_string_argument_with_native_error_is_k2():
    b = boundary()
    b.env = {}
    b.execute = lambda env, c: SimpleNamespace(error=True, content="Error: invalid reason")
    result = b.inspect_student(
        output(call("cancel_pending_order", order_id="#A1", reason="different reason")),
        prefix(),
        areal_r1.call_of(b.gold[1]),
    )
    assert result["error_kind"] == "K2"
