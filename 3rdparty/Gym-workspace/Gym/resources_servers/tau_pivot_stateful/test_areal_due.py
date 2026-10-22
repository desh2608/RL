# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ordered native-ledger and audited early-write regression constraints."""

import hashlib
import json
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).parent / "reference"))

from recipes.tau_voice_rl_pivot import areal_v2
from recipes.tau_voice_rl_pivot.areal_native import Boundary
from recipes.tau_voice_rl_pivot.pinned_consent import ConsentContent


def boundary():
    value = Boundary.__new__(Boundary)
    value.gold = [
        {
            "name": "cancel_pending_order",
            "arguments": {"order_id": "#A1"},
            "requestor": "assistant",
            "done": False,
            "gold_index": 0,
            "delta": [],
        }
    ]
    value.tool_type = lambda call: "write" if call["name"].startswith("cancel") else "read"
    return value


def test_reference_reads_survive_write_only_task_criteria():
    value = boundary()
    value.supplement_reference_reads(
        [
            {
                "source": "agent",
                "type": "tool_call",
                "tool_name": "get_order_details",
                "arguments": '{"order_id":"#A1"}',
            },
            {
                "source": "agent",
                "type": "tool_call",
                "tool_name": "cancel_pending_order",
                "arguments": '{"order_id":"#A1"}',
            },
        ]
    )
    assert [gold["name"] for gold in value.remaining()] == ["get_order_details", "cancel_pending_order"]
    assert value.required_lookup() == {"name": "get_order_details", "arguments": {"order_id": "#A1"}}
    value.gold[0]["done"] = True
    assert value.required_lookup() is None
    assert value.gold[1]["reference_write_anchor"]


def test_unanchored_write_has_no_reference_due_proof():
    value = boundary()
    value.supplement_reference_reads([])
    assert value.gold[0]["reference_write_anchor"] is False


def test_mixed_speech_write_before_lookup_is_k6():
    value = boundary()
    value.supplement_reference_reads(
        [
            {
                "source": "agent",
                "type": "tool_call",
                "tool_name": "get_order_details",
                "arguments": {"order_id": "#A1"},
            },
            {
                "source": "agent",
                "type": "tool_call",
                "tool_name": "cancel_pending_order",
                "arguments": {"order_id": "#A1"},
            },
        ]
    )
    value.native_check = lambda call: {"feasible": True, "step_due": True}
    result = value.inspect(
        [
            {"type": "message", "role": "assistant", "content": "I cancelled the order."},
            {"type": "function_call", "name": "cancel_pending_order", "arguments": {"order_id": "#A1"}},
        ],
        [{"type": "message", "role": "user", "content": "Order #A1"}],
    )
    assert result["error_kind"] == "K6"
    assert result["gold"]["name"] == "get_order_details"


@pytest.mark.parametrize(
    "case",
    json.loads((Path(__file__).parent / "consent_audit_turn52.json").read_text()),
    ids=lambda c: str(c["audit_index"]),
)
def test_audited_contexts_require_native_due_proof(case):
    proof = dict(
        kind="K4",
        proven=True,
        prefix_replayed=True,
        state_hash_at_boundary=case["state_hash_at_boundary"],
        consent_context=case["messages"],
        valid_prior_consent=False,
    )
    spec = dict(
        verifier_version="areal-v2:INBOX35-regression",
        error_kind="K4",
        gold=case["gold"],
        eligibility=proof,
        consent_checker_sha256="must-not-run",
    )
    graded = []
    result = areal_v2.evaluate(
        spec,
        [{"type": "message", "role": "assistant", "content": "May I proceed?"}],
        native_check=lambda call: {"feasible": True, "write_due": False},
        consent_check=lambda question: graded.append(question),
    )
    assert result["reward"] == 0
    assert not graded


def test_checker_bytes_survive_source_directory_removal(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    data = b"# frozen checker bytes\n"
    (source / "areal_policy_checks.py").write_bytes(data)
    checker = ConsentContent(
        domain="retail",
        gold={"name": "cancel_pending_order", "arguments": {}},
        initial_db={},
        history=[],
        native_result="{}",
        checker_root=source,
        checker_files={"areal_policy_checks.py": hashlib.sha256(data).hexdigest()},
    )
    (source / "areal_policy_checks.py").unlink()
    source.rmdir()
    restored = checker.local_checker_root()
    assert (restored / "areal_policy_checks.py").read_bytes() == data
    assert not str(restored).startswith("/lustre/")


def test_premature_close_before_lookup_is_k7():
    value = boundary()
    value.supplement_reference_reads(
        [
            {
                "source": "agent",
                "type": "tool_call",
                "tool_name": "get_order_details",
                "arguments": {"order_id": "#A1"},
            },
            {
                "source": "agent",
                "type": "tool_call",
                "tool_name": "cancel_pending_order",
                "arguments": {"order_id": "#A1"},
            },
        ]
    )
    value.native_check = lambda call: {"feasible": True, "step_due": True}
    result = value.inspect(
        [{"type": "function_call", "name": "end_call", "arguments": {}}],
        [{"type": "message", "role": "user", "content": "Order #A1"}],
    )
    assert result["error_kind"] == "K7"
    assert result["gold"]["name"] == "get_order_details"


@pytest.mark.parametrize("same_identity", [True, False])
def test_alternative_native_auth_discharges_only_same_identity(same_identity):
    from types import SimpleNamespace

    value = boundary()
    value.gold = [
        {
            "name": "find_user_id_by_name_zip",
            "arguments": {"zip": "60614"},
            "requestor": "assistant",
            "done": False,
            "gold_index": 0,
        }
    ]
    value.env = {}
    value._native_cache = {}
    value.tool_type = lambda call: "auth"
    value.execute = lambda env, call: SimpleNamespace(
        error=False,
        content=("native-user-1" if call["name"] == "find_user_id_by_email" or same_identity else "native-user-2"),
    )
    result = value.replay_call(
        {"name": "find_user_id_by_email", "arguments": {"email": "a@example.net"}}, "native-user-1"
    )
    assert value.gold[0]["done"] is same_identity
    assert result["gold_auth_result_equivalence"] == ([0] if same_identity else [])
