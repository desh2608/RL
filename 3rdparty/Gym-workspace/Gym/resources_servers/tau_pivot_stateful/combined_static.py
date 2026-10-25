# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned G6 predicates alongside the unchanged v2 native worker dispatch."""

import hashlib
import json
import sys
from functools import lru_cache
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "reference"))
from recipes.tau_voice_rl_pivot import g6_error_kinds, g6_structural

from resources_servers.tau_pivot_stateful import afterquery_v21


@lru_cache(maxsize=8)
def load_entry(path: str, expected: str) -> dict:
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("Combined static entry hash differs")
    entry = json.loads(data)
    if entry.get("mode") != "combined_static":
        raise ValueError("Static entry mode differs")
    return entry


@lru_cache(maxsize=32)
def load_table(path: str, expected: str) -> dict:
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("AfterQuery field table hash differs")
    return json.loads(data)


def verify(item: dict, ref: dict, output: list[dict]) -> dict:
    entry = load_entry(item["entry"], item["entry_sha256"])
    for key in ("record_id", "dataset", "verifier_version", "expected_turn_kind"):
        if entry[key] != ref[key]:
            raise ValueError("Combined verification reference mismatch: " + key)
    spec = entry["verification"]
    evaluator = entry["evaluator"]
    if evaluator == "g6_error_kinds":
        if spec.get("contract") != g6_error_kinds.G_CONTRACT:
            raise ValueError("G6 error-kind contract differs")
        if spec.get("state_hash_at_boundary") != entry["state_hash_at_boundary"]:
            raise ValueError("G6 boundary hash differs")
        return g6_error_kinds.evaluate(spec, output)
    if evaluator == "g6_refusal":
        boundary = entry["state_hash_at_boundary"]
        return g6_structural.refusal(spec, output, state_before=boundary, state_after=boundary)
    if evaluator == "afterquery_v21":
        field = entry["field_table"]
        cache = Path(item["entry"]).parent.parent
        table = load_table(str(cache / field["path"]), field["sha256"])
        expected = afterquery_v21.spec_for(afterquery_v21.version(table))
        if spec != expected:
            raise ValueError("AfterQuery specification/field-table version differs")
        return afterquery_v21.evaluate(spec, output, entry["v2_state"], table)
    raise ValueError("Unknown combined static evaluator")
