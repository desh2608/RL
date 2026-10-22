# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Grade K4 question content with exact approved bytes, never infer prior consent.

Temporary module bindings isolate each pinned closure from the native runtime's
SDG module. Each native worker processes one request at a time. Original files
and their import statements remain byte-identical.
"""

import hashlib
import importlib.util
import json
import sys
import types
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def isolated_policy(root: Path, files: dict):
    """Bind a checked, node-local policy closure for the duration of a check."""
    for name, expected in files.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Consent checker bytes changed: " + name)
    prefix = "recipes.tau_voice_sdg"
    saved = {key: value for key, value in sys.modules.items() if key == prefix or key.startswith(prefix + ".")}
    for key in saved:
        del sys.modules[key]
    package = types.ModuleType(prefix)
    package.__path__ = [str(root)]
    sys.modules[prefix] = package
    try:
        spec = importlib.util.spec_from_file_location(prefix + ".areal_policy_checks", root / "areal_policy_checks.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        yield module
    finally:
        for key in list(sys.modules):
            if key == prefix or key.startswith(prefix + "."):
                del sys.modules[key]
        sys.modules.update(saved)


class ConsentContent:
    """Pinned question grader bound to one native-replayed boundary."""

    def __init__(
        self,
        *,
        domain: str,
        gold: dict,
        initial_db: dict,
        history: list[dict],
        native_result: str,
        checker_root: Path,
        checker_files: dict,
    ) -> None:
        if domain not in {"airline", "retail", "telecom"}:
            raise ValueError("Unsupported consent domain")
        self.domain = domain
        self.gold = {key: gold[key] for key in ("name", "arguments")}
        self.initial_db = initial_db
        self.history = history
        self.native_result = native_result
        self.checker_root = checker_root
        self.checker_files = checker_files

    def __call__(self, question: str) -> dict:
        # This hypothetical yes assesses question contents only. It is never
        # passed to the consent-given detector or committed to native state.
        messages = [
            *self.history,
            {"role": "assistant", "content": question},
            {"role": "user", "content": "Yes"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "k4-content-only",
                        "function": {"name": self.gold["name"], "arguments": json.dumps(self.gold["arguments"])},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "k4-content-only", "content": self.native_result},
        ]
        with isolated_policy(self.checker_root, self.checker_files) as policy:
            checked = policy.checks(
                messages, {"domain_name": self.domain, "customer_scenario": {}, "policy": ""}, self.initial_db
            )["writes"][-1]
        if self.domain == "telecom":
            return {
                "C1": True,
                "C2": True,
                "C3": checked["gates"]["C3"],
                "C4": True,
                "native_policy_C1": checked["gates"]["C1"],
                "native_policy_C6": checked["gates"]["C6"],
            }
        return {key: checked["gates"][key] for key in ("C1", "C2", "C3", "C4")}
