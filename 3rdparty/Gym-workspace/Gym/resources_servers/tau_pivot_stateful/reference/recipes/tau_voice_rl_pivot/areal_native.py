# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""G5 native boundary replay, independent of SDG online verdicts.

Every task action, including reads, is retained. Native feasibility is distinct
from consent authorization. Non-reference reads and potentially commuting
writes require explicit handling; a raw mismatch is never a certificate.
"""

import copy
from pathlib import Path

from recipes.tau_voice_rl_pivot import areal_v2, replay_five
from recipes.tau_voice_rl_pivot.consent_given import detect, is_reask


class Boundary:
    """A native task environment and complete remaining gold action ledger."""

    def __init__(self, raw: dict, frozen: Path):
        from recipes.tau_voice_sdg.areal_gold_replay import constructor
        from recipes.tau_voice_sdg.areal_step_verifier import delta
        from tau2.data_model.tasks import InitialState, Task

        self.domain = raw["domain_name"]
        self.policy = raw["policy"]
        task = copy.deepcopy(raw["tau_task"])
        source = raw["source_artifacts"]
        asset = frozen / "assets" / ("areal-telecom" if self.domain == "telecom" else "areal")
        db = asset / source["db_path"]
        if replay_five.sha(db) != source["db_sha256"]:
            raise ValueError("Initial DB does not match pinned source")
        self.env = constructor(self.domain, str(db))()
        initial = Task.model_validate(task).initial_state or InitialState()
        self.env.set_state(initial.initialization_data, initial.initialization_actions, initial.message_history or [])
        self.gold = []
        shadow = copy.deepcopy(self.env)
        for index, action in enumerate(task["evaluation_criteria"].get("actions") or []):
            before = replay_five.state_of(shadow)
            result = self.execute(shadow, action)
            if self.failed(result):
                raise ValueError("Gold action not executable: " + action["name"])
            self.gold.append(
                {
                    "name": action["name"],
                    "arguments": action["arguments"],
                    "requestor": action.get("requestor", "assistant"),
                    "done": False,
                    "gold_index": index,
                    "delta": delta(before, replay_five.state_of(shadow)),
                }
            )
        self.expected_state = replay_five.state_of(shadow)
        self.criteria = task["evaluation_criteria"]
        self.initial_db_sha256 = source["db_sha256"]
        self.task = task

    @staticmethod
    def execute(env, call: dict):
        from tau2.data_model.message import ToolCall

        return env.get_response(
            ToolCall(
                id="g5-replay",
                name=call["name"],
                arguments=call["arguments"],
                requestor=call.get("requestor", "assistant"),
            )
        )

    @staticmethod
    def failed(result) -> bool:
        return bool(result.error or str(result.content).lower().startswith(("error:", "error ")))

    def tool_type(self, call: dict) -> str:
        toolkit = self.env.user_tools if call.get("requestor") == "user" else self.env.tools
        if toolkit is None or not toolkit.has_tool(call["name"]):
            return "unknown"
        kind = toolkit.tool_type(call["name"])
        return "auth" if call["name"].startswith("find_user_id") else kind

    def remaining(self) -> list[dict]:
        return [g for g in self.gold if not g["done"]]

    def native_check(self, call: dict) -> dict:
        """Execute on a copy and prove the remaining task can still complete."""
        from recipes.tau_voice_sdg.areal_step_verifier import delta

        call = {**call, "requestor": call.get("requestor", "assistant")}
        match = next(
            (
                g
                for g in self.remaining()
                if g["requestor"] == call["requestor"]
                and areal_v2.typed_equal(
                    {k: g[k] for k in ("name", "arguments")}, {k: call[k] for k in ("name", "arguments")}
                )
            ),
            None,
        )
        shadow = copy.deepcopy(self.env)
        before = replay_five.state_of(shadow)
        result = self.execute(shadow, call)
        changed = delta(before, replay_five.state_of(shadow))
        if self.failed(result):
            return {"feasible": False, "native_error": str(result.content), "matched_gold_index": None}
        if match is None:
            return {
                "feasible": False,
                "native_error": None,
                "matched_gold_index": None,
                "delta": changed,
                "hold": "non_reference_action_requires_independent_proof",
            }
        for gold in self.remaining():
            if gold is match:
                continue
            future = self.execute(shadow, gold)
            if self.failed(future):
                return {
                    "feasible": False,
                    "matched_gold_index": match["gold_index"],
                    "hold": "remaining_gold_not_executable",
                }
        basis = self.criteria.get("reward_basis") or ["DB"]
        complete = "DB" not in basis or replay_five.state_of(shadow) == self.expected_state
        if "ENV_ASSERTION" in basis:
            from tau2.data_model.tasks import EnvAssertion

            complete = complete and all(
                shadow.run_env_assertion(EnvAssertion.model_validate(assertion), raise_assertion_error=False)
                for assertion in self.criteria.get("env_assertions") or []
            )
        return {
            "feasible": complete,
            "matched_gold_index": match["gold_index"],
            "delta": changed,
            "remaining_completion_verified": complete,
            "native_result": result.content,
        }

    def replay_call(self, call: dict, recorded: str) -> dict:
        """Commit a recorded prefix call only after exact native result parity."""
        result = self.execute(self.env, call)
        if result.content != recorded:
            raise ValueError("Recorded tool result differs from native replay: " + call["name"])
        for gold in self.remaining():
            if gold["requestor"] == call.get("requestor", "assistant") and areal_v2.typed_equal(
                {k: gold[k] for k in ("name", "arguments")}, {k: call[k] for k in ("name", "arguments")}
            ):
                if not self.failed(result):
                    gold["done"] = True
                break
        return {
            "name": call["name"],
            "result_sha256": replay_five.digest(recorded),
            "native_error": self.failed(result),
        }

    def state(self) -> dict:
        return {"environment": replay_five.state_of(self.env), "ledger": self.gold}

    def consent_required(self, call: dict) -> bool:
        return self.tool_type(call) == "write" and (self.domain != "telecom" or call["name"] == "refuel_data")

    def inspect(self, output: list[dict], prefix: list[dict]) -> dict:
        """Propose supported boundaries; separate holds from proved failures.

        This is candidate generation. A proposed kind still needs pinned K4
        content controls, HTTP parity, reproduction and independent audit.
        """
        shape, candidate = areal_v2.one_turn(output)
        if shape == "invalid":
            return {"status": "hold", "reason": "unsupported_turn_shape"}
        if shape == "message":
            for gold in self.remaining():
                if gold["requestor"] != "assistant" or self.tool_type(gold) != "write":
                    continue
                consent = detect(prefix, gold)
                if consent["valid_prior_consent"] and is_reask(candidate, gold):
                    native = self.native_check(gold)
                    if native["feasible"]:
                        return {
                            "status": "candidate",
                            "error_kind": "K5",
                            "gold": gold,
                            "consent": consent,
                            "native": native,
                            "witness": "recorded_reask_after_consent",
                        }
            return {"status": "continue", "reason": "speech_facts_outside_K1_K7"}
        kind = self.tool_type(candidate)
        if candidate["name"] in areal_v2.G_CLOSE_TOOLS:
            remaining = [g for g in self.remaining() if g["requestor"] == "assistant"]
            return {
                "status": "hold" if remaining else "continue",
                "reason": "K7_needs_current_request_continuation_witness" if remaining else "no_gold_remaining",
            }
        if kind in {"read", "auth"}:
            return {"status": "continue", "reason": "read_alternatives_not_classified_as_errors"}
        if kind != "write":
            return {"status": "hold", "reason": "unknown_tool_type"}
        native = self.native_check(candidate)
        if not native["feasible"]:
            # A native exception or exact gold mismatch alone is insufficient
            # for K1/K2/K3. Do not mis-certify these as missing consent.
            return {"status": "hold", "reason": "write_needs_kind_specific_error_witness", "native": native}
        consent = detect(prefix, candidate)
        if self.consent_required(candidate) and not consent["valid_prior_consent"]:
            return {
                "status": "candidate",
                "error_kind": "K4",
                "gold": candidate,
                "consent": consent,
                "native": native,
                "witness": "feasible_write_without_consent",
            }
        return {"status": "continue", "reason": "native_gold_write_authorized", "native": native, "consent": consent}

    def inspect_student(self, output: list[dict], prefix: list[dict], reference_call: dict | None) -> dict:
        """Use a correct native reference turn as additional B0 evidence.

        A remaining alternative gold action is held rather than treated as an
        error. Only native effects that differ from completing the intended
        reference step support K1/K2; inconsequential argument differences stay
        held. K3 remains blocked by its separate adversarial gate.
        """
        shape, candidate = areal_v2.one_turn(output)
        if reference_call is None or shape != "call":
            return self.inspect(output, prefix)
        target = next(
            (
                g
                for g in self.remaining()
                if g["requestor"] == "assistant"
                and areal_v2.typed_equal({k: g[k] for k in ("name", "arguments")}, reference_call)
            ),
            None,
        )
        if target is None:
            return {"status": "hold", "reason": "reference_call_not_in_remaining_gold"}
        good = self.native_check(reference_call)
        if not good["feasible"]:
            return {"status": "hold", "reason": "reference_not_native_feasible", "native": good}
        consent = detect(prefix, reference_call)
        authorized = not self.consent_required(reference_call) or consent["valid_prior_consent"]
        if areal_v2.typed_equal(candidate, reference_call):
            return self.inspect(output, prefix)
        if any(
            areal_v2.typed_equal(candidate, {k: g[k] for k in ("name", "arguments")})
            for g in self.remaining()
            if g["requestor"] == "assistant"
        ):
            return {"status": "hold", "reason": "remaining_alternative_gold_action"}
        if candidate["name"] in areal_v2.G_CLOSE_TOOLS:
            if authorized:
                return {
                    "status": "candidate",
                    "error_kind": "K7",
                    "gold": reference_call,
                    "native": good,
                    "consent": consent,
                    "witness": "executable_current_reference_continues",
                }
            return {"status": "hold", "reason": "reference_write_needs_consent"}
        if not authorized or self.tool_type(reference_call) != "write" or self.tool_type(candidate) != "write":
            return {"status": "hold", "reason": "read_alternative_or_unauthorized_reference"}
        # S1: open-string-only changes (for example cancellation reasons) are
        # not mistakes. A closed scalar/identifier difference needs native
        # consequence evidence too; this narrow subset avoids schema guessing.
        if candidate["name"] == reference_call["name"]:
            changed = [
                key
                for key in candidate["arguments"].keys() | reference_call["arguments"].keys()
                if not areal_v2.typed_equal(candidate["arguments"].get(key), reference_call["arguments"].get(key))
            ]
            if not any(
                key.endswith(("_id", "_ids")) or isinstance(reference_call["arguments"].get(key), (int, float, bool))
                for key in changed
            ):
                return {"status": "hold", "reason": "open_or_unregistered_field_difference"}
            error_kind = "K2"
        else:
            error_kind = "K1"
        shadow = copy.deepcopy(self.env)
        result = self.execute(shadow, candidate)
        failing = self.failed(result)
        for gold in self.remaining():
            if gold is target:
                continue
            if self.failed(self.execute(shadow, gold)):
                failing = True
                break
        basis = self.criteria.get("reward_basis") or ["DB"]
        failing = failing or ("DB" in basis and replay_five.state_of(shadow) != self.expected_state)
        if not failing:
            return {"status": "hold", "reason": "no_native_consequence_witness"}
        return {
            "status": "candidate",
            "error_kind": error_kind,
            "gold": reference_call,
            "native": good,
            "consent": consent,
            "witness": "native_reference_replacement_fails_completion",
            "bad_native_result": result.content,
        }
