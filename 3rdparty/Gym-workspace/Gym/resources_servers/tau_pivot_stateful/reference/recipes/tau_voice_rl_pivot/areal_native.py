# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""G5 native boundary replay, independent of SDG online verdicts.

Every task action, including reads, is retained. Native feasibility is distinct
from consent authorization. Non-reference reads and potentially commuting
writes require explicit handling; a raw mismatch is never a certificate.
"""

import copy
import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Any

from recipes.tau_voice_rl_pivot import areal_v2, replay_five
from recipes.tau_voice_rl_pivot.consent_given import detect, is_reask


class Boundary:
    """A native task environment and complete remaining gold action ledger."""

    def __init__(self, raw: dict, frozen: Path) -> None:
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
        self.supplement_reference_reads(raw["result"]["trajectory"]["messages"])
        self.expected_state = replay_five.state_of(shadow)
        self.criteria = task["evaluation_criteria"]
        self.initial_db_sha256 = source["db_sha256"]
        self.task = task
        self._native_cache = {}

    def supplement_reference_reads(self, messages: list[dict]) -> None:
        """Retain ordered reference reads omitted from write-only task criteria.

        Only read/auth calls preceding an exact task-gold write are inserted.
        They carry explicit reference-trace provenance. Prefix execution still
        requires byte-exact native result parity; feasibility/completion still
        executes native tools and checks the task's final state/assertions.
        No SDG online verdict supplies evidence of correctness.
        """
        pending = []
        attached = {}
        anchored = set()
        known = {(g["name"], json.dumps(g["arguments"], sort_keys=True)) for g in self.gold}
        for index, message in enumerate(messages):
            if message.get("type") != "tool_call" or message.get("source") != "agent":
                continue
            arguments = message["arguments"]
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            call = {"name": message["tool_name"], "arguments": arguments}
            key = (call["name"], json.dumps(arguments, sort_keys=True))
            if self.tool_type(call) in {"read", "auth"} and key not in known:
                pending.append(
                    {
                        **call,
                        "requestor": "assistant",
                        "done": False,
                        "delta": [],
                        "ledger_source": "recorded_reference_read",
                        "reference_index": index,
                    }
                )
                known.add(key)
            if self.tool_type(call) == "write":
                match = next(
                    (
                        g
                        for g in self.gold
                        if g["requestor"] == "assistant"
                        and areal_v2.typed_equal({k: g[k] for k in ("name", "arguments")}, call)
                    ),
                    None,
                )
                if match is not None:
                    anchored.add(match["gold_index"])
                    attached.setdefault(match["gold_index"], []).extend(pending)
                    pending = []
        ordered = []
        for gold in self.gold:
            ordered.extend(attached.get(gold["gold_index"], []))
            ordered.append(
                {
                    **gold,
                    "evaluation_gold_index": gold["gold_index"],
                    "ledger_source": "task_gold",
                    "reference_write_anchor": gold["gold_index"] in anchored,
                }
            )
        for index, gold in enumerate(ordered):
            gold["gold_index"] = index
        self.gold = ordered

    @staticmethod
    def execute(env: Any, call: dict) -> Any:
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
    def failed(result: Any) -> bool:
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
        """Reuse immutable feasibility results only within one replay boundary."""
        key = json.dumps(call, sort_keys=True)
        if key not in self._native_cache:
            self._native_cache[key] = self._native_check_uncached(call)
        result = copy.deepcopy(self._native_cache[key])
        remaining = self.remaining()
        due = remaining[0] if remaining else None
        result["next_remaining_gold"] = copy.deepcopy(due)
        result["step_due"] = bool(
            due
            and due["requestor"] == call.get("requestor", "assistant")
            and areal_v2.typed_equal(
                {k: due[k] for k in ("name", "arguments")}, {k: call[k] for k in ("name", "arguments")}
            )
        )
        if self.tool_type(call) == "write":
            result["step_due"] = result["step_due"] and due.get("reference_write_anchor", False) if due else False
        result["write_due"] = result["step_due"] and self.tool_type(call) == "write"
        return result

    def _native_check_uncached(self, call: dict) -> dict:
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
        self._native_cache.clear()
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
        equivalent_auth = []
        if not self.failed(result) and self.tool_type(call) == "auth":
            for gold in self.remaining():
                if self.tool_type(gold) != "auth":
                    continue
                expected = self.execute(copy.deepcopy(self.env), gold)
                if not self.failed(expected) and expected.content == result.content:
                    gold["done"] = True
                    equivalent_auth.append(gold["gold_index"])
        return {
            "name": call["name"],
            "requestor": call.get("requestor", "assistant"),
            "gold_auth_result_equivalence": equivalent_auth,
            "result_sha256": replay_five.digest(recorded),
            "native_error": self.failed(result),
        }

    def state(self) -> dict:
        return {"environment": replay_five.state_of(self.env), "ledger": self.gold}

    def consent_required(self, call: dict) -> bool:
        return self.tool_type(call) == "write" and (self.domain != "telecom" or call["name"] == "refuel_data")

    def charge(self, call: dict, native_result: str | None) -> str | None:
        """Compute an identifying amount from native result postings only."""
        try:
            result = json.loads(native_result) if native_result else None
        except ValueError:
            result = None
        if not isinstance(result, dict):
            return None
        if call["name"] == "exchange_delivered_order_items":
            value = result.get("exchange_price_difference")
            return str(abs(value)) if value is not None else None
        if call["name"] == "send_certificate":
            return str(call["arguments"]["amount"])
        before = replay_five.state_of(self.env)["agent"]
        key = call["arguments"].get("reservation_id", call["arguments"].get("order_id"))
        collection = "reservations" if "reservation_id" in call["arguments"] else "orders"
        old = before.get(collection, {}).get(key, {})
        new_payments = result.get("payment_history", [])[len(old.get("payment_history", [])) :]
        return str(sum(abs(Decimal(str(payment["amount"]))) for payment in new_payments)) if new_payments else None

    def required_lookup(self) -> dict | None:
        """Native ledger's next outstanding assistant read/auth, never a later read."""
        remaining = self.remaining()
        if remaining and remaining[0]["requestor"] == "assistant" and self.tool_type(remaining[0]) in {"read", "auth"}:
            return {key: remaining[0][key] for key in ("name", "arguments")}
        return None

    def missing_lookup(self, output: list[dict], prefix: list[dict]) -> dict | None:
        """A recorded write attempt or explicit completion claim before a due lookup."""
        lookup = self.required_lookup()
        if not lookup:
            return None
        shape, value = areal_v2.one_turn(output)
        attempted = shape == "call" and self.tool_type(value) == "write"
        if shape == "invalid" and isinstance(output, list):
            for message in output:
                candidate_shape, call = areal_v2.one_turn([message])
                if candidate_shape == "call" and self.tool_type(call) == "write":
                    attempted = True
        claim = shape == "message" and bool(
            re.search(
                r"\bI (?:have |already )?(?:cancelled|canceled|updated|changed|booked|returned|exchanged|refunded)\b|"
                r"\b(?:order|reservation|booking|address|payment) (?:has been|is now) "
                r"(?:cancelled|canceled|updated|changed|booked|returned|exchanged|refunded)\b",
                value,
                re.I,
            )
        )
        if not (attempted or claim):
            return None
        if not areal_v2.lookup_arguments_public(prefix, lookup):
            return {"status": "hold", "reason": "required_lookup_arguments_not_public", "gold": lookup}
        native = self.native_check(lookup)
        if not native["feasible"] or not native["step_due"]:
            return {"status": "hold", "reason": "required_lookup_not_native_feasible", "native": native}
        return {
            "status": "candidate",
            "error_kind": "K6",
            "gold": lookup,
            "native": native,
            "consent": detect(prefix, lookup),
            "witness": "write_or_claim_before_next_native_lookup",
            "failing_shape": shape,
        }

    def inspect(self, output: list[dict], prefix: list[dict]) -> dict:
        """Propose supported boundaries; separate holds from proved failures.

        This is candidate generation. A proposed kind still needs pinned K4
        content controls, HTTP parity, reproduction and independent audit.
        """
        shape, candidate = areal_v2.one_turn(output)
        missing = self.missing_lookup(output, prefix)
        if missing:
            return missing
        if shape == "invalid":
            return {"status": "hold", "reason": "unsupported_turn_shape"}
        if shape == "message":
            for gold in self.remaining():
                if (
                    gold is not self.remaining()[0]
                    or gold["requestor"] != "assistant"
                    or self.tool_type(gold) != "write"
                ):
                    continue
                if not ("?" in candidate or re.search(r"(?:^|[.!]\s*)(?:please )?confirm\b", candidate, re.I)):
                    continue
                native = self.native_check(gold)
                amount = self.charge(gold, native.get("native_result"))
                consent = detect(prefix, gold, native_amount=amount)
                if (
                    consent["authorized_prior_consent"]
                    and not consent["k5_eligible"]
                    and is_reask(candidate, gold, prefix=prefix, native_amount=amount)
                ):
                    return {
                        "status": "hold",
                        "reason": "K5_precision_witness_not_certifiable",
                        "consent": consent,
                        "native": native,
                    }
                if (
                    consent["k5_eligible"]
                    and native["write_due"]
                    and is_reask(candidate, gold, prefix=prefix, native_amount=amount)
                ):
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
            lookup = self.required_lookup()
            if lookup:
                good = self.native_check(lookup)
                if good["feasible"] and good["step_due"] and areal_v2.lookup_arguments_public(prefix, lookup):
                    return {
                        "status": "candidate",
                        "error_kind": "K7",
                        "gold": lookup,
                        "native": good,
                        "consent": detect(prefix, lookup),
                        "witness": "premature_close_before_required_lookup",
                    }
            remaining = [g for g in self.remaining() if g["requestor"] == "assistant"]
            targets = []
            for gold in remaining:
                if self.tool_type(gold) != "write":
                    continue
                good = self.native_check(gold)
                given = detect(prefix, gold, native_amount=self.charge(gold, good.get("native_result")))
                if good["feasible"] and given["authorized_prior_consent"]:
                    targets.append(gold)
            if len(targets) == 1:
                return self.inspect_student(output, prefix, {key: targets[0][key] for key in ("name", "arguments")})
            return {
                "status": "hold" if remaining else "continue",
                "reason": "K7_needs_current_request_continuation_witness" if remaining else "no_gold_remaining",
            }
        if kind in {"read", "auth"}:
            lookup = self.required_lookup()
            if lookup and not areal_v2.typed_equal(candidate, lookup):
                return self.inspect_student(output, prefix, lookup)
            return {"status": "continue", "reason": "read_alternatives_not_classified_as_errors"}
        if kind != "write":
            return {"status": "hold", "reason": "unknown_tool_type"}
        native = self.native_check(candidate)
        if not native["feasible"]:
            # A surviving confirmation identifies a current requested write.
            # Only one such executable target is allowed; multiple possibilities
            # and policy-error outputs remain holds rather than guessed labels.
            targets = []
            for gold in self.remaining():
                if gold["requestor"] != "assistant" or self.tool_type(gold) != "write":
                    continue
                feasible = self.native_check(gold)
                given = detect(prefix, gold, native_amount=self.charge(gold, feasible.get("native_result")))
                if feasible["feasible"] and given["authorized_prior_consent"]:
                    targets.append(gold)
            if len(targets) == 1:
                target = {key: targets[0][key] for key in ("name", "arguments")}
                return self.inspect_student(output, prefix, target)
            # A native exception or exact gold mismatch alone is insufficient
            # for K1/K2/K3. Do not mis-certify these as missing consent.
            return {"status": "hold", "reason": "write_needs_kind_specific_error_witness", "native": native}
        consent = detect(prefix, candidate, native_amount=self.charge(candidate, native.get("native_result")))
        if not native["write_due"]:
            return {"status": "hold", "reason": "write_not_next_remaining_gold", "native": native, "consent": consent}
        if self.consent_required(candidate) and not consent["authorized_prior_consent"]:
            if not consent["k4_eligible"]:
                return {
                    "status": "hold",
                    "reason": "INBOX34_segment_assent_or_uncertainty_blocks_K4",
                    "consent": consent,
                    "native": native,
                }
            if not areal_v2.lookup_arguments_public(prefix, candidate):
                return {"status": "hold", "reason": "write_arguments_not_public", "native": native, "consent": consent}
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
        held. K3 source contexts remain deferred under INBOX33.
        """
        shape, candidate = areal_v2.one_turn(output)
        missing = self.missing_lookup(output, prefix)
        if missing:
            return missing
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
        if not good["feasible"] or not good["step_due"]:
            return {"status": "hold", "reason": "reference_not_native_feasible_or_not_due", "native": good}
        consent = detect(prefix, reference_call, native_amount=self.charge(reference_call, good.get("native_result")))
        authorized = not self.consent_required(reference_call) or consent["authorized_prior_consent"]
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
        if self.tool_type(reference_call) in {"read", "auth"} and self.tool_type(candidate) in {"read", "auth"}:
            # A non-reference or differently ordered read is ordinarily valid.
            # Admit only a same-tool wrong identifier, a visible correct ID,
            # and an actual native lookup error. No commuting-read inference.
            if candidate["name"] == reference_call["name"]:
                changed_ids = [
                    key
                    for key, value in reference_call["arguments"].items()
                    if key.endswith("_id") and not areal_v2.typed_equal(value, candidate["arguments"].get(key))
                ]
                caller_text = " ".join(str(m.get("content", "")) for m in prefix if m.get("role") == "user")
                grounded = changed_ids and all(
                    str(reference_call["arguments"][key]) in caller_text for key in changed_ids
                )
                bad = self.execute(copy.deepcopy(self.env), candidate)
                if grounded and self.failed(bad):
                    return {
                        "status": "candidate",
                        "error_kind": "K6",
                        "gold": reference_call,
                        "native": good,
                        "consent": consent,
                        "bad_native_result": bad.content,
                        "witness": "caller_identifier_and_native_lookup_error",
                        "changed_id_fields": changed_ids,
                    }
            return {"status": "hold", "reason": "alternative_read_without_error_witness"}
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
        if failing:
            return {
                "status": "hold",
                "reason": "native_write_error_K3_deferred_needs_structural_witness",
                "bad_native_result": result.content,
                "gold": reference_call,
            }
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
