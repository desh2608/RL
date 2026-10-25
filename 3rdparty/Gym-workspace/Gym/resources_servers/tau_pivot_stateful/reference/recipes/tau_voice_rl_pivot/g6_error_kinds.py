# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""G6 INBOX50 consent/due-action guards; published R21/v2.1 scorers stay frozen."""

import copy
import json
import re

from recipes.tau_voice_rl_pivot import areal_r21, areal_v2, replay_five
from recipes.tau_voice_rl_pivot.consent_given import detect


G_CONTRACT = "G6-INBOX50-v2"
G_CLOSE = re.compile(
    r"transfer|human agent|specialist|representative|supervisor|escalat|hold on|goodbye|have a great day|anything else today",
    re.I,
)
G_SUCCESS = re.compile(
    r"\b(?:successfully|completed|processed|confirmed|booked|cancelled|canceled|updated|refunded|issued|changed|done)\b",
    re.I,
)
G_INFEASIBLE = re.compile(
    r"not enough|no (?:available )?seats|unavailable|ineligible|not eligible|cannot (?:be |modify|cancel|update)|not (?:allowed|permitted)|insufficient|sold out",
    re.I,
)


def evaluate(spec, output):
    """Score one turn against privately bound feasible/grounded control witnesses."""
    shape, value = areal_v2.one_turn(output)
    message = shape == "message" and bool(value.strip()) and not G_CLOSE.search(value)
    approved = (
        any(areal_v2.typed_equal(value, c) for c in spec.get("approved_calls", [])) if shape == "call" else False
    )
    bound = spec.get("contract") == G_CONTRACT and spec.get("prefix_state_consistent") is True
    kind = spec.get("error_kind")
    if kind == "K0":
        extra_close = shape == "message" and re.search(
            r"\bbye\b|take care|thank(?:s| you) for (?:calling|contacting)|end (?:the )?(?:call|conversation)",
            value,
            re.I,
        )
        valid = approved or (
            not spec.get("approved_calls")
            and spec.get("no_gold_call_due") is True
            and message
            and not extra_close
            and not areal_r21.handoff_language(value)
        )
    elif kind == "K9a":
        valid = approved
    elif kind == "K10":
        consent = spec.get("witness", {}).get("consent", {})
        valid = (
            approved
            and spec.get("lenient_consent_test_passed") is True
            and consent.get("consent_label") == "given"
            and consent.get("basis") != "frozen_AQ_eligibility"
        )
    elif kind == "K9b":
        valid = approved or (
            message
            and not G_SUCCESS.search(value)
            and bool(
                re.match(
                    r"^(?:Let me|I(?: will| can| need to|\'ll)) (?:first )?(?:check|review|verify|look into)\b",
                    value,
                    re.I,
                )
            )
        )
    elif kind == "K4":
        # A narrow positive family; candidate generation only admits unmistakable
        # non-consent turns, never a differently worded consent question.
        valid = shape == "message" and value == spec.get("consent_question")
    else:
        valid = False
    return dict(
        reward=int(bool(bound and valid)),
        predicates=dict(
            contract=bound,
            single_turn=shape != "invalid",
            spoken_continuation=bool(message),
            approved_call=approved,
            kind_rule=bool(valid),
        ),
    )


def call_output(call):
    return [dict(type="function_call", name=call["name"], arguments=copy.deepcopy(call["arguments"]))]


def controls(spec, positive, failed):
    probes = dict(
        positive=(positive, 1),
        failing_output=(failed, 0),
        empty=([], 0),
        reasoning_only=([dict(type="reasoning", content="thinking")], 0),
        close=([dict(type="message", role="assistant", content="Goodbye, have a great day.")], 0),
        handoff=([dict(type="message", role="assistant", content="Let me transfer you to a human agent.")], 0),
        wrong_tool=(call_output(dict(name="__WRONG_TOOL__", arguments={})), 0),
    )
    if spec.get("approved_calls"):
        wrong = copy.deepcopy(spec["approved_calls"][0])
        keys = list(wrong["arguments"])
        if keys:
            wrong["arguments"][keys[0]] = "__WRONG_IDENTIFIER_OR_VALUE__"
            probes["wrong_argument"] = (call_output(wrong), 0)
    if spec["error_kind"] == "K0":
        probes["filler_when_call_due"] = (
            [dict(type="message", role="assistant", content="I can continue helping with your request.")],
            int(not spec.get("approved_calls") and spec.get("no_gold_call_due") is True),
        )
    if spec["error_kind"] == "K9b":
        probes["success_claim"] = (
            [dict(type="message", role="assistant", content="I successfully completed the booking.")],
            0,
        )
        probes["future_success_claim"] = ([dict(type="message", role="assistant", content="It will be booked.")], 0)
    return {
        key: dict(output=turn, expected=expected, **evaluate(spec, turn)) for key, (turn, expected) in probes.items()
    }


class Proposer:
    """Native candidate qualification shared across the two pinned runtime families."""

    def __init__(self, boundary, row, family, consent_builder=None):
        self.boundary = boundary
        self.row = row
        self.family = family
        self.missed_emitted = set()
        self.consent_builder = consent_builder
        self.current_output = []
        self.native_cache = {}
        self.last_hold = None
        self.targets = boundary.gold if family == "areal" else boundary.steps
        self.performed = []
        for turn in row["turns"]:
            result = str(turn.get("native_tool_result", ""))
            if not result.lower().startswith("error"):
                self.performed.extend(c for c in turn.get("output", []) if c["type"] == "function_call")

    def kind(self, step):
        if self.family == "areal":
            return self.boundary.tool_type(step)
        if step["name"] == "log_verification":
            return "auth"
        return "read" if re.match(r"^(get|find|search|list|lookup|read|check|verify)_", step["name"]) else "write"

    def all_remaining(self):
        return self.boundary.remaining()

    def native(self, step):
        key = json.dumps(step, sort_keys=True)
        if key not in self.native_cache:
            self.native_cache[key] = (
                self.boundary.native_check(step) if self.family == "areal" else self.boundary.native_target(step)
            )
        return self.native_cache[key]

    def native_read(self, step):
        env = copy.deepcopy(self.boundary.env)
        before = replay_five.digest(replay_five.state_of(env))
        if self.family == "areal":
            result = self.boundary.execute(env, step)
            error = self.boundary.failed(result)
        else:
            from recipes.tau_voice_rl_pivot.g6_afterquery import failed
            from recipes.tau_voice_sdg.afterquery_adapter import direct_agent_call

            result = direct_agent_call(env, step["name"], step["arguments"], self.boundary.mapping)
            error = failed(result)
        return not error and before == replay_five.digest(replay_five.state_of(env))

    def given(self, step, prefix):
        native = self.native(step)
        amount = self.boundary.charge(step, native.get("native_result")) if self.family == "areal" else None
        return detect(prefix, step, native_amount=amount)

    def ready(self, step, prefix, *, write_lenient=False, allow_unconsented=False):
        if step.get("requestor", step.get("actor", "assistant")) != "assistant":
            return False
        if not areal_v2.lookup_arguments_public(prefix, step):
            return False
        if self.kind(step) != "write":
            return bool(self.native(step).get("feasible"))
        remaining = self.all_remaining()
        index = step.get("gold_index", 0)
        if any(
            (self.kind(s) in {"read", "auth"} or s.get("requestor", s.get("actor")) == "user")
            and s.get("gold_index", 0) < index
            for s in remaining
        ):
            return False
        if not self.native(step).get("feasible"):
            return False
        if self.family == "afterquery":
            from recipes.tau_voice_rl_pivot.g6_afterquery import build, scorer

            spec = scorer.spec_for(scorer.version(self.boundary.table))
            positive = call_output(step)
            row = dict(
                training_ready=True,
                labels=dict(tier="A1", hold_reason=None),
                public_prefix=prefix,
                rejected_output=self.current_output,
                expected_turn_kind="write",
                controls=dict(positive=dict(output=positive)),
                remaining_gold=remaining,
                verification=spec,
                v2_state=dict(ledger=remaining),
            )
            try:
                hold, _ = build.eligibility(row, step, remaining, self.boundary.table)
            except (KeyError, ValueError, TypeError):
                return False
            shape, actual = areal_v2.one_turn(self.current_output)
            eligible = not hold and not (shape == "call" and actual["name"] == step["name"])
            return eligible and (allow_unconsented or self.given(step, prefix)["consent_label"] == "given")
        if not self.boundary.consent_required(step):
            return True
        if allow_unconsented:
            return True
        given = self.given(step, prefix)
        return given["consent_label"] == "given" if write_lenient else given["authorized_prior_consent"]

    def missing(self, step):
        if self.row.get("termination") == "campaign_stop":
            return False  # External truncation is not evidence of a missed write.
        if not (self.row.get("native_outcome") or {}).get("db_differences"):
            return False
        if self.family == "areal" and not step.get("delta"):
            return False
        if self.family == "afterquery" and self.native_read(step):
            return False
        return not any(
            areal_v2.typed_equal({k: s[k] for k in ("name", "arguments")}, {k: step[k] for k in ("name", "arguments")})
            for s in self.performed
        )

    def propose(self, turn, output, prefix, state_consistent):
        if not state_consistent:
            return None
        self.current_output = output
        self.last_hold = None
        self.native_cache.clear()
        shape, value = areal_v2.one_turn(output)
        remaining = [s for s in self.all_remaining() if s.get("requestor", s.get("actor", "assistant")) == "assistant"]
        result = str(turn.get("native_tool_result", ""))
        needs_calls = turn.get("protocol_category") == "empty_visible_content" or (
            shape == "call" and result.lower().startswith("error") and "not found" in result.lower()
        )
        approved = [
            {k: s[k] for k in ("name", "arguments")}
            for s in remaining
            if needs_calls
            and self.ready(s, prefix, write_lenient=True)
            and (self.kind(s) != "write" or self.given(s, prefix)["consent_label"] == "given")
        ]
        kind = None
        positive = None
        witness = {}
        if turn.get("protocol_category") == "empty_visible_content":
            kind = "K0"
            positive = (
                call_output(approved[0])
                if approved
                else [dict(type="message", role="assistant", content="I can continue helping with your request.")]
            )
        elif shape == "call" and result.lower().startswith("error"):
            if re.search(
                r"not found|does not exist|unknown (?:user|customer|reservation|order|folio|account)", result, re.I
            ):
                same = [s for s in approved if s["name"] == value["name"] and not areal_v2.typed_equal(s, value)]
                idkeys = [k for k in value["arguments"] if re.search(r"(?:id|number|reference|code)$", k)]
                same = [s for s in same if any(s["arguments"].get(k) != value["arguments"].get(k) for k in idkeys)]
                if same:
                    kind = "K9a"
                    approved = same
                    positive = call_output(same[0])
                    witness = dict(native_error=result, identifier_keys=idkeys)
            elif G_INFEASIBLE.search(result) and self.kind(value) == "write":
                reads = []
                write_indices = [step.get("gold_index", 0) for step in self.targets if step["name"] == value["name"]]
                write_index = min(write_indices) if write_indices else -1
                for step in self.targets:
                    if (
                        self.kind(step) in {"read", "auth"}
                        and step.get("gold_index", 0) < write_index
                        and re.search(
                            r"flight|seat|availability|eligible|eligibility|order_details|product_details",
                            step["name"],
                            re.I,
                        )
                        and areal_v2.lookup_arguments_public(prefix, step)
                        and self.native_read(step)
                    ):
                        reads.append({k: step[k] for k in ("name", "arguments")})
                reads = narrow_availability_reads(value, result, reads)
                kind = "K9b"
                approved = reads
                positive = (
                    call_output(reads[-1])
                    if reads
                    else [
                        dict(
                            type="message",
                            role="assistant",
                            content="Let me check availability and eligibility before making a change.",
                        )
                    ]
                )
                witness = dict(native_error=result)
        if not kind:
            # INBOX36 order-insensitive gold remains authoritative. Doing another
            # remaining gold action is not an omission error on this turn.
            if shape == "call":
                matched = (
                    self.boundary.matches(value["name"], value["arguments"], "assistant")
                    if self.family == "afterquery"
                    else [
                        step
                        for step in remaining
                        if areal_v2.typed_equal(value, {k: step[k] for k in ("name", "arguments")})
                    ]
                )
                if matched:
                    return None
            for step in remaining:
                key = json.dumps({k: step[k] for k in ("name", "arguments")}, sort_keys=True)
                if key in self.missed_emitted or self.kind(step) != "write" or not self.missing(step):
                    continue
                if not self.ready(step, prefix, write_lenient=True, allow_unconsented=True):
                    continue
                desired = {k: step[k] for k in ("name", "arguments")}
                if shape == "call" and areal_v2.typed_equal(value, desired):
                    continue
                # A question can still be necessary (e.g. changed request); do not
                # infer omission from a single ordinary question without a witness.
                if shape == "message" and "?" in value:
                    continue
                consent = self.given(step, prefix)
                if (
                    self.family == "areal"
                    and self.boundary.consent_required(step)
                    and consent["consent_label"] == "given"
                ):
                    binding = lenient_write_binding(prefix, step, consent)
                    if not binding["ok"]:
                        self.last_hold = dict(reason="K10_upstream_offer_divergence", gold=step, binding=binding)
                        continue
                    consent = {**consent, "write_binding": binding}
                if consent["consent_label"] != "given":
                    clear_omission = shape == "message" and bool(G_CLOSE.search(value))
                    if not clear_omission or not consent["k4_eligible"] or not self.consent_builder:
                        self.last_hold = dict(reason="K10_lenient_consent_not_given", gold=step, consent=consent)
                        continue
                    positive = self.consent_builder(step, output, prefix, consent)
                    if not positive:
                        continue
                    kind = "K4"
                    approved = []
                else:
                    kind = "K10"
                    approved = [desired]
                    positive = call_output(desired)
                witness = dict(
                    final_db_differences=self.row["native_outcome"]["db_differences"],
                    earliest_observed_ready_boundary=turn["turn_index"],
                    missing_gold=step,
                    consent=consent,
                )
                self.missed_emitted.add(key)
                break
        if not kind:
            return None
        spec = dict(
            contract=G_CONTRACT,
            error_kind=kind,
            prefix_state_consistent=True,
            approved_calls=approved,
            witness=witness,
            state_hash_at_boundary=turn["state_hash"],
        )
        if kind == "K0":
            spec["no_gold_call_due"] = not approved
        if kind == "K10":
            spec["lenient_consent_test_passed"] = True
        if kind == "K4":
            spec["consent_question"] = positive[0]["content"]
        scores = controls(spec, positive, output)
        return dict(
            category=kind,
            verification=spec,
            positive=positive,
            control_results=scores,
            controls_matched=all(s["reward"] == s["expected"] for s in scores.values()),
            failed_output=output,
            public_prefix=prefix,
        )


def recertify_cached(candidate):
    """Re-check bound G6 candidates without inventing missing native witnesses.

    Legacy K0 records with no approved call did not store an exhaustive no-due
    proof. They are held until such a proof exists; an empty list alone is not
    upgraded into that proof. This routine does not replay or change native state.
    """
    result = copy.deepcopy(candidate)
    kind = result["category"]
    spec = result["verification"]
    if kind not in {"K0", "K9a", "K9b", "K10"}:
        return result, None
    if not result.get("controls_matched") or not spec.get("prefix_state_consistent"):
        return None, dict(reason="cached_source_not_qualified")
    spec["contract"] = G_CONTRACT
    prefix = result["public_prefix"]
    if kind == "K10":
        gold = spec["witness"]["missing_gold"]
        consent = detect(prefix, gold)
        if consent["consent_label"] != "given":
            return None, dict(
                reason="K10_lenient_consent_not_given",
                consent=consent,
                gold=gold,
                disposition="K4_if_qualified_consent_positive_else_hold",
            )
        if result["domain"] in {"airline", "retail", "telecom"}:
            binding = lenient_write_binding(prefix, gold, consent)
            if not binding["ok"]:
                return None, dict(reason="K10_confirmed_write_binding_failed", consent=consent, binding=binding)
            consent["write_binding"] = binding
        spec["witness"]["consent"] = consent
        spec["lenient_consent_test_passed"] = True
    elif kind == "K0":
        prior = spec.get("approved_calls", [])
        approved = []
        consent_checks = []
        for call in prior:
            # Preserve native-qualified read/auth calls. Writes additionally need
            # the newly required public lenient-consent witness.
            is_read = re.match(r"^(get|find|search|list|lookup|read|check|verify)_", call["name"])
            is_auth = call["name"] == "log_verification"
            if is_read or is_auth:
                approved.append(call)
            else:
                consent = detect(prefix, call)
                consent_checks.append(dict(call=call, consent=consent))
                if consent["consent_label"] == "given":
                    approved.append(call)
        spec["approved_calls"] = approved
        spec["cached_write_consent_checks"] = consent_checks
        if approved:
            result["positive"] = call_output(approved[0])
            spec["no_gold_call_due"] = False
        elif spec.get("no_gold_call_due") is not True:
            return None, dict(
                reason="K0_cached_no_due_proof_missing", prior_approved_calls=prior, consent_checks=consent_checks
            )
    positive = result.get("positive") or result["control_results"]["positive"]["output"]
    result["control_results"] = controls(spec, positive, result["failed_output"])
    result["controls_matched"] = all(v["reward"] == v["expected"] for v in result["control_results"].values())
    if not result["controls_matched"]:
        return None, dict(reason="updated_control_mismatch", control_results=result["control_results"])
    result["recertification_contract"] = G_CONTRACT
    return result, None


def lenient_write_binding(prefix, gold, consent):
    """Lenient assent still binds the offered business action, target and values."""
    from recipes.tau_voice_rl_pivot.confirmed_write import (
        conflicts,
        discussion_target,
        mentions,
        offered_tools,
        public_objects,
    )

    if consent.get("consent_label") != "given":
        return dict(ok=False, reason="no_given_witness")
    witness = consent.get("witness")
    if witness:
        offer, offer_index, reply_index = witness["offer"], witness["offer_index"], witness["reply_index"]
    else:
        assents = consent.get("write_segment", {}).get("assents", [])
        assent = assents[-1] if assents else None
        grounded = assent.get("agent_grounding") if assent else None
        if not grounded:
            return dict(ok=False, reason="no_bound_offer_for_lenient_assent")
        offer, offer_index, reply_index = grounded["text"], grounded["index"], assent["reply_index"]
    tools = offered_tools(offer, prefix[:offer_index])
    if not tools:
        for m in reversed(prefix[:offer_index]):
            if m.get("type") == "message" and m.get("role") == "assistant":
                previous = str(m.get("content", ""))
                tools = offered_tools(previous, prefix)
                if tools:
                    offer = previous + " " + offer
                    break
    if gold["name"] not in tools:
        return dict(ok=False, reason="confirmed_tool_mismatch", tools=sorted(tools))
    target = discussion_target(prefix[: reply_index + 1], gold)
    if not target["ok"]:
        return dict(ok=False, reason="confirmed_target_mismatch", target=target)
    reply = str(prefix[reply_index].get("content", ""))
    bad = conflicts(prefix, offer + " " + reply, gold)
    if bad:
        return dict(ok=False, reason="confirmed_argument_conflict", conflicts=bad)
    # A subset return/exchange cannot silently become an all-items write.
    if "item_ids" in gold["arguments"]:
        objects = public_objects(prefix)
        missing = []
        for ident in gold["arguments"]["item_ids"]:
            names = {str(obj.get("name", "")) for obj in objects if str(obj.get("item_id")) == str(ident)}
            if not mentions(offer + " " + reply, ident) and not any(
                name and mentions(offer + " " + reply, name) for name in names
            ):
                missing.append(ident)
        if missing:
            return dict(ok=False, reason="confirmed_item_set_unproven", unconfirmed_item_ids=missing)
    return dict(
        ok=True,
        reason="lenient_assent_with_tool_target_values_bound",
        offer_index=offer_index,
        reply_index=reply_index,
    )


def narrow_availability_reads(failed_call, native_error, reads):
    """An unavailable itinerary leg cannot be fixed by looking up another leg."""
    arguments = failed_call.get("arguments", {})
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    match = re.search(r"\bflight\s+([A-Za-z0-9]+)", native_error, re.I)
    flights = arguments.get("flights") or []
    leg = next((i for i, f in enumerate(flights) if match and str(f.get("flight_number")) == match.group(1)), None)
    if leg is None:
        return reads
    date = flights[leg].get("date")
    origin = arguments.get("origin")
    destination = arguments.get("destination")
    if arguments.get("flight_type") == "round_trip" and leg == len(flights) - 1 and leg > 0:
        origin, destination = destination, origin
    selected = []
    for read in reads:
        values = read["arguments"]
        if date and values.get("date") and values["date"] != date:
            continue
        if origin and values.get("origin") and values["origin"] != origin:
            continue
        if destination and values.get("destination") and values["destination"] != destination:
            continue
        selected.append(read)
    return selected
