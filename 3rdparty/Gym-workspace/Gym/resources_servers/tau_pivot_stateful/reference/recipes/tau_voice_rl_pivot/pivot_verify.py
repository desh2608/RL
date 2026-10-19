# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reference binary predicates over an isolated, replayed boundary state.

Native callbacks are supplied by the frozen-archive replay harness. This module
never instantiates a model, invokes a service, or uses a model judgment.
"""

import re
from decimal import Decimal


def text_of(output):
    messages = [x for x in output if x.get("type", "message") == "message"]
    texts = []
    for item in messages:
        content = item.get("content", "")
        if isinstance(content, str):
            texts.append(content)
        else:
            texts.extend(x.get("text", "") for x in content if isinstance(x, dict))
    return " ".join(texts)


def cosmetic_control_text(text):
    """Control-only wording fix; no checker or policy-visible field changes."""
    return re.sub(r"Use your (gift card|travel certificate|certificate|paypal)\s+ending\s*\.", r"Use your \1.", text)


def evaluate(spec, output, *, native_check=None, consent_check=None):
    """Evaluate the declared predicates; native preview operates on a copy."""
    if spec["expected_turn_kind"] == "completion_message":
        calls = [x for x in output if x.get("type") == "function_call"]
        message = (
            bool(output)
            and all(
                x.get("type", "message") == "message" and x.get("role", "assistant") == "assistant" for x in output
            )
            and bool(text_of(output).strip())
        )
        decision = {}
        checks = {"message_only": bool(message), "not_write": not calls}
        if set(checks) != {p["id"] for p in spec["predicates"]}:
            raise ValueError("Completion spec/evaluator mismatch")
        return {"reward": int(all(checks.values())), "predicates": checks, "native": decision}
    output = [x for x in output if x.get("type") != "reasoning"]
    calls = [x for x in output if x.get("type") == "function_call"]
    text = text_of(output)
    kind = spec["expected_turn_kind"]
    results = {}
    allowed = all(x.get("type", "message") in {"message", "function_call"} for x in output)
    assistant_role = all(x.get("role", "assistant") == "assistant" for x in output)
    shape = len(output) == 1 and len(calls) == 1 if kind == "write" else not calls and bool(text.strip())
    shape = shape and allowed and assistant_role
    results["turn_shape"] = bool(shape)
    results["not_end"] = all(
        x.get("name") not in {"end_call", "transfer_to_human_agents", "transfer_to_specialist"} for x in calls
    )
    native = None
    if calls and native_check is not None:
        try:
            native = native_check(calls[0])
        except (ValueError, KeyError, TypeError) as exc:
            native = {"accepted": False, "reason": "invalid_candidate", "error": type(exc).__name__}
    if kind == "write":
        # No unconditional read acceptance: explicit remaining-gold identity is mandatory.
        results["remaining_gold_match"] = bool(native and native.get("accepted") and native.get("matched_gold"))
    elif kind == "consent_message":
        checks = consent_check(text) if consent_check is not None and shape else {}
        for predicate in spec["predicates"]:
            if predicate["id"].startswith("policy_"):
                results[predicate["id"]] = bool(checks.get(predicate["id"]))
    elif kind == "elicit_message":
        tokens = set(re.findall(r"[a-z0-9]+", text.casefold()))
        for predicate in spec["predicates"]:
            if predicate["id"].startswith("elicit_"):
                results[predicate["id"]] = bool(tokens & set(predicate["tokens"]))
    else:
        raise ValueError("Unsupported expected_turn_kind: " + str(kind))
    declared = {p["id"] for p in spec["predicates"]}
    if set(results) != declared:
        raise ValueError("Spec/evaluator predicate mismatch: " + str(set(results) ^ declared))
    return {"reward": int(all(results.values())), "predicates": results, "native": native}


def consent_predicates(policy, context, proposed, unmet, *, telecom=False, telecom_policy=None):
    """Build content checks from the frozen per-domain policy functions.

    No synthetic caller affirmation is accepted as evidence of real authorization:
    this checks the contents of a proposed consent question only.
    """
    name, args = proposed["name"], proposed["arguments"]
    before, result, messages = context["before"], context["result"], context["messages"]

    def check(text):
        verdict = {}
        prior = [m["content"] for m in messages if m.get("role") == "assistant" and m.get("content")]
        for gate in unmet:
            passed = False
            if gate == "C1":
                if telecom:
                    # Frozen telecom C1 is the caller's requested GB amount; a
                    # missing request requires elicitation, not consent repair.
                    passed = False
                else:
                    passed = not policy.detail_checks(name, args, text, before, prior)
                    passed = passed and bool(
                        re.search(r"[?]|\b(confirm|shall|proceed|would you|do you)\b", text, re.I)
                    )
            elif gate == "C2":
                method, methods = policy.payment_for(before, args)
                original = policy.original_payments(before, args)
                required = {p["payment_id"] for p in args.get("payment_methods", []) if p.get("payment_id")} or {
                    args.get("payment_method_id", args.get("payment_id"))
                }
                found, ambiguous = policy.payment_descriptions(text, methods, original)
                passed = found == required and not ambiguous
            elif gate == "C3":
                amount = (
                    telecom_policy.refuel_charge(before, args)
                    if telecom
                    else policy.charged_amount(name, args, before, result)
                )
                passed = amount is not None and Decimal(str(amount)).quantize(Decimal(".01")) in [
                    v.quantize(Decimal(".01")) for v in policy.money_values(text)
                ]
                if amount == 0 and re.search(r"\b(no (extra |additional )?charge|free)\b", text, re.I):
                    passed = True
            elif gate == "C4":
                passed = bool(policy.REMINDER.search(text))
            verdict["policy_" + gate] = bool(passed)
        return verdict

    return check


def spec_for(kind, verifier_version, *, unmet=(), elicit=None):
    if kind == "completion_message":
        return {
            "kind": "stateful_native_step",
            "verifier_version": verifier_version,
            "expected_turn_kind": kind,
            "reward": "1 iff all predicates pass, else 0",
            "predicates": [
                {
                    "id": "message_only",
                    "rule": "assistant message only; end_call scores zero in G1",
                    "code_path": "pivot_verify.evaluate:completion_message",
                },
                {
                    "id": "not_write",
                    "rule": "no write or transfer",
                    "code_path": "pivot_verify.evaluate:completion_message",
                },
            ],
        }
    predicates = [
        {"id": "turn_shape", "rule": "single_tool_call" if kind == "write" else "message_only"},
        {"id": "not_end", "rule": "no end/transfer while required steps remain"},
    ]
    if kind == "write":
        predicates.append(
            {
                "id": "remaining_gold_match",
                "rule": "frozen native checker accepts AND matches a remaining gold action; reads not unconditionally accepted",
                "code_path": "replay_five.native_check",
            }
        )
    if kind == "consent_message":
        predicates.extend(
            {
                "id": "policy_" + gate,
                "rule": "frozen domain-specific content predicate",
                "code_path": "pivot_verify.consent_predicates:" + gate,
            }
            for gate in unmet
        )
    if kind == "elicit_message":
        predicates.extend(
            {
                "id": "elicit_" + key,
                "rule": "schema-derived missing argument token overlap",
                "tokens": sorted(tokens),
                "code_path": "pivot_verify.evaluate:elicit_message",
            }
            for key, tokens in (elicit or {}).items()
        )
    return {
        "kind": "stateful_native_step",
        "verifier_version": verifier_version,
        "expected_turn_kind": kind,
        "predicates": predicates,
        "reward": "1 iff all predicates pass, else 0",
    }
