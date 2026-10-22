# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""G5 INBOX33 consent eligibility, independent of SDG content grading.

Consumes public Responses messages only. A witness identifies the offer, its
next caller response, and its action/detail grounding. This is a deterministic
linguistic detector, not a native-prefix certificate or a policy checker.
"""

import json
import re
from decimal import Decimal
from typing import Any


G_ACTIONS = {
    "cancel": r"cancel(?:l?ing|lation|led|ed)?|void",
    "return": r"return(?:ing|ed)?",
    "exchange": r"exchang(?:e|ing)|swap",
    "book": r"book(?:ing|ed)?|reserv(?:e|ing)",
    "update": r"updat(?:e|ing)|chang(?:e|ing)|modify|modification|switch|add|remov(?:e|ing)|upgrad(?:e|ing)|downgrad(?:e|ing)",
    "refuel": r"refuel|top[ -]?up|add",
}
G_OFFER = re.compile(
    r"\b(?:shall|may|can|could|should|would)\s+(?:I|we)\b|"
    r"\b(?:do|would) you (?:like|want|agree|approve|confirm)|\bwant (?:me to|to)|"
    r"\b(?:please confirm|can you confirm|confirm\s*\?|proceed\s*\?|okay\s*\?|ok\s*\?|"
    r"is (?:that|this|everything) (?:correct|right|okay|ok)|does (?:that|this) (?:sound|look)|"
    r"go ahead\s*\?|confirm\b|are you (?:okay|ok|ready)|would that work|do you (?:confirm|authorize))",
    re.I,
)
G_AFFIRM = re.compile(
    r"^\s*(?:yes|yeah|yep|yup|sure|ok(?:ay)?|correct|confirmed|absolutely|affirmative|"
    r"sounds good|that works|that sounds (?:good|right)|confirm|that['’]s (?:right|correct|fine)|that is (?:right|correct|fine)|"
    r"please (?:do|proceed|go ahead)|go ahead|proceed|I (?:confirm|agree|approve))\b",
    re.I,
)
G_VETO = re.compile(
    r"\b(?:no|not|never|but|don['’]?t|do not|wait|instead|hold|actually|except|rather|"
    r"only if|unless|before|however)\b",
    re.I,
)
G_CHANGE = re.compile(
    r"\b(?:change|update|modify|book|switch|replace|make it|I(?:'d| would) (?:like|prefer)|use (?:my|the|a)|"
    r"add|remove|cancel|return|exchange|upgrade)\b",
    re.I,
)
G_CHOICE = re.compile(
    r"(?:^|[.!?]\s*)(?:which|what|how much|how many|choose|select)\b[^?]*\?|\b(?:prefer|choose|select)\b[^?]*\bor\b[^?]*\?",
    re.I,
)


def text(message: dict) -> str:
    """Extract text without admitting tool output as caller/agent speech."""
    value = message.get("content", "")
    if isinstance(value, list):
        return " ".join(x.get("text", "") for x in value if isinstance(x, dict))
    return value if isinstance(value, str) else ""


def action_family(name: str) -> str | None:
    """Map native write names to confirmation action families."""
    for family in ("cancel", "return", "exchange", "refuel", "book", "update"):
        if name.startswith(family):
            return family
    if name.startswith("modify"):
        return "update"
    return None


def leaves(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [leaf for child in value.values() for leaf in leaves(child)]
    if isinstance(value, list):
        return [leaf for child in value for leaf in leaves(child)]
    return [str(value)] if isinstance(value, (str, int, float)) and not isinstance(value, bool) else []


def public_details(messages: list[dict], gold: dict) -> list[str]:
    """Read identifying descriptions from public native tool results only."""
    targets = {value for value in leaves(gold.get("arguments", {})) if value}
    details = set()

    def visit(value: Any) -> None:
        if isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, dict):
            if any(
                str(value.get(key, "")) in targets
                for key in ("id", "item_id", "order_id", "reservation_id", "line_id", "user_id", "customer_id")
            ):
                for key in (
                    "name",
                    "options",
                    "first_name",
                    "last_name",
                    "origin",
                    "destination",
                    "cabin",
                    "date",
                    "brand",
                    "last_four",
                    "passengers",
                    "flights",
                    "payment_history",
                ):
                    details.update(leaves(value.get(key)))
            for child in value.values():
                visit(child)

    for message in messages:
        if message.get("type") != "function_call_output":
            continue
        value = message.get("output")
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                continue
        visit(value)
    return sorted(details)


def amount_details(native_amount: Any) -> list[str]:
    """Render a native posting amount as identifying text, not a C3 verdict."""
    if native_amount is None:
        return []
    amount = abs(Decimal(str(native_amount)))
    values = [
        str(amount.normalize()),
        format(amount, "f"),
        format(amount, ",.2f"),
        "$" + format(amount, "f"),
        format(amount, "f") + " dollars",
    ]
    if amount == amount.to_integral_value():
        values += [str(int(amount)), format(int(amount), ",")]
    if amount == 0:
        values += ["free", "no charge", "no additional charge"]
    return values


def offer_witness(offer: str, gold: dict, *, identifying: list[str] | None = None) -> dict:
    """Decide whether an offer grounds the intended write, with explicit evidence."""
    family = action_family(gold["name"])
    actions = [key for key, pattern in G_ACTIONS.items() if re.search(r"\b(?:" + pattern + r")\b", offer, re.I)]
    details = [
        value
        for value in [*leaves(gold.get("arguments", {})), *(identifying or [])]
        if len(value) >= 2
        and re.search(r"(?<!\w)(?<!\d\.)" + re.escape(value).replace("_", "[ _]") + r"(?!\w|\.\d)", offer, re.I)
    ]
    # A different explicit operation is not authorized merely by sharing an ID.
    incompatible = family not in actions and bool(
        re.search(
            r"\b(?:cancel|return|exchange|book)\s+(?:your |the |this |both |these )?(?:order|item|reservation|booking|flight|ticket)\b|"
            r"\b(?:may I|shall I|can I|I will|I[’\']ll)\s+(?:cancel|return|exchange|book)\b",
            offer,
            re.I,
        )
    )
    non_write_question = bool(
        re.search(
            r"\b(?:may|shall|can|could|should) (?:I|we) (?:explain|list|show|search|check|ask|verify)\b[^?]*\?",
            offer,
            re.I,
        )
    )
    target_mismatch = False
    for key, noun in (("reservation_id", "reservation"), ("order_id", "order")):
        target = str(gold.get("arguments", {}).get(key, ""))
        named = re.findall(r"\b" + noun + r"\s+(#[A-Z0-9]+|[A-Z0-9]*[0-9][A-Z0-9]*)\b", offer)
        if target and named and target.lstrip("#") not in [value.lstrip("#") for value in named]:
            target_mismatch = True
    grounded = family in actions or bool(details)
    return {
        "is_offer": bool(G_OFFER.search(offer)) or ("?" in offer and grounded and not G_CHOICE.search(offer)),
        "choice": bool(G_CHOICE.search(offer)),
        "action_family": family,
        "matched_actions": actions,
        "matched_details": details,
        "incompatible_action": incompatible,
        "grounded": grounded,
        "non_write_question": non_write_question,
        "target_mismatch": target_mismatch,
        "eligible_offer": (bool(G_OFFER.search(offer)) or ("?" in offer and grounded))
        and grounded
        and not incompatible
        and not G_CHOICE.search(offer)
        and not non_write_question
        and not target_mismatch,
    }


def is_reask(message: str, gold: dict, *, prefix: list[dict] | None = None, native_amount: Any = None) -> bool:
    """K5 needs an actual request, not a declarative 'To confirm' readback."""
    return offer_witness(
        message, gold, identifying=[*public_details(prefix or [], gold), *amount_details(native_amount)]
    )["eligible_offer"] and bool("?" in message or re.search(r"(?:^|[.!]\s*)(?:please )?confirm\b", message, re.I))


def assent_clause(reply: str) -> str:
    """The affirmative clause may follow an emotional preamble; additions are separate."""
    parts = re.split(r"(?<=[.!?])\s+", reply)
    for part in parts:
        if G_AFFIRM.search(part) or re.search(r"\b(?:yes|please go ahead|please proceed)\b", part, re.I):
            return re.split(r"\band\s+(?:also|please add)|\bAlso\b", part, maxsplit=1, flags=re.I)[0]
    return parts[0] if parts else reply


def explicit_instruction(reply: str) -> bool:
    return bool(
        re.search(
            r"\b(?:please\s+)?(?:go ahead|proceed|make that change)\b|\bplease\s+(?:switch|cancel|change|update|add|modify|exchange|return)\b",
            reply,
            re.I,
        )
    )


def changed_request(reply: str, offer: str = "", gold: dict | None = None) -> bool:
    """Restated verbs are assent; a correction or new target is not."""
    clause = assent_clause(reply)
    if not G_CHANGE.search(clause):
        return False
    if re.search(r"\b(?:another|instead|different|other card|make it|change the date)\b", clause, re.I):
        return True
    # Only concrete new values contradict an offer. Generic action/card nouns
    # are restatements, and native target IDs may be repeated after 'that one'.
    allowed = offer + " " + " ".join(leaves((gold or {}).get("arguments", {})))
    normalize = lambda value: re.sub(r"\W|_", "", value).lower()
    allowed = normalize(allowed)
    for token in re.findall(r"\b[A-Za-z_]*\d[A-Za-z_\d:.-]*\b", clause):
        if normalize(token) not in allowed:
            return True
    return False


def detect(messages: list[dict], gold: dict, *, native_amount: Any = None) -> dict:
    """Find surviving consent in the current write segment.

    A committed identical write closes the segment; a different write in a
    confirmed batch does not consume permission for remaining batch members. A later caller request/change or
    withdrawal invalidates the witness; reads and unchanged clarifications do
    not. The next *caller* turn must affirm the immediately preceding spoken
    offer; intervening tool results do not manufacture an affirmative.
    """
    identifying = [*public_details(messages, gold), *amount_details(native_amount)]
    witness = None
    uncertain = None
    pending = None
    attempts = []
    resets = []
    for index, message in enumerate(messages):
        name = message.get("name", "")
        arguments = message.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:
                arguments = None
        if message.get("type") == "function_call" and name == gold["name"] and arguments == gold.get("arguments"):
            witness = pending = uncertain = None
            resets.append({"index": index, "reason": "intervening_write"})
        role = message.get("role")
        if role == "assistant" and text(message):
            candidate = offer_witness(text(message), gold, identifying=identifying)
            pending = {"offer_index": index, "offer": text(message), **candidate}
        elif role == "user":
            reply = text(message)
            clause = assent_clause(reply)
            offer = pending["offer"] if pending else ""
            changed = changed_request(reply, offer, gold)
            reason_reply = bool(re.search(r"\breason\b", offer, re.I)) and bool(
                re.fullmatch(r"(?:both (?:are )?)?no longer needed[.!]?", reply.strip(), re.I)
            )
            veto = bool(G_VETO.search(clause)) and not reason_reply
            instruction = explicit_instruction(clause)
            affirmative = (bool(G_AFFIRM.search(clause)) or instruction) and not veto and not changed
            # Details stated in a proposal followed by explicit go-ahead suffice.
            # A plain choice/payment/reason answer never supplies permission.
            contextual = bool(
                pending
                and pending["grounded"]
                and not pending["incompatible_action"]
                and not pending["target_mismatch"]
                and not pending["non_write_question"]
            )
            proposal = bool(
                re.search(
                    r"\b(?:would cost|new total|refund amount|costs? .*more|adding .*cost|before I|once I)\b",
                    offer,
                    re.I,
                )
            )
            direct = contextual and (instruction or proposal)
            # An unresolved choice invalidates earlier assent; an explicit
            # instruction after making that choice is itself a new go-ahead.
            choice_answer = bool(
                pending
                and (pending["choice"] or re.search(r"\b(?:reason for|need a reason|what.*reason)\b", offer, re.I))
                and not instruction
            )
            if pending is not None:
                attempt = {**pending, "reply_index": index, "reply": reply, "clean_affirmative": affirmative}
                attempts.append(attempt)
                if (pending["eligible_offer"] or direct) and affirmative and not choice_answer:
                    witness = attempt
                    uncertain = None
            reason_answer = bool(re.search(r"\b(?:reason for|need a reason|what.*reason)\b", offer, re.I))
            choice_resets = choice_answer and (reason_answer or not witness or not witness["eligible_offer"])
            if choice_resets and witness is not None and not veto and not changed:
                uncertain = {
                    "prior_witness": witness,
                    "reply_index": index,
                    "reason": "information_answer_after_prior_assent_requires_audit",
                }
            if veto or changed:
                uncertain = None
            if veto or changed or choice_resets:
                if witness is not None:
                    resets.append({"index": index, "reason": "caller_change_or_withdrawal", "text": reply})
                witness = None
            pending = None
    return {
        "valid_prior_consent": witness is not None,
        "witness": witness,
        "uncertain_prior_consent": uncertain,
        "attempts": attempts,
        "resets": resets,
        "detector_contract": "G5-INBOX33",
    }
