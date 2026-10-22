# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""G5 INBOX31 consent eligibility, independent of SDG content grading.

Consumes public Responses messages only. A witness identifies the offer, its
next caller response, and its action/detail grounding. This is a deterministic
linguistic detector, not a native-prefix certificate or a policy checker.
"""

import json
import re
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
    r"\b(?:do|would) you (?:like|want|agree|approve|confirm)|"
    r"\b(?:please confirm|can you confirm|confirm\s*\?|proceed\s*\?|okay\s*\?|ok\s*\?|"
    r"is (?:that|this|everything) (?:correct|right|okay|ok)|does (?:that|this) (?:sound|look)|"
    r"go ahead\s*\?|confirm\b|are you (?:okay|ok|ready)|would that work|do you (?:confirm|authorize))",
    re.I,
)
G_AFFIRM = re.compile(
    r"^\s*(?:yes|yeah|yep|yup|sure|ok(?:ay)?|correct|confirmed|absolutely|affirmative|"
    r"sounds good|that's (?:right|correct|fine)|that is (?:right|correct|fine)|"
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


def offer_witness(offer: str, gold: dict) -> dict:
    """Decide whether an offer grounds the intended write, with explicit evidence."""
    family = action_family(gold["name"])
    actions = [key for key, pattern in G_ACTIONS.items() if re.search(r"\b(?:" + pattern + r")\b", offer, re.I)]
    details = [
        value
        for value in leaves(gold.get("arguments", {}))
        if len(value) >= 2 and re.search(r"(?<!\w)" + re.escape(value).replace("_", "[ _]") + r"(?!\w)", offer, re.I)
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
        "is_offer": bool(G_OFFER.search(offer)),
        "choice": bool(G_CHOICE.search(offer)),
        "action_family": family,
        "matched_actions": actions,
        "matched_details": details,
        "incompatible_action": incompatible,
        "grounded": grounded,
        "non_write_question": non_write_question,
        "target_mismatch": target_mismatch,
        "eligible_offer": bool(G_OFFER.search(offer))
        and grounded
        and not incompatible
        and not G_CHOICE.search(offer)
        and not non_write_question
        and not target_mismatch,
    }


def is_reask(message: str, gold: dict) -> bool:
    """K5 needs an actual request, not a declarative 'To confirm' readback."""
    return offer_witness(message, gold)["eligible_offer"] and bool(
        "?" in message or re.search(r"(?:^|[.!]\s*)(?:please )?confirm\b", message, re.I)
    )


def changed_request(reply: str, offer: str = "") -> bool:
    """Ignore assent restatements; retain new dates/items/amounts as changes."""
    if not G_CHANGE.search(reply):
        return False
    # Restating the offered operation is assent. New quantities, identifiers,
    # subjects or an additional operation are changes, even after "yes".
    if re.search(r"\b(?:also|another|now|new|instead|different)\b", reply, re.I):
        return True
    salient = re.findall(r"\b(?:\w*\d\w*|bags?|dates?|cabins?|passengers?|items?|payment|address|card)\b", reply, re.I)
    if any(not re.search(r"\b" + re.escape(word) + r"\b", offer, re.I) for word in salient):
        return True
    for pattern in G_ACTIONS.values():
        if re.search(r"\b(?:" + pattern + r")\b", reply, re.I) and not re.search(
            r"\b(?:" + pattern + r")\b", offer, re.I
        ):
            return True
    return False


def detect(messages: list[dict], gold: dict) -> dict:
    """Find surviving consent in the current write segment.

    A committed identical write closes the segment; a different write in a
    confirmed batch does not consume permission for remaining batch members. A later caller request/change or
    withdrawal invalidates the witness; reads and unchanged clarifications do
    not. The next *caller* turn must affirm the immediately preceding spoken
    offer; intervening tool results do not manufacture an affirmative.
    """
    witness = None
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
            witness = pending = None
            resets.append({"index": index, "reason": "intervening_write"})
        role = message.get("role")
        if role == "assistant" and text(message):
            candidate = offer_witness(text(message), gold)
            pending = {"offer_index": index, "offer": text(message), **candidate}
        elif role == "user":
            reply = text(message)
            affirmative = (
                bool(G_AFFIRM.search(reply))
                and not G_VETO.search(reply)
                and not changed_request(reply, pending["offer"] if pending else "")
            )
            if pending is not None:
                attempt = {**pending, "reply_index": index, "reply": reply, "clean_affirmative": affirmative}
                attempts.append(attempt)
                if pending["eligible_offer"] and affirmative:
                    witness = attempt
            if G_VETO.search(reply) or changed_request(reply, pending["offer"] if pending else ""):
                if witness is not None:
                    resets.append({"index": index, "reason": "caller_change_or_withdrawal", "text": reply})
                witness = None
            pending = None
    return {
        "valid_prior_consent": witness is not None,
        "witness": witness,
        "attempts": attempts,
        "resets": resets,
        "detector_contract": "G5-INBOX31",
    }
