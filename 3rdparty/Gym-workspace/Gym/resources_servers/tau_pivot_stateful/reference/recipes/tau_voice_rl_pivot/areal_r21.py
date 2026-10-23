# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""INBOX38 monotone post-filters; old R2 native results remain reusable."""

import copy
import re

from recipes.tau_voice_rl_pivot import areal_r2, areal_v2
from recipes.tau_voice_rl_pivot.confirmed_write import mentions


G_HANDOFF = re.compile(
    r"\b(?:transfer\w*|human\s+agents?|specialists?|representatives?|supervisors?|"
    r"escalat\w*|hold\s+on|goodbye|have\s+a\s+great\s+day|anything\s+else\b.{0,60}\btoday)\b",
    re.I,
)
G_PAYMENT_KEYS = {"payment_id", "payment_method_id"}
G_METHOD = re.compile(
    r"\b(?:credit\s+card|gift\s+card|debit\s+card|paypal|visa|mastercard|amex|"
    r"(?:original|same|existing)\s+(?:payment\s+method|card|account))\b",
    re.I,
)
G_REQUEST = r"(?:what|which|where|how|provide|specify|tell\s+me|give\s+me|need|require)"
G_FIELDS = {
    "reason": r"reason|why",
    "zip": r"zip(?:\s+code)?|postal\s+code",
    "city": r"city|town",
    "state": r"state|province",
    "country": r"country",
    "address1": r"address|street",
    "address2": r"address|apartment|suite|unit",
    "cabin": r"cabin|class",
    "flights": r"flight|date|departure|arrival",
    "passengers": r"passenger|birth|name|passport",
    "total_baggages": r"bags?|baggage",
    "gb_amount": r"gigabytes?|GB|data\s+amount",
    "order_id": r"order",
    "reservation_id": r"reservation|booking",
    "item_ids": r"items?|products?",
    "new_item_ids": r"items?|products?|size|color|colour",
}


def handoff_language(text: str) -> list[str]:
    text = re.sub(r"[\s\-–—]+", " ", text)
    return [m.group(0).casefold() for m in G_HANDOFF.finditer(text)]


def requested_fields(text: str, arguments: dict) -> list[str]:
    """Recognize requests for parameter values, not a plain 'Confirm?' recap."""
    result = []
    method = bool(
        re.search(
            r"\b(?:which|what|provide|specify|choose|select|need|require|confirm)\b.{0,55}"
            r"\b(?:payment|refund\s+method|card|account)\b|"
            r"\b(?:how|where)\b.{0,60}\b(?:refund|pay|reimburse|send|receive)\b|"
            r"\brefund\b.{0,150}\bor\b.{0,100}\?|"
            r"\bwould\s+you\s+like\b.{0,55}\b(?:back|refunded)\b.{0,30}\bto\b",
            text,
            re.I,
        )
    )
    for key in arguments:
        if key in G_PAYMENT_KEYS:
            if method:
                result.append(key)
            continue
        alias = G_FIELDS.get(key)
        if alias and re.search(r"\b" + G_REQUEST + r"\b.{0,55}\b(?:" + alias + r")\b", text, re.I):
            result.append(key)
        elif key == "reason" and re.search(r"\bwhy\b.{0,50}\?", text, re.I):
            result.append(key)
    return result


def offer_states_value(offer: str, key: str, value: object) -> bool:
    if isinstance(value, (str, int, float)) and str(value) and mentions(offer, value):
        return True
    if key in G_PAYMENT_KEYS:
        return bool(G_METHOD.search(offer) or re.search(r"\b(?:credit_card|gift_card|paypal)_\w+", offer, re.I))
    if isinstance(value, list):
        return bool(value) and all(
            offer_states_value(offer, key, leaf)
            for item in value
            for leaf in (item.values() if isinstance(item, dict) else [item])
        )
    return False


def qualify(spec: dict) -> dict:
    base = areal_r2.qualify(spec)
    if spec["error_kind"] != "K5" or base["status"] != "candidate":
        return base
    proof = spec["eligibility"]
    if "r21_failing_output" not in proof:
        return dict(status="hold", reason="K5_missing_failing_turn_witness", rule="required_information")
    shape, value = areal_v2.one_turn(proof["r21_failing_output"])
    if shape != "message":
        return base
    offer = base["witness"]["offer"]
    args = spec["gold"]["arguments"]
    missing = [k for k in requested_fields(value, args) if not offer_states_value(offer, k, args[k])]
    if missing:
        return dict(
            status="hold",
            reason="K5_required_information_not_in_offer",
            rule="required_information",
            witness=dict(fields=missing, question=value, confirmed_offer=offer),
        )
    return base


def postprocess(spec: dict, output: list[dict], prior: dict) -> dict:
    """Apply exactly the same new predicates to live or saved R2 results."""
    result = copy.deepcopy(prior)
    kind = spec["error_kind"]
    if kind == "K7":
        shape, value = areal_v2.one_turn(output)
        matches = handoff_language(value) if shape == "message" else []
        result["predicates"]["R21_no_spoken_handoff_or_closing"] = not matches
        result["r21_handoff_matches"] = matches
        result["reward"] = int(bool(prior["reward"]) and not matches)
    elif kind == "K5":
        qualification = qualify(spec)
        result["r21_qualification"] = qualification
        allowed = qualification["status"] == "candidate"
        result["predicates"]["R21_not_required_information"] = allowed
        result["reward"] = int(bool(prior["reward"]) and allowed)
    return result


def evaluate(spec: dict, output: list[dict], *, native_check=None, consent_check=None) -> dict:
    base = {**spec, "contract": "G5-INBOX37-R2"}
    prior = areal_r2.evaluate(base, output, native_check=native_check, consent_check=consent_check)
    return postprocess(spec, output, prior)
