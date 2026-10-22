# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""G5 INBOX34 consent eligibility, independent of SDG content grading.

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
    # Flight and baggage writes share a reservation ID and the broad update
    # family. An offer for one must not authorize an unoffered add-on write.
    bags = bool(re.search(r"\b(?:bags?|baggage|luggage)\b", offer, re.I))
    flights = bool(
        re.search(
            r"\b(?:flights?|cabin|economy|business|departure|outbound|upgrade)\b|\bchang(?:e|ing) to\b", offer, re.I
        )
    )
    if gold["name"] == "update_reservation_baggages" and flights and not bags:
        incompatible = True
    if gold["name"] == "update_reservation_flights" and bags and not flights:
        incompatible = True
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


def _detect_turn49(messages: list[dict], gold: dict, *, native_amount: Any = None) -> dict:
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


G_SEGMENT_ASSENT = re.compile(
    r"\b(?:yes|yeah|yep|sure|ok(?:ay)?|sounds good|that (?:works|sounds good)|"
    r"that['’]s (?:fine|right)|fine|go ahead|proceed|please do|"
    r"please (?:cancel|return|exchange|book|reserve|update|change|modify|switch|add|remove|upgrade|downgrade|refuel)|"
    r"let['’]s go with|I['’]ll take|I (?:confirm|agree|approve)|confirm|correct)\b",
    re.I,
)


def write_segment(messages: list[dict], gold: dict, *, identifying: list[str]) -> dict:
    """Conservative segment from the introducing request, after an identical commit.

    Unknown boundaries retain the whole available prefix. Mere clarifications,
    changed dates, payment answers and reason answers never erase prior assent.
    A different batch write never closes this target's segment.
    """
    floor = 0
    for index, message in enumerate(messages):
        arguments = message.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:
                arguments = None
        if (
            message.get("type") == "function_call"
            and message.get("name") == gold["name"]
            and arguments == gold.get("arguments")
        ):
            floor = index + 1
    start = floor
    reason = "unknown_introduction_keep_available_prefix"
    family = action_family(gold["name"])
    for index in range(floor, len(messages)):
        message = messages[index]
        if message.get("role") != "user":
            continue
        value = text(message)
        if family and re.search(r"\b(?:" + G_ACTIONS[family] + r")\b", value, re.I):
            related = offer_witness(value, gold, identifying=identifying)
            if not related["target_mismatch"] and not related["incompatible_action"]:
                # Include a preceding grounded agent offer when the caller's
                # first write request is itself an assent ("I'll take 2GB").
                previous = next(
                    (j for j in range(index - 1, floor - 1, -1) if messages[j].get("role") in {"user", "assistant"}),
                    None,
                )
                start = index
                if previous is not None and messages[previous].get("role") == "assistant":
                    prior = offer_witness(text(messages[previous]), gold, identifying=identifying)
                    if prior["grounded"]:
                        start = previous
                reason = "first_related_caller_write_request"
                break
    return {"start_index": start, "after_identical_write": floor, "reason": reason}


def unrelated_named_item(messages: list[dict], clause: str, identifying: list[str]) -> bool:
    """Ignore only a clearly different item name visible in native public data."""
    names = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if isinstance(value.get("name"), str) and (value.get("item_id") or value.get("product_id")):
                names.add(value["name"])
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for message in messages:
        if message.get("type") == "function_call_output":
            value = message.get("output")
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except ValueError:
                    continue
            visit(value)
    mentioned = {name for name in names if re.search(r"\b" + re.escape(name) + r"\b", clause, re.I)}
    target_names = {name for name in names if name in identifying}
    return bool(mentioned - target_names) and not bool(mentioned & target_names)


def segment_assent(messages: list[dict], gold: dict, *, native_amount: Any = None) -> dict:
    """INBOX34 asymmetric K4 veto, independent of detailed readback gates.

    Ambiguous assent is retained as a veto. Explicit negations and clearly
    different named targets are the only ignored matches. Grounding determines
    given versus hold; lack of grounding never turns an assent into K4.
    """
    identifying = [*public_details(messages, gold), *amount_details(native_amount)]
    segment = write_segment(messages, gold, identifying=identifying)
    assents = []
    ignored = []
    grounded = []
    for index in range(segment["start_index"], len(messages)):
        message = messages[index]
        value = text(message)
        if message.get("role") == "assistant" and value:
            detail = offer_witness(value, gold, identifying=identifying)
            if detail["grounded"] and not detail["target_mismatch"]:
                grounded.append({"index": index, "text": value, "grounding": detail})
        if message.get("role") != "user":
            continue
        matches = list(G_SEGMENT_ASSENT.finditer(value))
        if not matches:
            continue
        # A negated clause is ignored, not the entire reply: "not 6952. But
        # yes, proceed" still contains assent. Questions remain vetoes/holds.
        for match in matches:
            left = max(value.rfind(mark, 0, match.start()) for mark in ".!?;") + 1
            right = min([pos for mark in ".!?;" if (pos := value.find(mark, match.end())) >= 0] or [len(value)])
            clause = value[left:right]
            related = offer_witness(clause, gold, identifying=identifying)
            family = action_family(gold["name"])
            pending_verbs = G_ACTIONS.get(family, r"send|issue")
            negated = bool(
                re.search(
                    r"\b(?:don['’]?t|do not|not|never|won['’]?t|cannot|can['’]?t)\s*$",
                    value[left : match.start()],
                    re.I,
                )
            ) or bool(
                re.search(
                    r"\b(?:not yet|hold off|wait)\b|"
                    r"\b(?:don['’]?t|do not|never|won['’]?t|cannot|can['’]?t)\s+(?:want to\s+)?(?:"
                    + pending_verbs
                    + r"|proceed|go ahead|do (?:it|that))\b",
                    clause,
                    re.I,
                )
            )
            different = related["target_mismatch"] or unrelated_named_item(messages, clause, identifying)
            evidence = {
                "reply_index": index,
                "reply": value,
                "token": match.group(),
                "token_start": match.start(),
                "clause": clause,
            }
            if negated or different:
                ignored.append({**evidence, "reason": "explicit_negation" if negated else "different_named_write"})
            else:
                assents.append(
                    {
                        **evidence,
                        "caller_question": "?" in value,
                        "agent_grounding": grounded[-1] if grounded else None,
                    }
                )
    return {**segment, "assents": assents, "ignored": ignored}


G_K5_AFFIRMATIVE = re.compile(
    r"\s*(?:(?:yes|yeah|yep|yup|sure|ok(?:ay)?|absolutely|affirmative|confirmed|correct|"
    r"please|proceed|go ahead|do (?:it|that)|sounds good|that works|"
    r"that(?:['’]s| is| sounds) (?:right|good|fine|correct)|I (?:confirm|agree|approve)|"
    r"with (?:it|that|the (?:change|update|booking|cancellation|return|exchange|upgrade))|"
    r"thank you|thanks)\s*[,!.;:-]?\s*)+",
    re.I,
)


def precise_write_witness(messages: list[dict], witness: dict | None) -> bool:
    """A conservative permission certificate; broad K4 vetoes remain separate."""
    if not witness or not witness["clean_affirmative"] or not G_K5_AFFIRMATIVE.fullmatch(witness["reply"]):
        return False
    last_user = max((i for i, message in enumerate(messages) if message.get("role") == "user"), default=-1)
    if last_user != witness["reply_index"] or re.search(r"\bor\b", witness["offer"], re.I):
        return False
    family = witness["action_family"]
    if witness["matched_actions"] and family not in witness["matched_actions"]:
        return False
    noun = re.search(
        r"\bwith the (change|update|booking|cancellation|return|exchange|upgrade)\b", witness["reply"], re.I
    )
    noun_families = {
        "change": "update",
        "update": "update",
        "upgrade": "update",
        "booking": "book",
        "cancellation": "cancel",
        "return": "return",
        "exchange": "exchange",
    }
    if noun and noun_families[noun.group(1).lower()] != family:
        return False
    permission = bool(witness["matched_actions"]) or bool(
        re.search(
            r"\b(?:shall|may|can|could|should|would) (?:I|we) (?:proceed|go ahead|do (?:it|that))\b|"
            r"\b(?:proceed|go ahead)\s*\?",
            witness["offer"],
            re.I,
        )
    )
    if "?" in witness["offer"]:
        return witness["eligible_offer"] and permission
    return (
        witness["grounded"]
        and not witness["incompatible_action"]
        and not witness["target_mismatch"]
        and not witness["non_write_question"]
        and explicit_instruction(witness["reply"])
    )


def precise_k5_witness(messages: list[dict], witness: dict | None) -> bool:
    """K5 additionally excludes permission given after a declarative proposal."""
    return bool(
        witness and witness["eligible_offer"] and "?" in witness["offer"] and precise_write_witness(messages, witness)
    )


def detect(messages: list[dict], gold: dict, *, native_amount: Any = None) -> dict:
    """Keep turn49 positive witnesses; add the asymmetric INBOX34 K4 veto.

    `consent_label` is the census classification. Newly recovered given cases
    block K4 but do not automatically create new K5/write-positive witnesses:
    `authorized_prior_consent` additionally requires the turn49 witness. A
    question in its affirmative reply is a hold even with an old witness.
    """
    result = _detect_turn49(messages, gold, native_amount=native_amount)
    guard = segment_assent(messages, gold, native_amount=native_amount)
    relevant = guard["assents"]
    witness = result["witness"]
    witness_question = bool(witness and "?" in witness["reply"])
    last = relevant[-1] if relevant else None
    if witness_question or (last and last["caller_question"]):
        label = "hold"
    elif result["valid_prior_consent"]:
        label = "given"
    elif last:
        label = "given" if last["agent_grounding"] else "hold"
    elif result["uncertain_prior_consent"]:
        label = "hold"
    else:
        label = "K4_candidate"
    authorized = bool(result["valid_prior_consent"] and label == "given" and precise_write_witness(messages, witness))
    k5 = bool(authorized and precise_k5_witness(messages, witness))
    return {
        **result,
        "detector_contract": "G5-INBOX35",
        "write_segment": guard,
        "consent_label": label,
        "k4_eligible": label == "K4_candidate" and not relevant,
        "authorized_prior_consent": authorized,
        "k5_eligible": k5,
    }
