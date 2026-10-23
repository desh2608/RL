# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Conservative, inspectable INBOX37 binding of a write to its confirmed offer.

Structured public tool data supplies entity identities. Explicit conflicting
speech values veto a write; an omitted value still needs public grounding.
Unresolved intent is a hold, never an inferred gold permission.
"""

import json
import re
from collections.abc import Iterator

from recipes.tau_voice_rl_pivot import areal_v2
from recipes.tau_voice_rl_pivot.consent_given import detect


def norm(value: object) -> str:
    return re.sub(r"[\s_\-–—]+", " ", str(value)).strip().casefold()


def mentions(text: str, value: object) -> bool:
    return bool(re.search(r"(?<!\w)" + re.escape(norm(value)) + r"(?!\w)", norm(text))) if str(value) else False


def walk(value: object) -> Iterator[dict]:
    if isinstance(value, dict):
        yield value
        for key, child in value.items():
            if isinstance(child, dict) and re.fullmatch(r"\d{7,12}", str(key)):
                yield {"item_id": key, **child}
            if isinstance(child, dict) and str(key).startswith(("credit_card_", "gift_card_", "paypal_")):
                yield {"payment_id": key, **child}
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)


def public_objects(prefix: list[dict]) -> list[dict]:
    objects = []
    for m in prefix:
        value = m.get("arguments") if m.get("type") == "function_call" else m.get("output")
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                continue
        objects.extend(walk(value))
    return objects


def known_ids(prefix: list[dict], key: str) -> set[str]:
    values = {str(x[key]) for x in public_objects(prefix) if x.get(key) is not None}
    return values


def target_ids(text: str, key: str, known: set[str]) -> set[str]:
    found = {x for x in known if mentions(text, x)}
    if key == "order_id":
        found.update(re.findall(r"#[A-Za-z0-9]+", text))
    if key == "reservation_id":
        for token in re.findall(
            r"\b(?:reservation|booking)(?:\s+(?:id|number))?\s*[:#]?\s*([A-Za-z0-9]{5,10})\b", text, re.I
        ):
            if any(c.isdigit() for c in token) or token.isupper():
                found.add(token)
    return found


def discussion_target(prefix: list[dict], gold: dict) -> dict:
    """Latest caller answer/agent question takes precedence over older targets."""
    key = next(
        (k for k in ("order_id", "reservation_id", "line_id", "customer_id", "user_id") if k in gold["arguments"]),
        None,
    )
    if gold["name"] in {"modify_user_address", "modify_pending_order_address"}:
        for m in reversed(prefix):
            if m.get("type") != "message" or m.get("role") != "assistant":
                continue
            text = str(m.get("content", ""))
            profile = bool(re.search(r"\b(?:default|profile)(?: shipping)? address\b", text, re.I))
            order = bool(re.search(r"\b(?:order|shipping)\b", text, re.I) and "address" in text.lower())
            if profile or order:
                if gold["name"] == "modify_user_address":
                    return {"ok": profile, "scope": "profile" if profile else "order_address"}
                if profile and not order:
                    return {"ok": False, "scope": "profile_not_order_address"}
                break
            methods = offered_tools(text, prefix)
            if methods and not methods.intersection({"modify_user_address", "modify_pending_order_address"}):
                return {"ok": False, "scope": "different_write_under_discussion", "tools": sorted(methods)}
    if key is None:
        return {"ok": True, "scope": "no_order_or_reservation_target"}
    public_ids = known_ids(prefix, key)
    known = public_ids | {str(gold["arguments"][key])}
    for index in range(len(prefix) - 1, -1, -1):
        m = prefix[index]
        if m.get("type") == "message":
            ids = target_ids(str(m.get("content", "")), key, known)
        elif m.get("type") == "function_call":
            args = m.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {}
            ids = {str(args[key])} if key in args else set()
        else:
            continue
        if ids:
            return {"ok": str(gold["arguments"][key]) in ids, "key": key, "targets": sorted(ids), "index": index}
    if key in {"line_id", "customer_id", "user_id"} and public_ids == {str(gold["arguments"][key])}:
        return {"ok": True, "key": key, "targets": sorted(public_ids), "scope": "unique_public_identity"}
    return {"ok": False, "key": key, "targets": [], "reason": "discussion_target_unresolved"}


def offered_tools(text: str, prefix: list[dict]) -> set[str]:
    t = norm(text)
    tools = set()
    if re.search(r"\b(cancel|cancelling|canceling)\b", t):
        tools.add(
            "cancel_reservation" if re.search(r"\b(flight|reservation|booking)\b", t) else "cancel_pending_order"
        )
    if re.search(r"\b(return|returning)\b", t) and not re.search(r"\breturn flight\b", t):
        tools.add("return_delivered_order_items")
    if "address" in t or "shipping" in t:
        if re.search(r"\b(profile|default)\b", t):
            tools.add("modify_user_address")
        if re.search(r"\b(order|shipping)\b", t):
            tools.add("modify_pending_order_address")
    if re.search(
        r"\b(?:change|switch|update|replace)\b.{0,55}\bpayment(?: method)?\b|\bpayment method\b.{0,35}\b(?:to|instead)\b",
        t,
    ):
        tools.add("modify_pending_order_payment")
    item_change = bool(re.search(r"\b(?:change|modify|replace|swap|exchange)\b", t)) and bool(
        re.search(r"\b(items?|size|color|colour|style|crew neck|v neck)\b", t)
    )
    if item_change:
        ids = target_ids(text, "order_id", known_ids(prefix, "order_id"))
        orders = [
            x
            for x in public_objects(prefix)
            if x.get("order_id") and x.get("status") and (not ids or str(x["order_id"]) in ids)
        ]
        delivered = bool(orders and orders[-1]["status"] == "delivered")
        tools.add("exchange_delivered_order_items" if "exchange" in t and delivered else "modify_pending_order_items")
    if re.search(r"\b(?:add|remove|change|update|set|increase|decrease)\b.{0,50}\b(?:bags?|baggage)\b", t):
        tools.add("update_reservation_baggages")
    if re.search(r"\b(passenger|birth|passport)\b", t) and re.search(r"\b(change|update|correct|modify)\b", t):
        tools.add("update_reservation_passengers")
    if re.search(r"\b(flights?|cabin|economy|business|seats?)\b", t) and not tools.intersection(
        {"cancel_reservation", "update_reservation_baggages", "update_reservation_passengers"}
    ):
        tools.add(
            "book_reservation"
            if re.search(r"\b(?:book|reserve|booking)\b", t)
            and not re.search(r"\b(?:change|update|switch|downgrade|upgrade)\b", t)
            else "update_reservation_flights"
        )
    if re.search(r"\b(?:certificate|compensation)\b", t) and re.search(r"\b(?:send|issue|provide|offer)\b", t):
        tools.add("send_certificate")
    if re.search(r"\b(?:add|refuel|buy|purchase)\b.{0,35}\b(?:gb|data)\b", t):
        tools.add("refuel_data")
    if re.search(r"\b(?:enable|activate|turn on)\b.{0,25}\broaming\b", t):
        tools.add("enable_roaming")
    return tools


def active_text(offer: str) -> str:
    # Old values in an explicit from-X-to-Y phrase are not selected values.
    return re.sub(r"\bfrom\s+.{1,140}?\s+to\s+", " to ", offer, flags=re.I)


def conflicts(prefix: list[dict], offer: str, gold: dict) -> list[dict]:
    args = gold["arguments"]
    objects = public_objects(prefix)
    active = active_text(offer)
    bad = []

    def reject(field, offered, expected):
        bad.append({"field": field, "offered": offered, "gold": expected})

    if "flights" in args:
        expected = {str(x["flight_number"]) for x in args["flights"]}
        universe = {str(x["flight_number"]) for x in objects if x.get("flight_number")}
        stated = {x for x in universe if mentions(active, x)}
        for token in re.findall(r"\bflight\s+([A-Za-z0-9]{3,10})\b", active, re.I):
            if any(c.isdigit() for c in token):
                stated.add(token)
        if stated - expected:
            reject("flights.flight_number", sorted(stated), sorted(expected))
        if re.search(r"\b(?:same|current|original) flights?\b", active, re.I):
            record = next(
                (
                    x
                    for x in reversed(objects)
                    if x.get("reservation_id") == args.get("reservation_id") and isinstance(x.get("flights"), list)
                ),
                None,
            )
            if record is None:
                reject("flights", "same flights unresolved", args["flights"])
            elif {x["flight_number"] for x in record["flights"]} != expected:
                reject("flights.flight_number", [x["flight_number"] for x in record["flights"]], sorted(expected))
        dates = {(int(x["date"][5:7]), int(x["date"][8:10])) for x in args["flights"] if x.get("date")}
        months = {
            name: i
            for i, name in enumerate(
                [
                    "january",
                    "february",
                    "march",
                    "april",
                    "may",
                    "june",
                    "july",
                    "august",
                    "september",
                    "october",
                    "november",
                    "december",
                ],
                1,
            )
        }
        for month, day in re.findall(r"\b(" + "|".join(months) + r")\s+(\d{1,2})\b", active, re.I):
            if (months[month.lower()], int(day)) not in dates:
                reject("flights.date", month + " " + day, sorted(dates))
        for date in re.findall(r"\b\d{4}-\d{2}-\d{2}\b", active):
            if (int(date[5:7]), int(date[8:10])) not in dates:
                reject("flights.date", date, sorted(dates))
    if "cabin" in args:
        cabins = re.findall(r"\bbasic[ _-]economy\b|\beconomy\b|\bbusiness\b", active, re.I)
        if any(norm(x) != norm(args["cabin"]) for x in cabins):
            reject("cabin", cabins, args["cabin"])
    if any(k in args for k in ["address1", "city", "state", "zip"]):
        street = re.findall(
            r"\b\d{1,6}\s+[A-Za-z0-9][^,;?\n]{1,65}?\s(?:Lane|Street|Road|Avenue|Drive|Court|Boulevard|Way|Ln|St|Rd|Ave|Dr|Ct|Blvd)\b",
            active,
            re.I,
        )
        if street and "address1" in args and any(norm(x) != norm(args["address1"]) for x in street):
            reject("address1", street, args["address1"])
        for state, zip_code in re.findall(r"\b([A-Z]{2})\s+(\d{5}(?:-\d{4})?)\b", active):
            if "state" in args and norm(state) != norm(args["state"]):
                reject("state", state, args["state"])
            if "zip" in args and str(zip_code) != str(args["zip"]):
                reject("zip", zip_code, args["zip"])
        if re.search(r"\b(?:no|without) (?:apartment|apt|suite|unit)\b", active, re.I) and args.get("address2"):
            reject("address2", "explicit none", args["address2"])
    if "new_item_ids" in args:
        ids = {str(x) for x in args["new_item_ids"]}
        entities = [x for x in objects if str(x.get("item_id")) in ids and isinstance(x.get("options"), dict)]
        if not entities:
            reject("new_item_ids", "public item attributes unresolved", sorted(ids))
        else:
            selected = re.split(r"\b(?:refund|price difference|charged|total)\b", active, flags=re.I)[0]
            split = re.search(
                r"\b(?:change|modify|swap|replace|exchange)\b.{0,150}?\b(?:to|with|for)\b(.*)", selected, re.I
            )
            selected = split.group(1) if split else selected
            explicit = {x for x in known_ids(prefix, "item_id") if mentions(selected, x)}
            if explicit - ids:
                reject("new_item_ids", sorted(explicit), sorted(ids))
            options = {}
            for entity in objects:
                for key, value in (entity.get("options") or {}).items():
                    if isinstance(value, (str, int, float)):
                        options.setdefault(key, set()).add(str(value))
            for key, values in options.items():
                stated = {value for value in values if mentions(selected, value)}
                expected = {str(x["options"][key]) for x in entities if key in x["options"]}
                if stated and expected and {norm(x) for x in stated} - {norm(x) for x in expected}:
                    reject("new_item_ids.options." + key, sorted(stated), sorted(expected))
    if "passengers" in args:
        names = {" ".join(str(x.get(k, "")) for k in ["first_name", "last_name"]).strip() for x in args["passengers"]}
        for name in re.findall(r"\b(?:name|passenger)\s+to\s+([A-Z][a-z]+(?: [A-Z][a-z]+){1,3})", active):
            if norm(name) not in {norm(x) for x in names}:
                reject("passengers.name", name, sorted(names))
        dobs = {str(x.get("dob")) for x in args["passengers"] if x.get("dob")}
        for date in re.findall(r"\b\d{4}-\d{2}-\d{2}\b", active):
            if dobs and date not in dobs:
                reject("passengers.dob", date, sorted(dobs))
    if "reason" in args:
        reasons = [x for x in ["no longer needed", "ordered by mistake"] if mentions(active, x)]
        if reasons and norm(args["reason"]) not in {norm(x) for x in reasons}:
            reject("reason", reasons, args["reason"])
    payment_key = next((k for k in ["payment_id", "payment_method_id"] if k in args), None)
    if payment_key:
        expected = str(args[payment_key])
        ids = set(re.findall(r"\b(?:credit_card|gift_card|paypal)_[A-Za-z0-9]+\b", active, re.I))
        if ids and expected not in ids:
            reject(payment_key, sorted(ids), expected)
        last4 = re.findall(r"\bending(?: in)?\s*(\d{4})\b", active, re.I)
        for digits in last4:
            matches = {
                str(x["payment_id"])
                for x in objects
                if x.get("payment_id")
                and any(str(x.get(k, "")) == digits for k in ["last_four", "last4", "last_four_digits"])
            }
            if matches and expected not in matches:
                reject(payment_key, sorted(matches), expected)
        if re.search(r"\bgift card\b", active, re.I) and not expected.startswith("gift_card"):
            reject(payment_key, "gift card", expected)
        if re.search(r"\bpaypal\b", active, re.I) and not expected.startswith("paypal"):
            reject(payment_key, "PayPal", expected)
    quantity_key = "gb_amount" if "gb_amount" in args else "total_baggages" if "total_baggages" in args else None
    if quantity_key:
        pattern = r"\b(\d+(?:\.\d+)?)\s*GB\b" if quantity_key == "gb_amount" else r"\b(\d+)\s+(?:checked )?bags?\b"
        values = {float(x) for x in re.findall(pattern, active, re.I)}
        if values and (len(values) != 1 or float(args[quantity_key]) not in values):
            reject(quantity_key, sorted(values), args[quantity_key])
    return bad


def confirmed_write(prefix: list[dict], gold: dict) -> dict:
    consent = detect(prefix, gold)
    witness = consent.get("witness")
    result = {"ok": False, "reason": "upstream_offer_divergence", "detail": "no_current_confirmation"}
    if not consent["authorized_prior_consent"] or not witness:
        return result
    offer = witness["offer"]
    tools = offered_tools(offer, prefix[: witness["offer_index"]])
    if not tools:
        # Generic proceed questions may refer to the immediately preceding proposal.
        for m in reversed(prefix[: witness["offer_index"]]):
            if m.get("type") == "message" and m.get("role") == "assistant":
                candidate = str(m.get("content", ""))
                tools = offered_tools(candidate, prefix)
                if tools:
                    offer = candidate + " " + offer
                    break
    result.update(
        offer=offer, offer_index=witness["offer_index"], reply_index=witness["reply_index"], tools=sorted(tools)
    )
    if gold["name"] not in tools:
        return {**result, "detail": "confirmed_tool_mismatch"}
    target = discussion_target(prefix[: witness["offer_index"] + 1], gold)
    if not target["ok"]:
        return {**result, "detail": "confirmed_target_mismatch", "target": target}
    if gold["name"] == "modify_user_address" and not re.search(
        r"\b(?:profile|default)(?: shipping)? address\b", offer, re.I
    ):
        return {**result, "detail": "profile_target_unresolved"}
    bad = conflicts(prefix, offer, gold)
    if bad:
        return {**result, "detail": "confirmed_argument_conflict", "conflicts": bad}
    if not areal_v2.lookup_arguments_public(prefix, gold):
        return {**result, "detail": "omitted_argument_not_public"}
    for m in prefix[witness["reply_index"] + 1 :]:
        shape, call = areal_v2.one_turn([m])
        if shape == "call" and call["name"] == gold["name"]:
            keys = [
                k
                for k in ["order_id", "reservation_id", "user_id", "customer_id", "line_id"]
                if k in gold["arguments"]
            ]
            if all(areal_v2.typed_equal(call["arguments"].get(k), gold["arguments"][k]) for k in keys):
                return {**result, "detail": "confirmation_already_consumed"}
    return {**result, "ok": True, "reason": "confirmed_write_matches", "detail": "tool_target_values_bound"}
