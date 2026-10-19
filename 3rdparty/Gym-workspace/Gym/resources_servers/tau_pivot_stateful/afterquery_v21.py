# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Versioned, deterministic AfterQuery argument evaluator. No v1 checker imports.

The private schema table is content addressed; public tool identities remain exact.
Free text is a required-token predicate, not a semantic or truthfulness judge.
"""

import copy
import datetime as dt
import hashlib
import json
import re
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation
from pathlib import Path


FREE_TEXT = frozenset(
    {
        "description",
        "detail",
        "summary",
        "event",
        "notes",
        "reason",
        "evidence_summary",
        "wire_instructions",
        "authorized_by",
        "loss_summary",
    }
)
# Sentence-openers are not treated as proper names. This frozen list is part of
# the evaluator SHA, never inferred from model outputs or an audit draw.
NON_NAMES = frozenset(
    "a an the this that client caller customer guest user agent verified verification requested request please confirmed confirm no yes on at in for to from with and or but is was has had not none null submitted processed completed after before current same account bank payment transfer wire issue reason summary details refund return update cancel add remove set log record request complete done".split()
)
WORD = re.compile(r"[\w]+(?:[.@'’:/+-][\w]+|(?<=\d),(?=\d)\d+)*", re.UNICODE)


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def schema_key(name: str, schema: dict) -> str:
    return digest([name, schema])


def enum_values(schema: dict) -> list:
    if "enum" in schema:
        return schema["enum"]
    # Quoted names in a numeric/date/boolean constraint can refer to another
    # argument (e.g. limit_price required for limit/stop_limit order types).
    # Description options are string enums, never a replacement for numeric type.
    if schema.get("type") not in {None, "string"}:
        return []
    desc = schema.get("description", "")
    quoted = re.findall(r"['\"]([a-zA-Z0-9_ -]+)['\"]", desc)
    if len(quoted) >= 2:
        return list(dict.fromkeys(quoted))
    match = re.search(r"(?:must be )?one of\s*:?\s*([\w, /-]+)", desc, re.I)
    return [x.strip() for x in re.split(r",|\bor\b", match[1]) if x.strip()] if match else []


def compile_field(schema: dict, name: str = "") -> dict:
    field = copy.deepcopy(schema)
    description = schema.get("description", "")
    enums = enum_values(schema)
    field["v2_enum"] = enums
    field["v2_case_insensitive"] = True
    kind = schema.get("type")
    key = name.casefold()
    if enums:
        mode = "enum"
    elif not schema:
        mode = "json"
    elif kind == "object" or "properties" in schema:
        mode = "object"
    elif kind == "array":
        mode = "array"
    elif kind in {"number", "integer", "boolean", "null"}:
        mode = kind
    elif (
        schema.get("format") in {"date", "date-time"}
        or re.search(r"\b(date|dob|birth)\b", key.replace("_", " "))
        or re.search(r"MM?/DD?/YYYY|YYYY-MM-DD", description, re.I)
    ):
        mode = "datetime" if schema.get("format") == "date-time" or "time" in key else "date"
    elif key in {"time_verified", "timestamp", "datetime"}:
        mode = "datetime"
    elif (
        re.search(r"phone number", description, re.I)
        and re.search(r"e[ -]?mail", description, re.I)
        and re.search(r"\bor\b", description)
    ):
        mode = "contact"
    elif "phone" in key or re.search(r"phone number", description, re.I):
        mode = "phone"
    elif key in FREE_TEXT or re.search(r"summary|description|free[ -]?text", description, re.I):
        mode = "free_text"
    else:
        mode = "exact"
    field["v2_mode"] = mode
    # These policies are frozen in the content-addressed per-schema table.
    field["v21_structured_string"] = kind == "string" and bool(re.search(r"\brecord\s*\(|\bJSON\b", description, re.I))
    field["v21_empty_reference"] = "dont_care"
    field["v21_structured_keys"] = "casefold_remove_separators_reference_subset"
    field["v21_synonyms"] = {"dob": "dateofbirth"} if re.search(r"\bdob\b", description, re.I) else {}
    field["v21_short_text"] = (
        mode == "exact" and kind == "string" and not re.search(r"(?:^|_)(?:id|number|code|vin|identifier)(?:$|_)", key)
    )

    if mode == "object":
        field["properties"] = {k: compile_field(v, k) for k, v in sorted(schema.get("properties", {}).items())}
    if mode == "array":
        field["items"] = compile_field(schema.get("items", {}), name)
    return field


def build_table(rows: list[dict]) -> dict:
    schemas = {}
    for row in rows:
        for raw in row["tools"]:
            tool = raw.get("function", raw)
            schema = tool.get("parameters", tool.get("params", {}))
            key = schema_key(tool["name"], schema)
            schemas[key] = {"name": tool["name"], "schema": schema, "field": compile_field(schema)}
    return {"format": "afterquery-v2.1-field-table-1", "schemas": dict(sorted(schemas.items()))}


def version(table: dict) -> str:
    return "afterquery-v2.1:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest() + ":" + digest(table)


def public_name(name: str, names: set[str]) -> str:
    name = name.removeprefix("@discoverable:")
    if name in names:
        return name
    stripped = re.sub(r"_\d+$", "", name)
    return stripped if stripped in names else name


def ledger(row: dict) -> list[dict]:
    names = {t.get("function", t)["name"] for t in row["tools"]}
    schemas = {
        t.get("function", t)["name"]: t.get("function", t).get("parameters", t.get("function", t).get("params", {}))
        for t in row["tools"]
    }
    output = []
    for i, step in enumerate(row.get("remaining_gold") or []):
        if isinstance(step, dict):
            if step.get("done"):
                continue
            actor, name, args = step.get("requestor", "assistant"), step["name"], step["arguments"]
        else:
            actor, name, args = step
        name = public_name(name, names)
        output.append(
            {
                "gold_index": i,
                "actor": actor,
                "name": name,
                "arguments": args,
                "schema_key": schema_key(name, schemas[name]) if name in schemas else None,
            }
        )
    return output


def required_tokens(text: str) -> list[str]:
    tokens = set()
    for token in WORD.findall(text):
        # Every numeral-bearing token protects IDs, dates, amounts and quantities.
        # Capitalized words conservatively protect gold proper-noun spans.
        if any(c.isdigit() for c in token) or (token[0].isupper() and token.casefold() not in NON_NAMES):
            tokens.add(token_key(token))
    return sorted(tokens)


def text_has(text: str, token: str) -> bool:
    return token in {token_key(x) for x in WORD.findall(text)}


def token_key(token: str) -> str:
    if re.fullmatch(r"\d+(?:,\d{3})*(?:\.\d+)?", token):
        return format(Decimal(token.replace(",", "")).normalize(), "f")
    return token.casefold()


def date_value(value: str, *, timestamp: bool = False) -> str:
    text = str(value).strip()
    if not timestamp:
        for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
            try:
                return dt.datetime.strptime(text, fmt).date().isoformat()
            except ValueError:
                pass
        raise ValueError("invalid date")
    # Recognized timezone labels are retained; no assumed DST conversions.
    match = re.fullmatch(r"(.+?)\s+([A-Z]{2,5})", text)
    zone = match[2] if match else ""
    text = match[1] if match else text
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M"):
        try:
            return dt.datetime.strptime(text, fmt).isoformat() + (" " + zone if zone else "")
        except ValueError:
            pass
    return dt.datetime.fromisoformat(text.replace("Z", "+00:00")).isoformat() + (" " + zone if zone else "")


class Pairs(dict):
    """Retain duplicate literal JSON keys while preserving JSON object behavior."""

    def __init__(self, pairs: list):
        super().__init__(pairs)
        self.pairs = pairs


def empty_reference(value: object) -> bool:
    return value is None or value == "" or value == []


def structured(value: object) -> object:
    if isinstance(value, str):
        try:
            value = json.loads(value, object_pairs_hook=Pairs)
        except (ValueError, TypeError):
            return value
    return value


def key_name(key: str, field: dict) -> str:
    key = re.sub(r"[\W_]", "", key.casefold())
    return field.get("v21_synonyms", {}).get(key, key)


def structured_equal(actual: object, expected: object, field: dict) -> bool:
    if empty_reference(expected):
        return True
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        items = actual.pairs if isinstance(actual, Pairs) else list(actual.items())
        for key, value in expected.items():
            if empty_reference(value):
                continue
            values = [v for k, v in items if key_name(k, field) == key_name(key, field)]
            if not values or not all(structured_equal(v, value, field) for v in values):
                return False
        return True
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(structured_equal(a, b, field) for a, b in zip(actual, expected))
        )
    # Inside a structured string use the original canonical scalar rules, never
    # short-text or free-text leniency for identifiers or factual leaf values.
    return json_equivalent(actual, expected)


SHORT_STOP = frozenset("a an the of for to in on at by with and or is was be this that".split())


def short_tokens(value: object) -> list[str] | None:
    if not isinstance(value, str) or any(c.isdigit() for c in value):
        return None
    if re.search(r"[_/@:#=]", value) or re.fullmatch(r"[A-Z]{2,}", value):
        return None
    words = WORD.findall(value)
    if not 1 <= len(words) <= 4:
        return None
    tokens = sorted({w.casefold() for w in words} - SHORT_STOP)
    return tokens or None


def normalize(value: object, field: dict, *, gold: bool = False) -> object:
    mode = field["v2_mode"]
    if gold and empty_reference(value):
        return value
    if field.get("v21_structured_string") and not isinstance(structured(value), (dict, list)):
        raise ValueError("structured record required")
    if field.get("type") == "string" and isinstance(structured(value), (dict, list)):
        return structured(value)
    if mode == "json":
        # JSON Schema {} permits any JSON value. It supplies no free-text or
        # coercion rule, so compare that subtree exactly, including all its keys.
        return json.loads(canonical(value))
    if value is None:
        if field.get("type") == "null" or any(x.get("type") == "null" for x in field.get("anyOf", [])):
            return None
        raise ValueError("null not allowed")
    if mode == "object":
        if not isinstance(value, dict):
            raise ValueError("object required")
        props = field.get("properties", {})
        missing = set(field.get("required", [])) - value.keys()
        if missing:
            raise ValueError("required argument missing: " + ",".join(sorted(missing)))
        if value.keys() - props.keys():
            raise ValueError("unknown arguments")
        out = {}
        for k, f in props.items():
            if k in value:
                out[k] = normalize(value[k], f, gold=gold)
            elif "default" in f:
                out[k] = normalize(f["default"], f, gold=gold)
        return out
    if mode == "array":
        if not isinstance(value, list):
            raise ValueError("array required")
        return [normalize(x, field["items"], gold=gold) for x in value]
    if mode in {"number", "integer"}:
        if isinstance(value, bool):
            raise ValueError("boolean is not number")
        number = Decimal(str(value).replace(",", ""))
        if not number.is_finite() or (mode == "integer" and number != number.to_integral_value()):
            raise ValueError("invalid numeric value")
        if "minimum" in field and number < Decimal(str(field["minimum"])):
            raise ValueError("below minimum")
        if "maximum" in field and number > Decimal(str(field["maximum"])):
            raise ValueError("above maximum")
        quantum = Decimal(str(field.get("multipleOf", ".000000001")))
        return str((number / quantum).quantize(Decimal(1), rounding=ROUND_HALF_EVEN) * quantum)
    if mode == "boolean":
        if value is True or value is False:
            return value
        if str(value).casefold() in {"true", "false"}:
            return str(value).casefold() == "true"
        raise ValueError("boolean required")
    if mode == "null":
        raise ValueError("null required")
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        # The historical schema-stringified-object artifact has a deterministic
        # valid canonical representation, instead of silently accepting any JSON.
        if gold and field.get("type") == "string" and isinstance(value, (dict, list)):
            value = canonical(value)
        else:
            raise ValueError("string required")
    text = str(value).strip()
    if "minLength" in field and len(text) < field["minLength"]:
        raise ValueError("string too short")
    if "maxLength" in field and len(text) > field["maxLength"]:
        raise ValueError("string too long")
    if "pattern" in field and not re.search(field["pattern"], text):
        raise ValueError("string pattern mismatch")
    if mode == "enum":
        if field["v2_case_insensitive"]:
            text = text.casefold()
        options = [str(x).casefold() if field["v2_case_insensitive"] else str(x) for x in field["v2_enum"]]
        if text not in options:
            raise ValueError("invalid enum")
    elif mode in {"date", "datetime"}:
        if text == "" and field.get("default") == "" and not field.get("format"):
            return ""
        return date_value(text, timestamp=mode == "datetime")
    elif mode == "contact" and "@" in text:
        return text
    elif mode in {"phone", "contact"}:
        if text == "" and field.get("default") == "":
            return ""
        text = re.sub(r"\D", "", text)
        if not text:
            raise ValueError("empty phone")
    elif mode == "free_text" and not text:
        raise ValueError("empty free text")
    # Numeric-string identifier representations 2/2.0 canonicalize exactly;
    # leading-zero identifiers stay distinct. Case and separators stay exact.
    elif mode == "exact" and re.fullmatch(r"-?(?:0|[1-9]\d*)(?:\.0+)?", text):
        return str(Decimal(text).quantize(Decimal(1)))
    return text


def equivalent(actual: object, expected: object, field: dict) -> bool:
    if empty_reference(expected):
        return True
    mode = field["v2_mode"]
    if field.get("v21_structured_string") and not isinstance(structured(expected), (dict, list)):
        raise ValueError("structured record reference required")
    if field.get("type") == "string" and isinstance(structured(expected), (dict, list)):
        return structured_equal(structured(actual), structured(expected), field)
    if mode == "object":
        if not isinstance(actual, dict) or not isinstance(expected, dict):
            return False
        props = field.get("properties", {})
        if actual.keys() - props.keys():
            return False
        for key, prop in props.items():
            gold = expected.get(key, prop.get("default"))
            if empty_reference(gold):
                continue
            if key not in actual and "default" not in prop:
                return False
            if not equivalent(actual.get(key, prop.get("default")), gold, prop):
                return False
        return True
    if mode == "array":
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(equivalent(x, y, field["items"]) for x, y in zip(actual, expected))
        )
    a, b = normalize(actual, field), normalize(expected, field, gold=True)
    if mode == "free_text":
        return bool(a) and all(text_has(a, token) for token in required_tokens(b))
    tokens = short_tokens(b) if field.get("v21_short_text") else None
    if tokens:
        return bool(a) and all(text_has(a, token) for token in tokens)
    if mode == "json":
        return json_equivalent(a, b)
    return a == b


def json_equivalent(actual: object, expected: object) -> bool:
    """Exact unknown-schema structure, with the registered scalar number coercion."""
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(actual, dict) or isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and isinstance(expected, dict)
            and actual.keys() == expected.keys()
            and all(json_equivalent(actual[k], expected[k]) for k in actual)
        )
    if isinstance(actual, list) or isinstance(expected, list):
        return (
            isinstance(actual, list)
            and isinstance(expected, list)
            and len(actual) == len(expected)
            and all(json_equivalent(a, b) for a, b in zip(actual, expected))
        )
    numeric = lambda value: (
        isinstance(value, (int, float))
        or isinstance(value, str)
        and bool(re.fullmatch(r"-?(?:0|[1-9]\d*)(?:\.\d+)?", value))
    )
    if numeric(actual) and numeric(expected):
        field = {"v2_mode": "number"}
        return normalize(actual, field) == normalize(expected, field)
    return type(actual) is type(expected) and actual == expected


def evaluate(spec: dict, output: list[dict], state: dict, table: dict) -> dict:
    if spec["verifier_version"] != version(table):
        raise ValueError("v2 evaluator/table version mismatch")
    output = [x for x in output if x.get("type") != "reasoning"]
    call = output[0] if len(output) == 1 else {}
    shape = call.get("type") == "function_call" and call.get("role", "assistant") == "assistant"
    not_end = call.get("name") not in {"end_call", "transfer_to_human_agents", "transfer_to_specialist"}
    matches = []
    error = None
    if shape and not_end:
        try:
            args = call.get("arguments", {})
            args = json.loads(args) if isinstance(args, str) else args
            if not isinstance(args, dict):
                raise ValueError("arguments must be object")
            for step in state["ledger"]:
                if step["actor"] != "assistant" or step["name"] != call["name"] or not step["schema_key"]:
                    continue
                field = table["schemas"][step["schema_key"]]["field"]
                try:
                    if equivalent(args, step["arguments"], field):
                        matches.append(step["gold_index"])
                except (ValueError, TypeError, InvalidOperation, OverflowError):
                    continue
        except (ValueError, TypeError, InvalidOperation, OverflowError) as exc:
            error = str(exc)
    predicates = {"turn_shape": bool(shape), "not_end": bool(not_end), "remaining_gold_match": bool(matches)}
    if set(predicates) != {p["id"] for p in spec["predicates"]}:
        raise ValueError("v2 predicate/spec mismatch")
    return {
        "reward": int(all(predicates.values())),
        "predicates": predicates,
        "native": {
            "accepted": bool(matches),
            "matched_gold": bool(matches),
            "gold_indices": matches,
            "reason": error or ("v2_match" if matches else "v2_mismatch"),
        },
    }


def spec_for(verifier_version: str) -> dict:
    return {
        "kind": "afterquery_v21_step",
        "verifier_version": verifier_version,
        "expected_turn_kind": "write",
        "predicates": [
            {"id": k, "rule": v}
            for k, v in [
                ("turn_shape", "single assistant tool call"),
                ("not_end", "no end/transfer"),
                (
                    "remaining_gold_match",
                    "assistant identity plus schema canonical arguments and gold-required free-text tokens",
                ),
            ]
        ],
        "reward": "1 iff all predicates pass, else 0",
    }
