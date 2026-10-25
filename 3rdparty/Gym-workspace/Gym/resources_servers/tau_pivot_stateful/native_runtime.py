# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Node-local archived native runtime. Captures reference closures, never patches checkers."""

import argparse
import contextlib
import copy
import gc
import hashlib
import importlib.machinery
import inspect
import json
import os
import resource
import socket
import sys
import types
from functools import lru_cache
from pathlib import Path

from resources_servers.tau_pivot_stateful import schema_compare


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "reference"))
from recipes.tau_voice_rl_pivot import areal_v2
from recipes.tau_voice_rl_pivot import pivot_verify as reference
from recipes.tau_voice_rl_pivot import replay_five as replay


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap(frozen, package):
    socket.socket.connect = replay.no_network
    socket.create_connection = replay.no_network
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    os.environ["HF_HUB_OFFLINE"] = "1"
    if package:
        base = frozen / "site-packages/tau-company/tau2"
        parent = types.ModuleType("tau2")
        parent.__path__ = [str(base)]
        parent.__spec__ = importlib.machinery.ModuleSpec("tau2", loader=None, is_package=True)
        domains = types.ModuleType("tau2.domains")
        domains.__path__ = [str(package / "src/tau2/domains"), str(base / "domains")]
        domains.__spec__ = importlib.machinery.ModuleSpec("tau2.domains", loader=None, is_package=True)
        parent.domains = domains
        sys.modules["tau2"] = parent
        sys.modules["tau2.domains"] = domains


class Captured(Exception):
    pass


def build(row, raw, root, dest, package, gold):
    import cloudpickle

    pin = json.loads((root / "PIN_RECEIPT.json").read_text())
    assert pin["status"] == "PASS"
    # Hash check the complete archived closure once per worker, then record it.
    check_pin(str(root / "frozen"), json.dumps(pin["frozen_file_sha256"], sort_keys=True))
    names = pin.get("verifier_module_set") or (
        ["areal_step_verifier.py", "areal_policy_checks.py", "areal_telecom_policy.py"]
        if row["family"].startswith("areal")
        else ["afterquery_step_verifier.py", "afterquery_adapter.py"]
    )
    versions = {name: pin["frozen_file_sha256"]["code/recipes/tau_voice_sdg/" + name] for name in names}
    expected_version = row["family"] + ":" + str(row["launch"].get("code_commit")) + ":" + replay.digest(versions)
    assert expected_version == row["verification"]["verifier_version"], "Frozen checker version mismatch"
    report = {"gates": {}, "verifier_version": row["verification"]["verifier_version"]}
    payload = {}

    def capture(spec, output, *, native_check=None, consent_check=None):
        frame = inspect.currentframe().f_back
        while frame and frame.f_code.co_name not in {"run_areal", "run_afterquery"}:
            frame = frame.f_back
        if frame is None:
            raise RuntimeError("Reference boundary frame absent")
        env = frame.f_locals["env"]
        verifier = frame.f_locals["verifier"]
        state = {
            "environment": replay.state_of(env),
            "ledger": verifier.gold if row["family"].startswith("areal") else verifier.remaining,
        }
        if row["family"].startswith("afterquery"):
            state["caller_grants"] = copy.deepcopy(getattr(env.tools, "_user_discoverable_tools_state", None) or {})
        assert replay.digest(state) == report["state_hash_at_boundary"] == row["state_hash_at_boundary"], (
            "Boundary state mismatch"
        )
        payload.update(state=state, native_check=native_check, consent_check=consent_check)
        raise Captured()

    pivot = types.SimpleNamespace(
        **{k: getattr(reference, k) for k in ["cosmetic_control_text", "consent_predicates", "spec_for"]},
        evaluate=capture,
    )
    try:
        if row["family"].startswith("areal"):
            replay.run_areal(raw, root, report, pivot)
        else:
            replay.run_afterquery(raw, root, report, pivot, gold, gold_grants=True, ordered_callers=True)
    except Captured:
        pass
    assert payload and payload["native_check"]
    dest.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(row["record_id"].encode()).hexdigest()
    data = cloudpickle.dumps(
        {
            "native_check": payload["native_check"] if row["expected_turn_kind"] == "write" else None,
            "consent_check": payload["consent_check"],
        }
    )
    (dest / (key + ".pkl")).write_bytes(data)
    entry = {
        "record_id": row["record_id"],
        "dataset": row.get("dataset", "v1-areal"),
        "verifier_version": row["verification"]["verifier_version"],
        "expected_turn_kind": row["expected_turn_kind"],
        "verification": row["verification"],
        "state": payload["state"],
        "state_hash_at_boundary": replay.digest(payload["state"]),
        "payload": key + ".pkl",
        "payload_sha256": hashlib.sha256(data).hexdigest(),
        "pinned_modules": {
            k: v for k, v in pin["frozen_file_sha256"].items() if k.startswith("code/recipes/tau_voice_sdg/")
        },
        "model_calls": 0,
    }
    (dest / (key + ".json")).write_text(json.dumps(entry, ensure_ascii=False) + "\n")
    return {
        "record_id": row["record_id"],
        "entry": key + ".json",
        "entry_sha256": sha(dest / (key + ".json")),
        "state_hash_at_boundary": entry["state_hash_at_boundary"],
        "matched": True,
    }


@lru_cache(maxsize=1)
def check_pin(root, hashes):
    for name, digest in json.loads(hashes).items():
        if sha(Path(root) / name) != digest:
            raise ValueError("Frozen asset changed: " + name)


def load_cached(root, entry_name, entry_sha):
    import cloudpickle

    path = Path(root) / entry_name
    assert sha(path) == entry_sha
    entry = json.loads(path.read_text())
    blob = Path(root) / entry["payload"]
    assert sha(blob) == entry["payload_sha256"] and replay.digest(entry["state"]) == entry["state_hash_at_boundary"]
    del entry["state"]  # Already hash-verified; callbacks own the required boundary.
    if entry.get("schema_contract") not in (None, schema_compare.CONTRACT):
        raise ValueError("Unknown schema comparison contract")
    if entry.get("schema_contract") == schema_compare.CONTRACT:
        entry["schema_verification"] = schema_compare.tree(entry["verification"], entry["schema_tools"])
    callbacks = cloudpickle.loads(blob.read_bytes())
    return entry, callbacks


def verify(root, item, ref, output):
    entry, callbacks = load_cached(str(root), item["entry"], item["entry_sha256"])
    for key in ["record_id", "dataset", "verifier_version", "expected_turn_kind"]:
        if entry[key] != ref[key]:
            raise ValueError("Verification reference mismatch: " + key)
    verification = entry.get("schema_verification", entry["verification"])
    if entry.get("schema_contract") == schema_compare.CONTRACT:
        output = schema_compare.tree(output, entry["schema_tools"])
    if ref["verifier_version"].startswith("areal-v2:"):
        proof = entry["verification"].get("eligibility", {})
        if proof.get("state_hash_at_boundary") != entry["state_hash_at_boundary"]:
            raise ValueError("G5 eligibility proof belongs to a different boundary")
        if entry["verification"].get("verifier_version") != ref["verifier_version"]:
            raise ValueError("G5 specification version differs from verification reference")
        return areal_v2.evaluate(verification, output, **callbacks)
    return reference.evaluate(verification, output, **callbacks)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["build", "serve"], required=True)
    p.add_argument("--frozen", type=Path, required=True)
    p.add_argument("--package", type=Path)
    p.add_argument("--gold", type=Path)
    p.add_argument("--input", type=Path)
    p.add_argument("--raw-root", type=Path)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--pin", type=Path)
    p.add_argument("--cache-entries", type=int)
    a = p.parse_args()
    if a.mode == "serve" and (a.cache_entries is None or a.cache_entries < 1):
        p.error("cache-entries must be positive")
    global load_cached
    if a.mode == "serve":
        load_cached = lru_cache(maxsize=a.cache_entries)(load_cached)
    bootstrap(a.frozen, a.package)
    if a.mode == "build":
        receipts = []
        for line in a.input.open():
            row = json.loads(line)
            raw = json.loads(
                (a.raw_root / (hashlib.sha256(row["record_id"].encode()).hexdigest() + ".json")).read_text()
            )
            root = a.cache / "restore"
            root.mkdir(parents=True, exist_ok=True)
            if not (root / "frozen").exists():
                (root / "frozen").symlink_to(a.frozen, target_is_directory=True)
            (root / "PIN_RECEIPT.json").write_bytes(a.pin.read_bytes())
            try:
                receipt = build(row, raw, root, a.cache / "states", a.package, a.gold)
            except Exception as exc:
                failed = {
                    "record_id": row["record_id"],
                    "matched": False,
                    "excluded": True,
                    "error": type(exc).__name__ + ": " + str(exc),
                }
                (a.cache / "FAILED_RECORD.json").write_text(json.dumps(failed, indent=2) + "\n")
                (a.cache / "CACHE_INDEX.json").write_text(
                    json.dumps(
                        {
                            "records": receipts + [failed],
                            "coverage": len(receipts),
                            "matched": len(receipts),
                            "status": "STOPPED",
                            "model_calls": 0,
                        },
                        indent=2,
                    )
                    + "\n"
                )
                raise
            receipts.append(receipt)
            print(json.dumps({"cached": len(receipts), "total_state_hash_match": True}), flush=True)
        with contextlib.redirect_stdout(sys.stderr):
            from recipes.tau_voice_rl_pivot.import_guard import assert_no_lustre_imports

            checked = assert_no_lustre_imports()
        (a.cache / "CACHE_INDEX.json").write_text(
            json.dumps(
                {
                    "records": receipts,
                    "coverage": len(receipts),
                    "matched": len(receipts),
                    "model_calls": 0,
                    "lustre_import_guard": checked,
                },
                indent=2,
            )
            + "\n"
        )
    else:
        protocol = sys.stdout
        count = 0
        for line in sys.stdin:
            try:
                query = json.loads(line)
                with contextlib.redirect_stdout(sys.stderr):
                    result = verify(a.cache, query["item"], query["ref"], query["output"])
                    if count == 0 or (count + 1) % 500 == 0:
                        from recipes.tau_voice_rl_pivot.import_guard import assert_no_lustre_imports

                        assert_no_lustre_imports()
                response = {"ok": True, "result": result}
            except Exception as exc:
                response = {"ok": False, "error": type(exc).__name__ + ": " + str(exc)}
            gc.collect()
            count += 1
            rss = (
                int(
                    next(
                        x.split()[1]
                        for x in Path("/proc/self/status").read_text().splitlines()
                        if x.startswith("VmRSS:")
                    )
                )
                * 1024
            )
            metrics = {
                "pid": os.getpid(),
                "calls": count,
                "rss_bytes": rss,
                "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                "cache": load_cached.cache_info()._asdict(),
            }
            response["worker"] = metrics
            if count == 1 or count % 500 == 0:
                print(json.dumps({"worker_rss": metrics}), file=sys.stderr, flush=True)
            protocol.write(json.dumps(response, default=str) + "\n")
            protocol.flush()


if __name__ == "__main__":
    main()
