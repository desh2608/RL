# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real HTTP control check from this bundle's samples and private records; no models."""

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import socket
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def response(output):
    fixed = []
    for i, raw in enumerate(output):
        if raw.get("type") == "reasoning":
            continue
        item = dict(raw)
        if item.get("type") == "function_call":
            item.update(id="fc_" + str(i), status="completed")
            if not isinstance(item.get("arguments"), str):
                item["arguments"] = json.dumps(item["arguments"], ensure_ascii=False)
            item.setdefault("call_id", "control_" + str(i))
        else:
            content = item.get("content", "")
            if isinstance(content, str):
                content = [{"type": "output_text", "text": content, "annotations": []}]
            item.update(
                id="msg_" + str(i),
                type="message",
                role="assistant",
                status="completed",
                content=content,
            )
        fixed.append(item)
    return {
        "id": "control",
        "created_at": 0,
        "model": "no-model",
        "object": "response",
        "output": fixed,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


def start(stage, port):
    from unittest.mock import MagicMock

    import uvicorn

    from nemo_gym.server_utils import ServerClient
    from resources_servers.tau_pivot_stateful.app import Config, TauPivotStateful

    config = json.loads((stage / "services.json").read_text())["tau_pivot_stateful"]["resources_servers"][
        "tau_pivot_stateful"
    ]
    config = Config(name="tau_pivot_stateful", host="127.0.0.1", port=port, **config)
    server = TauPivotStateful(config=config, server_client=MagicMock(spec=ServerClient))
    sock = socket.socket()
    sock.bind(("127.0.0.1", port))
    sock.listen(128)
    web = uvicorn.Server(uvicorn.Config(server.setup_webserver(), log_level="error"))
    thread = threading.Thread(target=web.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 60
    while not web.started:
        if not thread.is_alive() or time.monotonic() > deadline:
            raise RuntimeError("Resources server did not start")
        time.sleep(0.05)
    return server, web, thread, "http://127.0.0.1:" + str(sock.getsockname()[1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--stage", type=Path, required=True)
    p.add_argument("--out", type=Path)
    p.add_argument("--url")
    p.add_argument("--version", choices=["v2", "v3"])
    p.add_argument("--max-records", type=int)
    p.add_argument(
        "--baseline-rewards", action="store_true", help="Measure baseline rewards without imposing new expectations"
    )
    p.add_argument("--serve", action="store_true")
    p.add_argument("--serve-only", action="store_true")
    p.add_argument("--port", type=int, default=0)
    a = p.parse_args()
    meta = json.loads((a.stage / "STAGED.json").read_text())
    assert sha(a.stage / "services.json") == meta["services_sha256"]
    for name, pin in meta["code_files"].items():
        assert sha(Path(meta["code_root"]) / name) == pin
    records = {}
    samples = {}
    allowed = set(json.loads((a.stage / "CACHE_MANIFEST.json").read_text())["records"])
    for pool in ["text", "audio_only"]:
        for row in map(json.loads, (a.bundle / pool / "records.jsonl").open()):
            if row["record_id"] in allowed:
                records[row["record_id"]] = row
        for sample in map(json.loads, (a.bundle / pool / "all/sample.jsonl").open()):
            ident = sample["verification_ref"]["record_id"]
            if ident in allowed:
                samples[ident] = sample
    assert set(samples) == set(records) == allowed
    if a.version:
        records = {k: v for k, v in records.items() if v["version"] == a.version}
        samples = {k: samples[k] for k in records}
    if a.max_records:
        chosen = []
        for key, row in sorted(records.items()):
            label = (row["version"], row["category"])
            if (
                sum(records[x]["version"] == label[0] and records[x]["category"] == label[1] for x in chosen)
                < a.max_records
            ):
                chosen.append(key)
        records = {k: records[k] for k in chosen}
        samples = {k: samples[k] for k in records}
    server = web = thread = None
    if a.serve or a.serve_only:
        server, web, thread, a.url = start(a.stage, a.port)
    try:
        if a.serve_only:
            print("Resources server: " + a.url, flush=True)
            while thread.is_alive():
                time.sleep(1)
            return
        assert a.url and a.out
        a.out.mkdir(parents=True, exist_ok=False)
        endpoint = urlsplit(a.url)
        tasks = []
        for ident, row in sorted(records.items()):
            for name, c in row["controls"].items():
                tasks.append((ident, name, c["output"], c["expected"]))

        def call(task):
            ident, name, output, expected = task
            body = dict(
                responses_create_params=samples[ident]["responses_create_params"],
                verification_ref=samples[ident]["verification_ref"],
                response=response(output),
            )
            conn = http.client.HTTPConnection(endpoint.hostname, endpoint.port, timeout=360)
            try:
                conn.request("POST", "/verify", json.dumps(body), {"Content-Type": "application/json"})
                reply = conn.getresponse()
                data = json.loads(reply.read())
                ok = (
                    reply.status == 200
                    and not data.get("diagnostics", {}).get("error")
                    and data.get("reward") == expected
                )
                healthy = reply.status == 200 and not data.get("diagnostics", {}).get("error")
                return dict(
                    record_id=ident,
                    kind=records[ident]["kind"],
                    control=name,
                    expected=expected,
                    reward=data.get("reward"),
                    http_status=reply.status,
                    matched=ok,
                    healthy=healthy,
                    diagnostics=data.get("diagnostics", {}),
                    request_sha256=hashlib.sha256(
                        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
                    ).hexdigest(),
                )
            except Exception as e:
                return dict(record_id=ident, control=name, expected=expected, matched=False, error=str(e))
            finally:
                conn.close()

        results = []
        with (
            (a.out / "PARITY.jsonl").open("w") as stream,
            concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor,
        ):
            for x in executor.map(call, tasks):
                results.append(x)
                stream.write(json.dumps(x) + "\n")
                stream.flush()
                if len(results) % 100 == 0:
                    print(json.dumps(dict(done=len(results), matched=sum(x["matched"] for x in results))), flush=True)
        report = dict(
            status="PASS"
            if all(x.get("healthy") if a.baseline_rewards else x["matched"] for x in results)
            else "FAIL",
            records=len(records),
            expected=len(tasks),
            matched=sum(x["matched"] for x in results),
            failures=[x for x in results if not x["matched"]],
            runtime_keys=meta["keys"],
            code_commit=meta["code_commit"],
            model_calls=0,
            baseline_rewards=a.baseline_rewards,
        )
        (a.out / "PARITY.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "failures"}), flush=True)
        if report["status"] != "PASS":
            raise SystemExit(1)
    finally:
        if server is not None and server._engine is not None:
            server._engine.close()
        if web is not None:
            web.should_exit = True
            thread.join(timeout=15)


if __name__ == "__main__":
    main()
