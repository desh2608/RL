#!/usr/bin/env python3
"""Select the Ray Client port and publish Ray's actual dynamic GCS address."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import socket
import time


def launcher_lines(source):
    """Read this launcher's shell lines, excluding comments and Python heredocs.

    This is a deliberately narrow regression guard, not a general Bash parser.
    Unrecognized startup structure must be reviewed instead of silently accepted.
    """
    lines, pending, heredoc = [], "", None
    for raw in source.splitlines():
        if heredoc is not None:
            if raw.strip() == heredoc:
                heredoc = None
            continue
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        pending += raw.strip()
        if pending.endswith("\\"):
            pending = pending[:-1] + " "
            continue
        lexer = shlex.shlex(pending, posix=True, punctuation_chars=";&|()")
        lexer.whitespace_split = True
        lines.append((pending, list(lexer)))
        match = re.search(r"<<-?['\"]?([A-Za-z_]\w*)", pending)
        if match:
            heredoc = match.group(1)
        pending = ""
    if pending or heredoc:
        raise ValueError("Incomplete launcher shell/heredoc structure")
    return lines


def validate_launcher(source):
    """Reject known fixed-port and unscoped-cleanup regressions before allocation."""
    lines = launcher_lines(source)

    def require(condition, reason):
        if not condition:
            raise ValueError("Ray launcher contract: " + reason)

    for _, tokens in lines:
        for index, token in enumerate(tokens):
            require(token.rsplit("/", 1)[-1] not in ("pkill", "killall"), "global process cleanup is forbidden")
            if token in ("$ray", "${ray}") or token.rsplit("/", 1)[-1] == "ray":
                require(tokens[index + 1:index + 2] != ["stop"], "global ray stop is forbidden")
                if tokens[index + 1:index + 2] == ["start"]:
                    require(index == 0 and token == "$ray", "unrecognized Ray startup command")
            if token == "kill":
                require(tokens[index + 1:index + 3] in (["-0", "$ray_pid"], ["-TERM", "$ray_pid"]),
                        "signals must target only the registered Ray supervisor")

    common = re.search(r"(?ms)^args=\((.*?)\)\s*$", source)
    require(common is not None, "shared args array is missing or changed")
    assignments = [match.group(0) for line, _ in lines
                   for match in re.finditer(r"\bargs(?:\[[^\]]*\])?\+?=", line)]
    require(assignments == ["args="], "shared args may only be assigned once; later array overrides are forbidden")
    common_tokens = shlex.split(common.group(1), comments=True)
    required_ports = {
        "min-worker-port", "max-worker-port", "node-manager-port", "object-manager-port",
        "runtime-env-agent-port", "dashboard-agent-grpc-port", "dashboard-agent-listen-port",
        "metrics-export-port",
    }

    def ports(tokens):
        found = {}
        for token in tokens:
            if token.startswith("--") and "port" in token.split("=", 1)[0]:
                require("=" in token, "port flags must use explicit --name=value syntax")
                key, value = token[2:].split("=", 1)
                require(key not in found, "duplicate port flag: " + key)
                found[key] = value
        return found

    require(ports(common_tokens) == dict.fromkeys(required_ports, "0"),
            "all eight internal/worker ports must occur exactly once and be zero")
    starts = [(index, tokens) for index, (_, tokens) in enumerate(lines) if tokens[:2] == ["$ray", "start"]]
    require(len(starts) == 2, "expected exactly one head and one worker startup")
    for number, (index, tokens) in enumerate(starts):
        require("--block" in tokens and tokens[-1:] == ["&"] and tokens.count("${args[@]}") == 1,
                "each Ray startup must supervise its own background runtime with shared args")
        require(index + 2 < len(lines), "incomplete startup/PID/readiness sequence")
        require(lines[index + 1][1] == ["ray_pid=$!"], "capture the exact supervisor PID immediately")
        ready = lines[index + 2][1]
        require(ready[:3] == ["$python", "$run/scripts/ray_ports.py", "ready"] and
                "--supervisor-pid" in ready and
                ready[ready.index("--supervisor-pid") + 1:ready.index("--supervisor-pid") + 2] == ["$ray_pid"],
                "publish/join readiness immediately through the registered supervisor")
        if number == 0:
            require("--head" in tokens and ports(tokens) == {
                "port": "0", "dashboard-port": "0", "ray-client-server-port": "$client_port"},
                "head ports must be dynamic; no later fixed-port override is allowed")
            require(lines[index - 1][1] == ["client_port=$", "(", "$python", "$run/scripts/ray_ports.py",
                                           "client", "--ip", "$ip", ")"],
                    "select the nonzero Ray Client port immediately before head startup")
            require(ready[-2:] == ["--client-port", "$client_port"], "record the chosen Ray Client port")
        else:
            require("--head" not in tokens and "--address=$head_address" in tokens and not ports(tokens),
                    "worker must join the published head without overriding shared ports")
            require(ready[-2:] == ["--expected-address", "$head_address"], "verify the worker joined the published head")

    cleanup = source.split("cleanup() {", 1)
    require(len(cleanup) == 2 and "trap cleanup EXIT" in cleanup[1], "owned cleanup trap is missing")
    cleanup_tokens = [tokens for _, tokens in launcher_lines(cleanup[1].split("trap cleanup EXIT", 1)[0])]
    require(["kill", "-TERM", "$ray_pid"] in cleanup_tokens and ["wait", "$ray_pid"] in cleanup_tokens,
            "cleanup must signal and wait for its owned supervisor")
    return {"status": "PASS", "scope": "Known Ray launcher structural regressions; not a runtime or general Bash proof"}


def choose_client_port(ip):
    # Ray 2.56 rejects client port=0. Release this OS-selected port just before
    # starting Ray; a competing bind during that handoff must fail closed.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind((ip, 0))
        return listener.getsockname()[1]


def validate_address(address, ip, expected=None):
    if not address:
        raise ValueError("Ray has not published its address")
    host, port = address.rsplit(":", 1)
    if host != ip or not port.isdigit() or not 1 <= int(port) <= 65535:
        raise ValueError("Ray published an unexpected head address")
    if expected is not None and address != expected:
        raise ValueError("Worker joined a different Ray head")
    return address


def atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def wait_and_record(args):
    from ray._private.utils import read_ray_address
    import ray

    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        # This is the exact child PID started by this role, never a process scan.
        os.kill(args.supervisor_pid, 0)
        address = read_ray_address()
        if address:
            address = validate_address(address, args.head_ip, args.expected_address)
            record = {"ray_version": ray.__version__, "job_id": args.job_id,
                      "rank": args.rank, "gcs_address": address,
                      "supervisor_pid": args.supervisor_pid,
                      "client_port": args.client_port,
                      "internal_and_worker_ports": "OS-selected (port=0)"}
            atomic(args.run / ("ray-port-rank-%s.json" % args.rank), json.dumps(record, indent=2) + "\n")
            if args.rank == 0:
                atomic(args.run / "RAY_ADDRESS", address + "\n")
            print("[RAY_DYNAMIC_PORTS] " + json.dumps(record), flush=True)
            return
        time.sleep(0.5)
    raise TimeoutError("Ray supervisor did not publish a ready local address")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    check = actions.add_parser("check-launcher")
    check.add_argument("script", type=Path)
    check.add_argument("--receipt", type=Path, required=True)
    client = actions.add_parser("client")
    client.add_argument("--ip", required=True)
    ready = actions.add_parser("ready")
    ready.add_argument("--run", type=Path, required=True)
    ready.add_argument("--job-id", required=True)
    ready.add_argument("--rank", type=int, required=True)
    ready.add_argument("--head-ip", required=True)
    ready.add_argument("--supervisor-pid", type=int, required=True)
    ready.add_argument("--client-port", type=int)
    ready.add_argument("--expected-address")
    ready.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    if args.action == "check-launcher":
        receipt = validate_launcher(args.script.read_text())
        receipt.update(schema="nemotronvc.ray_launcher_contract.v1",
                       ray_role_sha256=hashlib.sha256(args.script.read_bytes()).hexdigest(),
                       ray_ports_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        atomic(args.receipt, json.dumps(receipt, indent=2) + "\n")
        print("[RAY_LAUNCHER_CONTRACT_PASS] " + str(args.script), flush=True)
    elif args.action == "client":
        print(choose_client_port(args.ip))
    else:
        wait_and_record(args)


if __name__ == "__main__":
    main()
