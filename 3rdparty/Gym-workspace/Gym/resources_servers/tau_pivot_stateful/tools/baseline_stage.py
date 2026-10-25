# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Set up a fresh baseline103d server over identical v2 cache entries and inputs."""

import argparse
import copy
import hashlib
import json
import subprocess
from pathlib import Path


p = argparse.ArgumentParser()
p.add_argument("--stage", type=Path, required=True)
p.add_argument("--bundle", type=Path, required=True)
a = p.parse_args()
stage = a.stage
bundle = a.bundle
meta = json.loads((stage / "STAGED.json").read_text())
baseline = stage / "baseline"
baseline.mkdir()
code = stage / "baseline-code"
code.mkdir()
pin = json.loads((bundle / "verifier/v2-baseline/CODE.json").read_text())
assert pin["commit"] == "103d56002d3d2d000923a6a128da442454c79f93"
archive = bundle / "verifier/v2-baseline/CODE.tar.gz"
assert hashlib.sha256(archive.read_bytes()).hexdigest() == pin["archive_sha256"]
subprocess.run(["tar", "-xzf", str(archive), "-C", str(code)], check=True)
for name, sha in pin["files"].items():
    assert hashlib.sha256((code / name).read_bytes()).hexdigest() == sha
v2 = {
    json.loads(line)["record_id"] for line in (bundle / "MANIFEST.jsonl").open() if json.loads(line)["version"] == "v2"
}
cache = json.loads((stage / "CACHE_MANIFEST.json").read_text())
cache["records"] = {k: v for k, v in cache["records"].items() if k in v2}
(baseline / "CACHE_MANIFEST.json").write_text(json.dumps(cache))
old = str(stage / "3rdparty/Gym-workspace/Gym")
new = str(code / "3rdparty/Gym-workspace/Gym")
runtimes = json.loads((stage / "RUNTIME.json").read_text())
for r in runtimes.values():
    r["command"] = [x.replace(old, new) for x in r["command"]]
    r["env"] = {k: v.replace(old, new) for k, v in r["env"].items()}
(baseline / "RUNTIME.json").write_text(json.dumps(runtimes))
services = json.loads((stage / "services.json").read_text())
conf = services["tau_pivot_stateful"]["resources_servers"]["tau_pivot_stateful"]
conf["cache_manifest"] = str(baseline / "CACHE_MANIFEST.json")
conf["runtime_manifest"] = str(baseline / "RUNTIME.json")
(baseline / "services.json").write_text(json.dumps(services))
result = copy.deepcopy(meta)
result.update(
    code_root=str(code),
    code_files=pin["files"],
    code_commit=pin["commit"],
    frontend_env={k: v.replace(old, new) for k, v in meta["frontend_env"].items()},
    records=len(cache["records"]),
    services_sha256=hashlib.sha256((baseline / "services.json").read_bytes()).hexdigest(),
)
(baseline / "STAGED.json").write_text(json.dumps(result))
print(json.dumps(dict(baseline_records=result["records"], stage=str(baseline))), flush=True)
