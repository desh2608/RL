# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage the shipped pinned code, runtime(s) and snapshots on local disk; never rebuild."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def write(p, x):
    Path(p).write_text(json.dumps(x, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--stage", type=Path, required=True)
    p.add_argument("--runtime-key", action="append")
    p.add_argument("--min-free-gib", type=int)
    a = p.parse_args()
    b = a.bundle.resolve()
    s = a.stage.resolve()
    assert not str(s).startswith("/lustre/"), "stage must be node-local"
    s.mkdir(parents=True, exist_ok=False)
    plan = json.loads((b / "verifier/STAGING.json").read_text())
    keys = set(a.runtime_key or plan["runtime_keys"])
    assert keys and keys <= set(plan["runtime_keys"])
    minimum = a.min_free_gib or (16 if a.runtime_key else 160)
    assert shutil.disk_usage(s).free >= minimum * 1024**3, "Insufficient node-local free disk"
    assert sha(b / "verifier/CODE.tar.gz") == plan["code_sha256"]
    subprocess.run(["tar", "-xzf", str(b / "verifier/CODE.tar.gz"), "-C", str(s)], check=True)
    code = s / "3rdparty/Gym-workspace/Gym"
    resource = code / "resources_servers/tau_pivot_stateful"
    pin = json.loads((b / "verifier/CODE.json").read_text())
    for name, h in pin["files"].items():
        assert sha(s / name) == h, ("code pin mismatch", name)
    cache = s / "cache"
    (cache / "entries").mkdir(parents=True)
    (cache / "payloads").mkdir()
    index = {}
    source = json.loads((b / "verifier/CACHE_MANIFEST.json").read_text())["records"]
    for ident, item in source.items():
        if item["runtime_key"] not in keys:
            continue
        ep = b / "verifier" / item["entry"]
        assert sha(ep) == item["entry_sha256"]
        entry = json.loads(ep.read_text())
        target = cache / "entries" / ep.name
        shutil.copyfile(ep, target)
        if item.get("mode") == "combined_static":
            field = entry.get("field_table")
            if field:
                payload = b / "verifier/cache" / field["path"]
                dest = cache / field["path"]
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    assert sha(payload) == field["sha256"]
                    shutil.copyfile(payload, dest)
        else:
            payload = b / "verifier/cache" / entry["payload"]
            dest = cache / entry["payload"]
            if not dest.exists():
                assert sha(payload) == entry["payload_sha256"]
                shutil.copyfile(payload, dest)
        index[ident] = {**item, "entry": str(target)}
    write(s / "CACHE_MANIFEST.json", dict(records=index))
    runtimes = {}
    front_env = None
    front_python = None
    receipts = []
    selected_archives = [x for x in plan["archives"] if x["runtime_key"] in keys]
    if not selected_archives:
        selected_archives = [plan["archives"][0]]
    for archive in selected_archives:
        key = archive["runtime_key"]
        src = b / "verifier" / archive["bundle_path"]
        assert src.stat().st_size == archive["bytes"] and sha(src) == archive["sha256"]
        dest = s / "runtimes" / key
        dest.mkdir(parents=True)
        subprocess.run(
            ["tar", "-xzf", str(src), "-C", str(dest)]
            if archive["format"] == "runtime-gzip"
            else ["tar", "--zstd", "-xf", str(src), "-C", str(dest)],
            check=True,
        )
        native = dest / "native" if archive["format"] == "runtime-gzip" else dest
        assets = dest / "assets"
        python = native / "python/bin/python3.13"
        assert python.exists()
        paths = [
            code,
            resource / "reference",
            native / "code",
            native / "tau2/src",
            assets / "deps",
            native / "site-packages",
            native / "site-packages/tau-company",
            native / "site-packages/gym",
            native / "tau-company-lab/src",
        ]
        env = dict(
            PYTHONHOME=str(native / "python"),
            LD_LIBRARY_PATH=str(native / "python/lib"),
            PYTHONPATH=":".join(map(str, paths)),
            TAU2_DATA_DIR=str(native / "tau2/data"),
            TMPDIR=str(s),
            PYTHONDONTWRITEBYTECODE="1",
            HF_HUB_OFFLINE="1",
            LITELLM_LOCAL_MODEL_COST_MAP="True",
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
        )
        runtimes[key] = dict(
            command=[
                str(python),
                str(resource / "native_runtime.py"),
                "--mode",
                "serve",
                "--frozen",
                str(native),
                "--cache",
                str(cache),
                "--cache-entries",
                "1",
            ],
            env=env,
            replicas=1,
        )
        if front_env is None:
            front_env = env
            front_python = str(python)
        receipts.append(dict(runtime_key=key, archive_sha256=archive["sha256"], compressed_bytes=archive["bytes"]))
    write(s / "RUNTIME.json", runtimes)
    template = (b / "verifier/services.yaml").read_text()
    config = template.replace("${pivot_stage}", str(s))
    (s / "services.json").write_text(config)
    (s / "tools").mkdir()
    shutil.copyfile(b / "verifier/check_controls.py", s / "tools/check_controls.py")
    write(
        s / "STAGED.json",
        dict(
            bundle=str(b),
            keys=sorted(keys),
            records=len(index),
            code_commit=pin["commit"],
            code_files=pin["files"],
            code_root=str(s),
            frontend_python=front_python,
            frontend_env=front_env,
            checker=str(s / "tools/check_controls.py"),
            checker_sha256=sha(s / "tools/check_controls.py"),
            services_sha256=sha(s / "services.json"),
            archives=receipts,
            cache_rebuilt=False,
            native_prefix_replays=0,
            model_calls=0,
        ),
    )
    print(
        json.dumps(dict(staged_records=len(index), runtime_keys=sorted(keys), code_commit=pin["commit"])), flush=True
    )


if __name__ == "__main__":
    main()
