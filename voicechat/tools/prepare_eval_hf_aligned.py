#!/usr/bin/env python3
"""CPU-only evaluation view: native HF weights plus authoritative base metadata.

Does not edit the native export, load weights, weaken template checks, or claim a
generation canary. The evaluator registers the resulting unregistered HF view.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(path):
    s = path.stat()
    return {"device": s.st_dev, "inode": s.st_ino, "size": s.st_size,
            "mtime_ns": s.st_mtime_ns}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--text-source", type=Path, required=True)
    p.add_argument("--step", type=int, required=True)
    a = p.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Run in a CPU Slurm allocation")
    source, base, output, tokenizer, text_source = (
        path.resolve() for path in (a.source, a.base, a.output, a.tokenizer, a.text_source)
    )
    if output.exists() or source == output or source in output.parents:
        raise ValueError("Output must be a new independent directory")
    complete = json.loads((source / "EXPORT_COMPLETE.json").read_text())
    if complete.get("step") != a.step:
        raise ValueError("Native export completion step differs")
    if Path(complete["source_model"]).resolve() != base:
        raise ValueError("Metadata base differs from the native export's source model")
    index = json.loads((source / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    if any(Path(n).is_absolute() or ".." in Path(n).parts for n in shards):
        raise ValueError("Unsafe indexed shard")
    if set(shards) != {f.name for f in source.glob("*.safetensors")}:
        raise ValueError("Unindexed or missing weight shards")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.with_name(output.name + ".partial." + os.environ["SLURM_JOB_ID"])
    stage.mkdir(exist_ok=False)
    files, overrides = [], []
    metadata_names = ("tokenizer_config.json", "tokenizer.json", "generation_config.json",
                      "special_tokens_map.json", "chat_template.jinja")
    try:
        for src in sorted(source.iterdir()):
            if src.is_symlink() or not src.is_file():
                raise ValueError("Expected regular top-level HF file: " + str(src))
            if src.name in {"EXPORT_RECEIPT.json", "SNAPSHOT_RECEIPT.json", "HF_SNAPSHOT.json"}:
                continue
            dst = stage / src.name
            before = identity(src)
            if src.name in shards or src.name == "model.safetensors.index.json":
                os.link(src, dst, follow_symlinks=False)
                if before != identity(src) or before != identity(dst):
                    raise ValueError("Weight/index identity changed")
                files.append({"path": src.name, "storage": "hardlink", "source_identity": before,
                              "view_identity": identity(dst)})
            else:
                if before["size"] > 64 * 1024 * 1024:
                    raise ValueError("Unexpected large metadata file")
                original = sha(src)
                shutil.copy2(src, dst)
                if original != sha(dst) or before != identity(src):
                    raise ValueError("Metadata changed while copying")
                files.append({"path": src.name, "storage": "copied_metadata",
                              "source_identity": before, "native_sha256": original})
        for name in metadata_names:
            authoritative = base / name
            if not authoritative.is_file():
                if name != "special_tokens_map.json":
                    raise ValueError("Missing authoritative metadata: " + name)
                continue
            before = identity(authoritative)
            dst = stage / name
            native_sha = sha(dst) if dst.exists() else None
            expected = sha(authoritative)
            shutil.copy2(authoritative, dst)
            if expected != sha(dst) or before != identity(authoritative):
                raise ValueError("Authoritative metadata changed during copy")
            comparison = None
            if name.endswith(".json") and (source / name).is_file():
                comparison = json.loads((source / name).read_text()) == json.loads(authoritative.read_text())
            overrides.append({"path": name, "source": str(authoritative), "native_sha256": native_sha,
                              "aligned_sha256": expected, "json_semantics_identical": comparison})
        if sha(stage / "config.json") != sha(source / "config.json"):
            raise ValueError("Model architecture config must remain unchanged")
        audit_dir = stage / "EVAL_PREFLIGHT"
        audit_dir.mkdir()
        for kind in ("full", "repeats"):
            cmd = [sys.executable, str(text_source / "runtime/audit_request_contract.py"),
                   "--manifest", str(text_source / "manifests" / kind / "manifest.jsonl"),
                   "--model-path", str(stage), "--tokenizer", str(tokenizer),
                   "--contracts", str(text_source / "config/model-template-contracts.json"),
                   "--fixtures", str(text_source / "config/model-template-fixtures.json"),
                   "--max-model-len", "163840", "--output", str(audit_dir / (kind + ".json"))]
            subprocess.run(cmd, check=True)
        for item in files:
            if item["source_identity"] != identity(source / item["path"]):
                raise ValueError("Native source changed before publication")
        provenance = {
            "schema_version": 1, "kind": "eval_only_metadata_reconciliation", "status": "cpu_validated",
            "source_hf": str(source), "source_step": a.step, "output_hf": str(output),
            "metadata_base_hf": str(base), "created_utc": datetime.now(timezone.utc).isoformat(),
            "slurm_job_id": os.environ["SLURM_JOB_ID"], "node": os.environ.get("SLURMD_NODENAME"),
            "weight_map_tensor_count": len(index["weight_map"]), "weight_shard_count": len(shards),
            "source_completion_sha256": sha(source / "EXPORT_COMPLETE.json"),
            "unchanged_index_sha256": sha(source / "model.safetensors.index.json"),
            "unchanged_config_sha256": sha(source / "config.json"), "files": files,
            "metadata_overrides": overrides, "tokenizer_path": str(tokenizer),
            "audited_text_source": str(text_source),
            "auditor_sha256": sha(text_source / "runtime/audit_request_contract.py"),
            "contracts_sha256": sha(text_source / "config/model-template-contracts.json"),
            "fixtures_sha256": sha(text_source / "config/model-template-fixtures.json"),
            "preflight": {kind: {"sha256": sha(audit_dir / (kind + ".json")),
                                 "datasets": json.loads((audit_dir / (kind + ".json")).read_text())["datasets"]}
                          for kind in ("full", "repeats")},
            "validation": {"metadata": True, "template_fixtures": True, "all_prompt_budgets": True,
                           "model_weight_loading": False, "server_parity": False, "generation": False},
            "notes": ["Same metadata reconciliation as warmup runbook change #14",
                      "Original native HF and other evaluation views remain unchanged except weight hardlink counts",
                      "No EXPORT_RECEIPT or SNAPSHOT_RECEIPT is claimed; normal eval coordinator will register this view"],
        }
        (stage / "EVAL_RECONCILIATION.json").write_text(json.dumps(provenance, indent=2) + "\n")
        if output.exists():
            raise ValueError("Output appeared before atomic publication")
        os.rename(stage, output)
        print(json.dumps({"status": "EVAL_HF_CPU_VALIDATED", "output": str(output),
                          "job_id": os.environ["SLURM_JOB_ID"], "step": a.step,
                          "preflight": provenance["preflight"]}, indent=2))
    except BaseException:
        # Preserve the exact failed staging directory for diagnosis; never touch source.
        print("Failed staging directory retained: " + str(stage), file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
