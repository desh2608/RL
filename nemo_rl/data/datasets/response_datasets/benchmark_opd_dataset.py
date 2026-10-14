# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Portable loader for benchmark-oriented, independent-view text OPD data."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from nemo_rl.data.datasets.raw_dataset import RawDataset
from nemo_rl.data.datasets.response_datasets.oai_format_dataset import (
    PreservingDataset,
)


TASK_NAME = "benchmark_text_opd_v1"
REQUIRED_KEYS = {
    "source_id",
    "task_name",
    "extra_env_info",
    "student_view",
    "teacher_view",
    "teacher_alias",
}
VERIFIER_TYPES = {
    "math",
    "mcqa",
    "instruction_following",
    "exact",
    "teacher_only",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_view(path: Path, line_number: int, owner: str, view: Any) -> None:
    if not isinstance(view, dict):
        raise TypeError(f"{path}:{line_number} {owner} view is not an object")
    if view.get("modality") != "text":
        raise ValueError(f"{path}:{line_number} {owner} view is not text")
    if view.get("multimodal_payload") != {}:
        raise ValueError(f"{path}:{line_number} {owner} has multimodal payload")
    messages = view.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{path}:{line_number} {owner} messages are empty")
    for message in messages:
        if not isinstance(message, dict):
            raise TypeError(f"{path}:{line_number} {owner} message is not an object")
        if message.get("role") not in {"system", "user", "assistant", "tool"}:
            raise ValueError(
                f"{path}:{line_number} {owner} message has invalid role"
            )
        if not isinstance(message.get("content"), str):
            raise TypeError(
                f"{path}:{line_number} {owner} message content is not text"
            )
    if not isinstance(view.get("tools", []), list):
        raise TypeError(f"{path}:{line_number} {owner} tools is not a list")


def _validate_verifier(path: Path, line_number: int, metadata: dict[str, Any]) -> None:
    verifier = metadata.get("verifier")
    if not isinstance(verifier, dict) or set(verifier) != {
        "schema_version",
        "type",
        "payload",
    }:
        raise ValueError(
            f"{path}:{line_number} verifier must contain exactly "
            "schema_version, type, and payload"
        )
    if verifier["schema_version"] != 1 or isinstance(
        verifier["schema_version"], bool
    ):
        raise ValueError(f"{path}:{line_number} verifier schema_version must be 1")
    if verifier["type"] not in VERIFIER_TYPES:
        raise ValueError(
            f"{path}:{line_number} unsupported verifier type {verifier['type']!r}"
        )
    payload = verifier["payload"]
    if not isinstance(payload, dict):
        raise TypeError(f"{path}:{line_number} verifier payload is not an object")
    if verifier["type"] == "teacher_only" and (
        payload.get("outcome_enabled") is not False
        or payload.get("local_execution") is not False
    ):
        raise ValueError(
            f"{path}:{line_number} teacher_only verifier must explicitly disable "
            "outcome and local execution"
        )


class BenchmarkOPDDataset(RawDataset):
    """Load a sealed benchmark mix while binding teacher aliases in memory.

    Unlike ``TextCoreOPDDataset``, this loader does not encode a fixed set of
    domains or row counts.  The recipe owns the expected SHA and row count, and
    an optional mapping may route semantic buckets to different teachers.
    """

    def __init__(
        self,
        data_path: str,
        expected_data_sha256: str,
        expected_rows: int | str,
        max_samples: int | str,
        runtime_teacher_alias: str,
        runtime_teacher_alias_by_domain: dict[str, str] | None = None,
        subset: str | None = None,
        split: str | None = None,
        split_validation_size: float = 0,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        del subset, split, seed, kwargs
        if not math.isclose(float(split_validation_size), 0.0):
            raise ValueError("benchmark OPD data forbids implicit dataset splitting")
        if not runtime_teacher_alias:
            raise ValueError("runtime_teacher_alias must be non-empty")

        expected_rows_int = int(expected_rows)
        max_samples_int = int(max_samples)
        if expected_rows_int <= 0 or max_samples_int <= 0:
            raise ValueError("expected_rows and max_samples must be positive")
        if max_samples_int > expected_rows_int:
            raise ValueError("max_samples exceeds expected_rows")

        path = Path(data_path)
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"benchmark OPD data is missing or symlinked: {path}")
        actual_sha256 = _sha256(path)
        if actual_sha256 != expected_data_sha256:
            raise ValueError(
                "benchmark OPD data SHA mismatch: "
                f"expected={expected_data_sha256} actual={actual_sha256}"
            )

        rows: list[dict[str, Any]] = []
        source_ids: set[str] = set()
        domain_counts: Counter[str] = Counter()
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or set(row) != REQUIRED_KEYS:
                    raise ValueError(
                        f"{path}:{line_number} must contain exactly "
                        f"{sorted(REQUIRED_KEYS)}"
                    )
                source_id = row["source_id"]
                if not isinstance(source_id, str) or not source_id:
                    raise ValueError(f"{path}:{line_number} has no source_id")
                if source_id in source_ids:
                    raise ValueError(f"duplicate source_id: {source_id}")
                source_ids.add(source_id)

                metadata = row["extra_env_info"]
                if not isinstance(metadata, dict):
                    raise TypeError(f"{path}:{line_number} metadata is not an object")
                domain = metadata.get("domain")
                if not isinstance(domain, str) or not domain:
                    raise ValueError(f"{path}:{line_number} has no semantic domain")
                domain_counts[domain] += 1
                _validate_verifier(path, line_number, metadata)

                _validate_view(path, line_number, "student", row["student_view"])
                _validate_view(path, line_number, "teacher", row["teacher_view"])
                if row["student_view"].get("tools", []) != row["teacher_view"].get(
                    "tools", []
                ):
                    raise ValueError(f"{path}:{line_number} tool schemas diverged")

                alias = runtime_teacher_alias
                if runtime_teacher_alias_by_domain is not None:
                    if domain not in runtime_teacher_alias_by_domain:
                        raise ValueError(
                            f"no runtime teacher alias configured for domain {domain!r}"
                        )
                    alias = runtime_teacher_alias_by_domain[domain]
                if not isinstance(alias, str) or not alias:
                    raise ValueError(f"invalid runtime teacher alias for {domain!r}")
                row["teacher_alias"] = alias
                row["task_name"] = TASK_NAME
                rows.append(row)

        if len(rows) != expected_rows_int:
            raise ValueError(
                f"expected {expected_rows_int} rows, found {len(rows)}"
            )
        selected_rows = rows[:max_samples_int]
        print(
            "[BENCHMARK_OPD_DATASET_OK] "
            f"rows={len(selected_rows)} source_rows={len(rows)} "
            f"domains={dict(sorted(domain_counts.items()))} sha256={actual_sha256}",
            flush=True,
        )
        self.task_name = TASK_NAME
        self.dataset = PreservingDataset(selected_rows)
        self.val_dataset = None
