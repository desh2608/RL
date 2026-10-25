# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run staged verifier checks or the resources server using its pinned frontend Python."""

import argparse
import hashlib
import json
import os
from pathlib import Path


p = argparse.ArgumentParser()
p.add_argument("--stage", type=Path, required=True)
a, rest = p.parse_known_args()
meta = json.loads((a.stage / "STAGED.json").read_text())
assert hashlib.sha256(Path(meta["checker"]).read_bytes()).hexdigest() == meta["checker_sha256"]
env = dict(os.environ, **meta["frontend_env"])
cmd = [meta["frontend_python"], meta["checker"], "--bundle", meta["bundle"], "--stage", str(a.stage)] + rest
os.execve(cmd[0], cmd, env)
