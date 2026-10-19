# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail a process if its import path or loaded Python modules reference Lustre."""

import os
import sys


def _is_lustre(path):
    if not path:
        return False
    resolved = os.path.realpath(os.fspath(path))
    return resolved == "/lustre" or resolved.startswith("/lustre/")


def assert_no_lustre_imports():
    bad_paths = [path for path in sys.path if _is_lustre(path)]
    checked = 0
    bad_modules = []
    for name, module in tuple(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if path:
            checked += 1
            if _is_lustre(path):
                bad_modules.append((name, path))
        for namespace_path in getattr(module, "__path__", ()) or ():
            checked += 1
            if _is_lustre(namespace_path):
                bad_modules.append((name, namespace_path))
    if bad_paths or bad_modules:
        raise RuntimeError(f"Lustre import violation: sys.path={bad_paths!r}; modules={bad_modules[:20]!r}")
    print(f"lustre-import-guard: ok ({checked} modules checked)", flush=True)
    return checked
