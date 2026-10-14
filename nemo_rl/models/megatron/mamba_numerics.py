"""Mamba numerics controls for Megatron policy workers (RL audit 2026-10-13, analysis/rl-mamba-audit-20261013).

1. FP32 training SSM states. ``megatron_cfg.mamba_training_ssm_states_dtype`` was never forwarded to the model config,
   so every MambaMixer silently used ``params_dtype`` (BF16) states. ``apply_state_dtype`` forwards it;
   ``check_state_dtype`` asserts every MambaMixer of the built model uses it.
2. Rank-consistent Mamba kernels. mamba_ssm's SSD kernels are ``@triton.autotune``d per process (23 autotuners), so
   ranks pick different configs and return different logprobs/gradients for identical inputs. The fix is mamba_ssm's own
   deterministic mode, enabled by ``megatron_cfg.env_vars.MAMBA_DETERMINISTIC: "1"`` (set before mamba_ssm is imported):
   one config per kernel at import and zero-initialised tile workspaces instead of atomics. Probe on 8 A100s
   (receipts/rl-mamba-audit-20261013/det-v1): forward and gradients bitwise identical across GPUs, gradients within
   1.2e-4 of free autotuning, no slowdown. Do NOT pin configs by editing Autotuner.configs: several workspaces are sized
   at import from the smallest BLOCK_SIZE_N of all configs, and pinning a larger block leaves uninitialised tiles
   (gradients off by 1e8, receipts/rl-mamba-audit-20261013/pin-v1).
``report_after_model`` checks both and prints one ``[MAMBA_NUMERICS]`` line per rank.
"""
from __future__ import annotations

import importlib
import json
import os

SSD_MODULES = (
    "mamba_ssm.ops.triton.ssd_bmm",
    "mamba_ssm.ops.triton.ssd_chunk_scan",
    "mamba_ssm.ops.triton.ssd_chunk_state",
    "mamba_ssm.ops.triton.ssd_state_passing",
    "mamba_ssm.ops.triton.ssd_combined",
)


def _to_dtype(value):
    import torch

    if value is None or isinstance(value, torch.dtype):
        return value
    aliases = {"float32": torch.float32, "fp32": torch.float32, "float": torch.float32,
               "bfloat16": torch.bfloat16, "bf16": torch.bfloat16, "float16": torch.float16, "fp16": torch.float16}
    key = str(value).lower().replace("torch.", "")
    if key not in aliases:
        raise ValueError(f"unsupported mamba_training_ssm_states_dtype {value!r}")
    return aliases[key]


def apply_state_dtype(model_cfg, megatron_cfg) -> None:
    """Forward megatron_cfg.mamba_training_ssm_states_dtype to the Megatron model config (was dropped -> BF16)."""
    dtype = _to_dtype(megatron_cfg.get("mamba_training_ssm_states_dtype"))
    if dtype is not None:
        if not hasattr(model_cfg, "mamba_training_ssm_states_dtype"):
            raise ValueError(f"{type(model_cfg).__name__} has no mamba_training_ssm_states_dtype field")
        model_cfg.mamba_training_ssm_states_dtype = dtype


def iter_autotuners():
    from triton.runtime.autotuner import Autotuner

    for module_name in SSD_MODULES:
        module = importlib.import_module(module_name)
        for name, obj in sorted(vars(module).items()):
            if isinstance(obj, Autotuner):
                yield f"{module_name}.{name}", obj


def check_determinism(megatron_cfg) -> dict:
    """If the recipe asks for MAMBA_DETERMINISTIC=1, verify it is active and every SSD kernel has one config."""
    wanted = str((megatron_cfg.get("env_vars") or {}).get("MAMBA_DETERMINISTIC", "")).startswith("1")
    from mamba_ssm.utils.determinism import use_deterministic_mode

    active = bool(use_deterministic_mode())
    tuners = list(iter_autotuners())
    several = [name for name, tuner in tuners if len(tuner.configs) > 1]
    if wanted and (not active or several or os.environ.get("TRITON_CACHE_AUTOTUNING") == "1"):
        raise RuntimeError(f"MAMBA_DETERMINISTIC requested but active={active}, multi-config kernels={several}, "
                           f"TRITON_CACHE_AUTOTUNING={os.environ.get('TRITON_CACHE_AUTOTUNING')}")
    return {"deterministic_requested": wanted, "deterministic_active": active, "ssd_autotuners": len(tuners),
            "multi_config_autotuners": len(several)}


def check_state_dtype(model, megatron_cfg) -> dict:
    """Assert every MambaMixer uses the requested state dtype; return a summary for the log line."""
    want = _to_dtype(megatron_cfg.get("mamba_training_ssm_states_dtype"))
    seen = {}
    modules = model if isinstance(model, (list, tuple)) else [model]
    for chunk in modules:
        for module in chunk.modules():
            if type(module).__name__ == "MambaMixer":
                got = getattr(module, "mamba_training_ssm_states_dtype", None)
                seen[str(got)] = seen.get(str(got), 0) + 1
                if want is not None and got != want:
                    raise RuntimeError(f"MambaMixer state dtype {got} != requested {want}")
    return {"mamba_mixers": sum(seen.values()), "state_dtypes": seen, "requested": str(want)}


def report_after_model(model, megatron_cfg) -> None:
    import torch

    summary = {**check_state_dtype(model, megatron_cfg), **check_determinism(megatron_cfg)}
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    print("[MAMBA_NUMERICS] " + json.dumps({"rank": rank, **summary}), flush=True)
