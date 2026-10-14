# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Opt-in numerical checks of restored HybridDeviceOptimizer state."""

import math
import os
from typing import Any

import torch


def _hdo_optimizers(optimizer):
    from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import (
        HybridDeviceOptimizer,
    )

    for wrapped in getattr(optimizer, "chained_optimizers", [optimizer]):
        inner = getattr(wrapped, "optimizer", None)
        if isinstance(inner, HybridDeviceOptimizer):
            yield inner


def _step(value: Any, location: str) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise RuntimeError(f"HDO resume: nonscalar step at {location}")
        value = value.item()
    if (
        value is None
        or not math.isfinite(float(value))
        or int(value) != value
        or value <= 0
    ):
        raise RuntimeError(f"HDO resume: invalid restored step {value!r} at {location}")
    return int(value)


def capture_hdo_resume_steps(optimizer) -> tuple[tuple[int | None, ...], ...]:
    """Capture checkpoint group metadata before any resumed optimizer update."""
    result = tuple(
        tuple(
            _step(group.get("step"), f"optimizer {oi}, group {gi}")
            if group["params"]
            else None
            for gi, group in enumerate(hdo.param_groups)
        )
        for oi, hdo in enumerate(_hdo_optimizers(optimizer))
    )
    if not result:
        raise RuntimeError(
            "HDO resume verification requested but no HDO optimizer was restored"
        )
    return result


def verify_hdo_resume(optimizer, expected_steps, *, rank: int) -> dict[str, Any]:
    """Check every restored master/moment without creating model-sized copies.

    The group steps come from distributed-checkpoint optimizer metadata, not
    the GRPO iteration (which need not equal the number of optimizer updates).
    All tensor comparisons stay on the existing device and use exact equality.
    """
    hdos = list(_hdo_optimizers(optimizer))
    if len(hdos) != len(expected_steps):
        raise RuntimeError("HDO resume: optimizer topology changed before first update")
    parameter_count = moment_count = element_count = 0
    seen_steps = set()
    for oi, (hdo, group_steps) in enumerate(zip(hdos, expected_steps)):
        if len(hdo.param_groups) != len(group_steps):
            raise RuntimeError(
                "HDO resume: optimizer groups changed before first update"
            )
        owners = {
            id(param): (suboptimizer, group)
            for suboptimizer in hdo.sub_optimizers
            for group in suboptimizer.param_groups
            for param in group["params"]
        }
        for gi, (group, expected_step) in enumerate(zip(hdo.param_groups, group_steps)):
            if not group["params"]:
                continue
            location = f"optimizer {oi}, group {gi}"
            group_step = _step(group.get("step"), location)
            if group_step != expected_step:
                raise RuntimeError(
                    f"HDO resume: group step changed at {location}: "
                    f"actual={group_step}, expected={expected_step}"
                )
            seen_steps.add(expected_step)
            for pi, original in enumerate(group["params"]):
                location = f"optimizer {oi}, group {gi}, parameter {pi}"
                inner = hdo.param_to_inner_param[original]
                suboptimizer, inner_group = owners[id(inner)]
                suboptimizer_class = (
                    f"{type(suboptimizer).__module__}.{type(suboptimizer).__qualname__}"
                )
                restored = hdo.state[original]
                actual = suboptimizer.state[inner]
                master = restored.get("master_param")
                if not isinstance(master, torch.Tensor):
                    raise RuntimeError(f"HDO resume: missing FP32 master at {location}")
                if (
                    inner.dtype != torch.float32
                    or master.dtype != torch.float32
                    or inner.device != master.device
                    or inner.shape != master.shape
                    or not torch.equal(inner.detach(), master.detach())
                ):
                    raise RuntimeError(
                        f"HDO resume: FP32 master mismatch at {location}"
                    )
                parameter_count += 1
                element_count += inner.numel()
                for key in ("exp_avg", "exp_avg_sq"):
                    saved_moment = restored.get(key)
                    actual_moment = actual.get(key)
                    if not isinstance(saved_moment, torch.Tensor) or not isinstance(
                        actual_moment, torch.Tensor
                    ):
                        raise RuntimeError(f"HDO resume: missing {key} at {location}")
                    if (
                        saved_moment.dtype != torch.float32
                        or actual_moment.dtype != torch.float32
                        or saved_moment.device != inner.device
                        or actual_moment.device != inner.device
                        or saved_moment.shape != inner.shape
                        or actual_moment.shape != inner.shape
                        or (
                            saved_moment is not actual_moment
                            and not torch.equal(saved_moment, actual_moment)
                        )
                    ):
                        raise RuntimeError(f"HDO resume: {key} mismatch at {location}")
                    moment_count += 1
                inner_group_step = _step(inner_group.get("step"), location)
                if inner_group_step != expected_step:
                    raise RuntimeError(
                        f"HDO resume: suboptimizer group step mismatch at {location}: "
                        f"actual={inner_group_step}, expected={expected_step}, "
                        f"suboptimizer={suboptimizer_class}"
                    )
                # Torch Adam stores a per-parameter step. TE FusedAdam uses its
                # group step; DCP/HDO also carries the restored scalar in state.
                for label, state in (("restored", restored), ("actual", actual)):
                    parameter_step = _step(state.get("step"), f"{location}, {label}")
                    if parameter_step != expected_step:
                        raise RuntimeError(
                            f"HDO resume: {label} parameter step mismatch at {location}: "
                            f"actual={parameter_step}, expected={expected_step}, "
                            f"suboptimizer={suboptimizer_class}"
                        )
    if parameter_count == 0:
        raise RuntimeError("HDO resume: no restored parameters were checked")
    return {
        "passed": True,
        "rank": rank,
        "hdo_count": len(hdos),
        "parameter_count": parameter_count,
        "parameter_elements": element_count,
        "moment_count": moment_count,
        "optimizer_steps": sorted(seen_steps),
        "master_exact_equal": True,
        "moments_exact_equal": True,
        "max_master_error": 0.0,
    }


def verify_master_matches_model(optimizer) -> dict[str, Any]:
    """Cross-check restored FP32 masters against the bf16 model shards they own.

    With the precision-aware DistributedOptimizer the HDO ``param_groups`` hold
    bf16 *views* of each model parameter's DP-owned range, and the restored FP32
    master lives in ``hdo.state[view]["master_param"]``. Every optimizer step
    rounds the master (RNE) into that view and the views are all-gathered, so at
    save time ``bf16(master) == view`` holds on every rank. Model weights are
    DP-replicated while the optimizer slices are DP-resharded independently, so
    a mis-sliced reshard (e.g. after changing the learner node count) shows up
    here as a mismatch even though the HDO-vs-suboptimizer checks, which read the
    same DCP load, stay green. Native FP32 parameters (the view *is* the master)
    are skipped. Non-fatal unless ``NVC_HDO_MASTER_MODEL_STRICT=1``; the counts
    go into the resume receipt either way.
    """
    checked = skipped = mismatched_params = mismatched_elements = 0
    max_abs_diff = 0.0
    for hdo in _hdo_optimizers(optimizer):
        for group in hdo.param_groups:
            for original in group["params"]:
                master = hdo.state.get(original, {}).get("master_param")
                if original.dtype == torch.float32 or not isinstance(master, torch.Tensor):
                    skipped += 1
                    continue
                view = original.detach()
                checked += 1
                if master.shape != view.shape:
                    mismatched_params += 1
                    mismatched_elements += view.numel()
                    continue
                expected = master.detach().to(device=view.device, dtype=view.dtype)
                if not torch.equal(view, expected):
                    mismatched_params += 1
                    mismatched_elements += int((view != expected).sum().item())
                    max_abs_diff = max(
                        max_abs_diff,
                        float((view.float() - expected.float()).abs().max().item()),
                    )
                del expected
    strict = os.environ.get("NVC_HDO_MASTER_MODEL_STRICT") == "1"
    result = {
        "master_model_checked_params": checked,
        "master_model_skipped_params": skipped,
        "master_model_mismatch_params": mismatched_params,
        "master_model_mismatch_elements": mismatched_elements,
        "master_model_max_abs_diff": max_abs_diff,
        "master_model_exact_equal": (mismatched_params == 0) if checked else None,
        "master_model_strict": strict,
    }
    if strict and mismatched_params:
        raise RuntimeError(
            "HDO resume: bf16(master) != model shard for "
            f"{mismatched_params} parameters ({mismatched_elements} elements, "
            f"max |diff| {max_abs_diff:.3e}) — optimizer slices do not match the "
            "model weights they own"
        )
    return result
