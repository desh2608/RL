"""Bounded observational diagnostics for the reviewed CPU-advantage K=8 arm.

Callers pass next-token columns [:, 1:] as consumed by ClippedPGLossFn.
The counts precede loss-level importance sampling. Existing learner IS metrics
remain authoritative; this module never recomputes or changes IS weights.
"""
from __future__ import annotations

import json
import math
import os
from collections import defaultdict

import torch

ENV_FLAG = "NVC_POSTMASK_K8_METRICS"
MAX_JSON_BYTES = 16 * 1024
MAX_ROWS = 2048
MAX_TOKENS = 98304
ROUTES = frozenset({
    "code_gen_simple_agent", "instruction_following_simple_agent",
    "math_with_judge_simple_agent",
    "search_pivot_single_step_tool_use_with_argument_comparison_agent",
    "single_step_tool_use_with_argument_comparison_agent",
    "structured_outputs_simple_agent",
    "toolcall_schema_single_step_tool_use_with_argument_comparison_agent",
})


def postmask_k8_metrics_enabled() -> bool:
    """Default off; the frozen runtime must explicitly export the value 1."""
    return os.environ.get(ENV_FLAG, "0") == "1"


def _check(condition, reason):
    if not condition:
        raise ValueError(reason)


def _fraction(numerator, denominator):
    return numerator / denominator if denominator else None


@torch.no_grad()
def collect_postmask_k8_metrics(*, rewards, advantages, token_mask, sample_mask,
                               group_ids, domains, step, row_chunk=64,
                               token_chunk=4096):
    """Return scalars only, without modifying inputs or allocating a full mask.

    Null denotes an undefined denominator, never an assumed zero. Reward
    dispersion is population std within retained K8 groups, then group-mean.
    Nonfinite rewards exclude a group from dispersion and are counted explicitly.
    Nonzero advantage fractions use finite valid tokens, with nonfinite counts.
    """
    tensors = (rewards, advantages, token_mask, sample_mask, group_ids)
    _check(all(isinstance(t, torch.Tensor) for t in tensors), "tensor_type")
    _check(all(t.device.type == "cpu" for t in tensors), "cpu_advantages_required")
    _check(advantages.ndim == 2 and token_mask.shape == advantages.shape, "token_shape")
    rows, width = advantages.shape
    _check(0 < rows <= MAX_ROWS and rows % 8 == 0 and 0 < width <= MAX_TOKENS, "k8_shape")
    _check(rewards.shape == sample_mask.shape == (rows,), "row_shape")
    _check(group_ids.shape in ((rows,), (rows, 1)), "group_shape")
    _check(group_ids.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8), "group_dtype")
    _check(isinstance(domains, (list, tuple)) and len(domains) == rows, "domain_shape")
    _check(type(step) is int and step > 0, "step")
    _check(type(row_chunk) is int and 1 <= row_chunk <= 64 and
           type(token_chunk) is int and 1 <= token_chunk <= 4096, "chunk_bound")
    _check(bool(torch.isfinite(sample_mask).all()), "nonfinite_sample_mask")
    groups = defaultdict(list)
    for index, group in enumerate(group_ids.reshape(-1).tolist()):
        groups[group].append(index)
    _check(all(len(indices) == 8 for indices in groups.values()), "group_membership")
    route_names = [d if isinstance(d, str) and d in ROUTES else "unknown" for d in domains]
    # O(rows) counters; all token intermediates are <=64*4096 elements.
    # Columns: valid, nonfinite, finite-nonzero, finite-positive, finite-negative.
    counts = torch.zeros((rows, 5), dtype=torch.int64)
    for start in range(0, rows, row_chunk):
        stop = min(start + row_chunk, rows)
        retained = sample_mask[start:stop].detach().ne(0).unsqueeze(-1)
        for offset in range(0, width, token_chunk):
            tokens = token_mask[start:stop, offset:offset + token_chunk].detach()
            _check(bool(torch.isfinite(tokens).all()), "nonfinite_token_mask")
            active = tokens.ne(0) & retained
            values = advantages[start:stop, offset:offset + token_chunk].detach()
            finite = torch.isfinite(values)
            finite_active = active & finite
            counts[start:stop, 0] += active.sum(dim=1)
            counts[start:stop, 1] += (active & ~finite).sum(dim=1)
            counts[start:stop, 2] += (finite_active & values.ne(0)).sum(dim=1)
            counts[start:stop, 3] += (finite_active & values.gt(0)).sum(dim=1)
            counts[start:stop, 4] += (finite_active & values.lt(0)).sum(dim=1)
    row_counts = counts.tolist()
    reward_values = rewards.detach().tolist()

    def summarize(indices, group_members):
        total = [sum(row_counts[i][column] for i in indices) for column in range(5)]
        histogram = [0] * 9
        eligible = mixed = nonfinite_reward_groups = 0
        dispersions = []
        for members in group_members:
            retained = [i for i in members if row_counts[i][0] > 0]
            histogram[len(retained)] += 1
            values = [reward_values[i] for i in retained]
            if not all(math.isfinite(value) for value in values):
                nonfinite_reward_groups += 1
                continue
            if len(values) >= 2:
                eligible += 1
                mixed += int(min(values) != max(values))
                mean = math.fsum(values) / len(values)
                dispersions.append(math.sqrt(math.fsum((v - mean) ** 2 for v in values) / len(values)))
        finite_tokens = total[0] - total[1]
        return {
            "rows": len(indices), "groups": len(group_members),
            "survivors_0_to_8": histogram,
            "retained_responses": sum(row_counts[i][0] > 0 for i in indices),
            "dispersion_groups": eligible, "mixed_reward_groups": mixed,
            "mixed_fraction_of_dispersion_groups": _fraction(mixed, eligible),
            "mean_retained_group_reward_std": _fraction(math.fsum(dispersions), len(dispersions)),
            "nonfinite_reward_groups": nonfinite_reward_groups,
            "valid_tokens": total[0], "nonfinite_advantage_tokens": total[1],
            "finite_valid_tokens": finite_tokens, "nonzero_advantage_tokens": total[2],
            "positive_advantage_tokens": total[3], "negative_advantage_tokens": total[4],
            "nonzero_fraction_of_finite_valid_tokens": _fraction(total[2], finite_tokens),
        }

    by_route = defaultdict(list)
    for members in groups.values():
        names = {route_names[i] for i in members}
        route = next(iter(names)) if len(names) == 1 else "mixed_routes"
        by_route[route].append(members)
    return {
        "schema": "postmask-k8-signal.v1", "status": "ok", "step": step,
        "scope": "post_sample_and_next_token_masks_post_penalty_clip_pre_loss_IS",
        "unknown_route_rows": route_names.count("unknown"),
        "global": summarize(list(range(rows)), list(groups.values())),
        "routes": {route: summarize([i for group in members for i in group], members)
                   for route, members in sorted(by_route.items())},
    }


def postmask_k8_diagnostics(**kwargs):
    """Serialize at most 16KiB; bad diagnostic inputs are explicit, nonfatal.

    This wrapper neither consumes RNG nor mutates training state. Undefined
    metrics remain null in JSON and are omitted from numeric logger metrics.
    """
    try:
        record = collect_postmask_k8_metrics(**kwargs)
        payload = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
        _check(len(payload.encode("utf-8")) <= MAX_JSON_BYTES, "json_byte_budget")
    except (ValueError, TypeError, OverflowError) as error:
        # Validation strings are fixed code-owned reasons; never echo tensors,
        # route names, metadata, text, or the arbitrary exception message.
        reason = str(error)
        allowed = {"tensor_type", "cpu_advantages_required", "token_shape", "k8_shape",
                   "row_shape", "group_shape", "group_dtype", "domain_shape", "step",
                   "chunk_bound", "nonfinite_sample_mask", "nonfinite_token_mask",
                   "group_membership", "json_byte_budget"}
        record = {"schema": "postmask-k8-signal.v1", "status": "unavailable",
                  "reason": reason if reason in allowed else "nonfinite_or_invalid_diagnostic",
                  "step": kwargs.get("step") if type(kwargs.get("step")) is int else None}
        payload = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
    metrics = {"postmask_signal/available": int(record["status"] == "ok")}

    def flatten(value, prefix):
        if isinstance(value, dict):
            for key, child in value.items(): flatten(child, prefix + "/" + key)
        elif isinstance(value, list):
            for index, child in enumerate(value): flatten(child, prefix + "/" + str(index))
        elif type(value) in (int, float):
            metrics[prefix] = value
    if record["status"] == "ok":
        flatten(record["global"], "postmask_signal/global")
        flatten(record["routes"], "postmask_signal/routes")
        metrics["postmask_signal/unknown_route_rows"] = record["unknown_route_rows"]
    return metrics, payload
