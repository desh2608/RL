"""Explicit tail padding and metadata-only optimizer coverage for one full pass.

This module does not alter raw rows, responses, rewards, masks, or optimizer
state. It is disabled unless data.full_pass is present. The original processed
dataset is stateless; StatefulDataLoader retains its native sampler cursor.
"""

import copy
import json
import math
import numbers
import os
from pathlib import Path


CONTRACT = {
    "schema": "nvc_full_pass_v1",
    "original_rows": 8000,
    "virtual_rows": 8064,
    "padding_rows": 64,
    "padding_original_start": 0,
    "num_prompts_per_step": 128,
    "generations_per_prompt": 8,
    "resume_step": 24,
    "resume_frontier": 3072,
    "data_sha256": "b47a3d04b69e81a2c04dfdd3ea17cf966317081e61b00fa38299bb60adda3829",
    "prior_prefix_coverage": "cursor_and_order_inference_only",
}
STATE_NAME = "full_pass_coverage.json"


def _check(condition, message):
    if not condition:
        raise ValueError("full_pass: " + message)


def _integer(value, name):
    if hasattr(value, "item"):
        value = value.item()
    _check(isinstance(value, numbers.Integral) and not isinstance(value, bool), name)
    return int(value)


def _scalar(value):
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, numbers.Real) and math.isfinite(float(value)):
        return float(value)
    _check(False, "nonfinite or nonscalar metadata")


def _values(batch, key, count):
    values = batch.get(key)
    if values is None:
        return None
    if hasattr(values, "detach"):
        values = values.detach().cpu().tolist()
    else:
        values = list(values)
    _check(len(values) == count, "row count: " + key)
    return [_scalar(value) for value in values]


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + str(os.getpid()))
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def checked_contract(config):
    raw = config.data.get("full_pass")
    if raw is None:
        return None
    contract = dict(raw)
    _check(contract == CONTRACT, "unsupported explicit contract")
    data, grpo = config.data, config.grpo
    _check(data["shuffle"] is False, "shuffle must remain false")
    _check(data["use_multiple_dataloader"] is False, "single loader required")
    _check(data["num_workers"] == 1, "native single worker required")
    _check(grpo.max_num_epochs == 1, "one virtual epoch required")
    _check(grpo.num_prompts_per_step == 128, "P128 required")
    _check(grpo.num_generations_per_prompt == 8, "K8 required")
    _check(grpo.max_num_steps == 63, "absolute target63 required")
    _check(grpo.use_dynamic_sampling is False, "dynamic sampling unsupported")
    _check(grpo.batch_multiplier == 1, "batch multiplier must remain1")
    _check(config.policy["train_global_batch_size"] == 1024, "GBS1024 required")
    return contract


class FullPassDatasetView:
    """Stateless map view: [0,8000) followed by original [0,64).

    No generic attribute forwarding is used: in particular __getitems__ must
    never bypass the mapping, and no new state_dict/load_state_dict protocol
    may appear in a previously stateless processed dataset.
    """

    def __init__(self, dataset, contract):
        _check(len(dataset) == contract["original_rows"], "original dataset length")
        _check(not hasattr(dataset, "state_dict"), "stateful dataset unsupported")
        _check(not hasattr(dataset, "load_state_dict"), "stateful dataset unsupported")
        self.source = dataset
        self.contract = contract

    def __len__(self):
        return self.contract["virtual_rows"]

    def __getitem__(self, index):
        index = _integer(index, "virtual index must be integral")
        if not 0 <= index < len(self):
            raise IndexError(index)
        original = index if index < 8000 else index - 8000
        # Delegate after mapping, so the native processor emits original idx.
        datum = self.source[original]
        _check("idx" in datum, "processed datum omitted original idx")
        _check(_integer(datum["idx"], "original idx") == original, "original idx mismatch")
        return datum

    def __getitems__(self, indices):
        return [self[index] for index in indices]


def maybe_full_pass_dataset(dataset, config, suffix=""):
    contract = checked_contract(config)
    if contract is None:
        return dataset
    _check(suffix == "", "multiple loader suffix unsupported")
    return FullPassDatasetView(dataset, contract)


class FullPassLedger:
    """Record identities only after the existing optimizer success boundary.

    The pre24 prefix is deliberately not represented as observed identities.
    Subsequent records contain actual stream ordinals, optional emitted row
    identities, and native post-mask next-token counts. Zero-valid-token rows
    are retained as observed samples, never described as gradient contribution.
    """

    def __init__(self, contract, checkpoint, current_step, consumed, monitor_dir):
        self.contract = contract
        current_step = _integer(current_step, "current step")
        consumed = _integer(consumed, "consumed samples")
        _check(24 <= current_step <= 63, "resume step outside continuation")
        _check(consumed == current_step * 128, "consumed samples mismatch")
        _check(checkpoint is not None, "fresh start forbidden")
        path = Path(checkpoint) / STATE_NAME
        if path.is_file():
            with path.open(encoding="utf-8") as stream:
                self.state = json.load(stream)
            _check(self.state["schema"] == "nvc_full_pass_coverage_v1", "coverage schema")
            _check(self.state["contract"] == contract, "coverage contract changed")
            _check(self.state["last_completed_step"] == current_step, "coverage/checkpoint step")
            _check(len(self.state["updates"]) == current_step - 24, "coverage update count")
        else:
            _check(current_step == 24 and consumed == 3072, "missing later coverage state")
            self.state = {
                "schema": "nvc_full_pass_coverage_v1",
                "contract": contract,
                "origin_checkpoint": str(Path(checkpoint).resolve()),
                "prefix": {"positions": [0, 3072], "closed_updates": 24,
                           "evidence": "cursor_and_order_inference_only",
                           "per_sample_successful_update_ledger_available": False},
                "last_completed_step": 24,
                "updates": [],
            }
        self.monitor_path = Path(monitor_dir) / STATE_NAME
        self.seen = set()
        for expected_step, update in enumerate(self.state["updates"], 25):
            _check(update["step"] == expected_step, "nonconsecutive saved updates")
            _check(len(update["groups"]) == 128, "saved group count")
            for group in update["groups"]:
                ordinal = _integer(group["stream_ordinal"], "saved ordinal")
                _check(3072 <= ordinal < 8064 and ordinal not in self.seen, "saved duplicate/range")
                self.seen.add(ordinal)

    def prepare(self, trajectories, repeated_batch, train_data, step):
        _check(step == self.state["last_completed_step"] + 1, "nonconsecutive update")
        _check(len(trajectories) == 128 and repeated_batch.size == 1024, "complete P128K8 required")
        masks = _values(train_data, "sample_mask", 1024)
        _check(masks is not None, "sample mask absent")
        tokens = train_data["token_mask"][:, 1:].detach().sum(dim=-1).cpu().tolist()
        _check(len(tokens) == 1024, "next-token mask row count")
        original_ids = _values(repeated_batch, "idx", 1024)
        task_ids = _values(repeated_batch, "_ng_task_index", 1024)
        rollout_ids = _values(repeated_batch, "_ng_rollout_index", 1024)
        attempts = _values(repeated_batch, "_ng_attempt_index", 1024)
        groups, pending = [], set()
        for position, trajectory in enumerate(trajectories):
            ordinal = _integer(trajectory["_ng_task_index"], "actual group ordinal")
            _check(3072 <= ordinal < 8064, "sampled ordinal outside continuation")
            _check(ordinal not in pending and ordinal not in self.seen, "duplicate trained ordinal")
            _check(trajectory["batch"].size == 8, "partial K8 group")
            pending.add(ordinal)
            original = ordinal if ordinal < 8000 else ordinal - 8000
            start, end = position * 8, (position + 1) * 8
            if original_ids is not None:
                _check(all(value == original for value in original_ids[start:end]), "emitted original idx mismatch")
            if task_ids is not None:
                _check(all(value == ordinal for value in task_ids[start:end]), "row/group ordinal mismatch")
            if rollout_ids is not None:
                _check(sorted(rollout_ids[start:end]) == list(range(8)), "rollout K8 identity mismatch")
            group = {
                "stream_ordinal": ordinal,
                "original_idx_by_view": original,
                "padding": ordinal >= 8000,
                "emitted_original_idx": original_ids[start:end] if original_ids is not None else None,
                "emitted_task_indices": task_ids[start:end] if task_ids is not None else None,
                "rollout_indices": rollout_ids[start:end] if rollout_ids is not None else None,
                "attempt_indices": attempts[start:end] if attempts is not None else None,
                "sample_mask": masks[start:end],
                "next_token_mask_counts": [_scalar(x) for x in tokens[start:end]],
            }
            groups.append(group)
        return {"step": step, "groups": groups}

    def commit(self, update):
        _check(update["step"] == self.state["last_completed_step"] + 1, "commit sequence")
        self.seen.update(group["stream_ordinal"] for group in update["groups"])
        self.state["updates"].append(update)
        self.state["last_completed_step"] = update["step"]
        self.state["summary"] = {
            "directly_observed_prompt_positions": len(self.seen),
            "directly_observed_unique_original_rows": len({x if x < 8000 else x - 8000 for x in self.seen}),
            "directly_observed_padding_positions": sum(x >= 8000 for x in self.seen),
            "missing_post24_positions": sorted(set(range(3072, 8064)) - self.seen),
            "post24_complete": self.seen == set(range(3072, 8064)),
            "all8000_per_sample_directly_proven": False,
            "scope": "post24_actual_optimizer_updates_plus_inferred_pre24_prefix; not_gradient_share",
        }
        _atomic_json(self.monitor_path, self.state)

    def checkpoint(self, checkpoint_path, save_step):
        _check(self.state["last_completed_step"] == save_step, "checkpoint coverage step")
        _atomic_json(Path(checkpoint_path) / STATE_NAME, self.state)


def maybe_full_pass_ledger(config, checkpoint, current_step, consumed, monitor_dir):
    contract = checked_contract(config)
    if contract is None:
        return None
    return FullPassLedger(contract, checkpoint, current_step, consumed, monitor_dir)
