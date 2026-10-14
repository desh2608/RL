"""Opt-in allocation, artifact, and length guards; no change to the OPD loss.

This module is deliberately stdlib-only so the policy decisions can be tested
without importing Ray, CUDA, or model libraries.
"""

import json
import math
import os
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class AllocationDeadline:
    """Request a save at safe boundaries, before starting another long step.

    A running CUDA/Ray operation cannot be interrupted safely. The estimate is
    a conservative admission heuristic, not a hard real-time guarantee.
    """

    def __init__(self, config, environ=None, clock=time.time):
        self.clock = clock
        environ = os.environ if environ is None else environ
        key = config.get("allocation_deadline_env")
        self.deadline = None
        if key:
            if key not in environ:
                raise ValueError(f"Missing allocation deadline environment: {key}")
            self.deadline = float(environ[key])
            if not math.isfinite(self.deadline) or self.deadline <= 0:
                raise ValueError("Allocation deadline must be a finite epoch timestamp")
        self.reserve = float(config.get("deadline_step_reserve_seconds", 900))
        if not math.isfinite(self.reserve) or self.reserve < 0:
            raise ValueError("deadline_step_reserve_seconds must be finite and nonnegative")
        self.durations = []

    def record_step(self, duration):
        if math.isfinite(duration) and duration > 0:
            self.durations = (self.durations + [float(duration)])[-8:]

    def should_save(self, *, allow_prediction=True):
        if self.deadline is None:
            return False
        prediction = max([self.reserve] + self.durations) if allow_prediction else 0
        return self.clock() + prediction >= self.deadline

    def receipt(self):
        return {
            "reason": "allocation_deadline",
            "time_epoch": self.clock(),
            "save_by_epoch": self.deadline,
            "next_step_reserve_seconds": max([self.reserve] + self.durations),
            "scope": "safe optimizer boundary; cannot preempt a running operation",
        }


def _quantile(values, q):
    values = sorted(values)
    if not values:
        return 0.0
    index = (len(values) - 1) * q
    low, high = math.floor(index), math.ceil(index)
    return values[low] + (values[high] - values[low]) * (index - low)


def tail_repetition(tokens, window=4096):
    """Repeated 32-token windows, stride 16, on a bounded response tail.

    This is a lexical diagnostic, not a semantic loop/correctness verifier.
    """
    tail = tokens[-window:]
    if hasattr(tail, "tolist"):
        tail = tail.tolist()
    grams = [tuple(tail[i : i + 32]) for i in range(0, len(tail) - 31, 16)]
    return 1.0 - len(set(grams)) / len(grams) if grams else 0.0


def metadata_domain(info):
    """Use explicit text metadata or the native Gym agent route for monitoring."""
    if not isinstance(info, dict):
        return "unknown"
    domain = info.get("domain")
    if domain is not None and str(domain):
        return str(domain)
    agent_ref = info.get("agent_ref")
    if isinstance(agent_ref, dict) and isinstance(agent_ref.get("name"), str) and agent_ref["name"]:
        return agent_ref["name"]
    return "unknown"


def resolve_monitor_metadata(batch, expected_rows):
    """Resolve row-aligned text metadata or native Gym route metadata."""
    if type(expected_rows) is not int or expected_rows < 0:
        raise ValueError("Invalid length-monitor row count")
    if "extra_env_info" in batch:
        metadata = batch["extra_env_info"]
        if not isinstance(metadata, list) or len(metadata) != expected_rows:
            raise ValueError("Length-monitor extra_env_info rows are not aligned")
        return metadata
    refs = batch.get("agent_ref")
    if not isinstance(refs, list) or len(refs) != expected_rows:
        raise ValueError("Length monitor requires aligned extra_env_info or Gym agent_ref")
    if any(not isinstance(ref, dict) or not isinstance(ref.get("name"), str)
           or not ref["name"].strip() for ref in refs):
        raise ValueError("Length-monitor Gym agent_ref must have a nonempty name")
    return [{"agent_ref": ref} for ref in refs]


class LengthMonitor:
    def __init__(self, config, output_dir, state=None):
        self.config = dict(config or {})
        self.enabled = bool(self.config.get("enabled", False))
        self.output_dir = Path(output_dir)
        self.state = state or {"baseline": {}, "streak": {}, "last_step": -1}
        self.min_samples = int(self.config.get("min_domain_samples", 8))
        self.baseline_batches = int(self.config.get("baseline_batches", 3))
        self.stop_batches = int(self.config.get("stop_after_consecutive", 3))
        if min(self.min_samples, self.baseline_batches, self.stop_batches) < 1:
            raise ValueError("Length-monitor counts must be positive")
        # Optional safety admission rule, independent of lexical loop detection
        # and relative baseline thresholds. Existing recipes remain unchanged.
        self.cap_stop_fraction = self.config.get("cap_hit_stop_fraction")
        self.cap_stop_by_domain = dict(self.config.get("cap_hit_stop_fraction_by_domain") or {})
        self.cap_stop_min_samples = int(self.config.get("cap_hit_stop_min_samples", 32))
        if self.cap_stop_min_samples < 1:
            raise ValueError("cap_hit_stop_min_samples must be positive")
        for threshold in [self.cap_stop_fraction, *self.cap_stop_by_domain.values()]:
            if threshold is not None and (not math.isfinite(float(threshold)) or not 0 < float(threshold) <= 1):
                raise ValueError("Cap stop fractions must be finite and in (0, 1]")

    def record_training_tokens(self, metadata, response_tokens, sample_masks, step):
        """Record domain coverage after sample filtering, before token IS weighting.

        This measures the sample/token masks, not gradient mass or the later
        per-token ICEPOP acceptance/importance weights.
        """
        if not self.enabled:
            return {}
        if not (len(metadata) == len(response_tokens) == len(sample_masks)):
            raise ValueError("Domain token accounting rows are not aligned")
        domains = {}
        for info, tokens, mask in zip(metadata, response_tokens, sample_masks):
            name = metadata_domain(info)
            values = domains.setdefault(name, {"responses": 0, "raw_response_tokens": 0,
                                               "sample_mask_valid_tokens": 0.0,
                                               "unmasked_responses": 0})
            if tokens < 0 or not math.isfinite(float(mask)) or mask < 0:
                raise ValueError("Invalid response-token count or sample mask")
            values["responses"] += 1
            values["raw_response_tokens"] += int(tokens)
            values["sample_mask_valid_tokens"] += float(tokens) * float(mask)
            values["unmasked_responses"] += int(mask > 0)
        total = sum(v["sample_mask_valid_tokens"] for v in domains.values())
        for values in domains.values():
            values["sample_mask_valid_token_share"] = values["sample_mask_valid_tokens"] / total if total else 0.0
        report = {"step": step, "time_epoch": time.time(), "domains": domains,
                  "scope": "response token mask times sample mask; excludes token importance weights"}
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with (self.output_dir / "domain_token_metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(report, sort_keys=True, allow_nan=False) + "\n")
        return {f"domain_tokens/{name}/{key}": value
                for name, values in domains.items() for key, value in values.items()}

    def observe(self, batch, step, cap):
        if not self.enabled or step <= self.state["last_step"]:
            return None
        domains = defaultdict(list)
        metadata = resolve_monitor_metadata(batch, len(batch["message_log"]))
        truncated = batch.get("truncated")
        for i, messages in enumerate(batch["message_log"]):
            info = metadata[i]
            domain = metadata_domain(info)
            lengths, repetitions = [], []
            for message in messages:
                if message.get("role") != "assistant" or "generation_logprobs" not in message:
                    continue
                tokens = message.get("token_ids", [])
                lengths.append(len(tokens))
                repetitions.append(tail_repetition(tokens))
            length = sum(lengths)
            # Gym's legacy ``truncated`` flag can mean a full context window,
            # not exhaustion of a single generation's output budget. Keep that
            # signal, but independently detect per-turn output-budget hits.
            # Do not sum separate assistant turns against a per-turn cap.
            output_cap_hit = any(turn_length >= cap for turn_length in lengths)
            trajectory_truncated = bool(truncated[i]) if truncated is not None else False
            cap_hit = trajectory_truncated or output_cap_hit
            repeated = length >= 8192 and max(repetitions, default=0) >= 0.80
            domains[domain].append(
                (length, cap_hit, repeated, output_cap_hit, trajectory_truncated)
            )
        return self.observe_rows(domains, step)

    def observe_rows(self, domains, step):
        """Rows have length/cap/repetition and optional separate cap signals."""
        report = {"step": step, "time_epoch": time.time(), "domains": {}, "alerts": [], "stop": False}
        for absent in self.state["streak"].keys() - domains.keys():
            self.state["streak"][absent] = 0
        for domain, rows in sorted(domains.items()):
            lengths = [int(row[0]) for row in rows]
            count = len(rows)
            if not count:
                continue
            stats = {
                "count": count,
                "mean": sum(lengths) / count,
                "p50": _quantile(lengths, 0.5),
                "p90": _quantile(lengths, 0.9),
                "cap_hit_fraction": sum(bool(row[1]) for row in rows) / count,
                "long_repetitive_tail_fraction": sum(bool(row[2]) for row in rows) / count,
            }
            if all(len(row) >= 5 for row in rows):
                stats["output_cap_hit_fraction"] = sum(bool(row[3]) for row in rows) / count
                stats["trajectory_truncated_fraction"] = sum(bool(row[4]) for row in rows) / count
            report["domains"][domain] = stats
            if count < self.min_samples:
                self.state["streak"][domain] = 0
                continue
            baseline = self.state["baseline"].setdefault(domain, [])
            extreme = stats["cap_hit_fraction"] >= 0.50 and stats["long_repetitive_tail_fraction"] >= 0.50
            inflated = False
            if len(baseline) >= self.baseline_batches:
                mean_ref = sum(item["mean"] for item in baseline) / len(baseline)
                p90_ref = sum(item["p90"] for item in baseline) / len(baseline)
                inflated = (
                    stats["mean"] >= max(4096, 2.5 * mean_ref)
                    and stats["p90"] >= max(8192, 2.0 * p90_ref)
                    and (stats["cap_hit_fraction"] >= 0.20 or stats["long_repetitive_tail_fraction"] >= 0.10)
                )
            elif not extreme:
                baseline.append({"mean": stats["mean"], "p90": stats["p90"]})
            cap_limit = self.cap_stop_by_domain.get(domain, self.cap_stop_fraction)
            cap_excess = (cap_limit is not None and count >= self.cap_stop_min_samples
                          and stats["cap_hit_fraction"] >= float(cap_limit))
            suspicious = extreme or inflated or cap_excess
            streak = self.state["streak"].get(domain, 0) + 1 if suspicious else 0
            self.state["streak"][domain] = streak
            stats["suspicious_streak"] = streak
            if suspicious:
                reason = ("sustained_cap_fraction" if cap_excess else
                          "cap_and_repetitive_tail" if extreme else "length_inflation_with_failure_signal")
                report["alerts"].append({"domain": domain, "reason": reason, "consecutive_batches": streak})
                report["stop"] |= streak >= self.stop_batches
        self.state["last_step"] = step
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with (self.output_dir / "length_metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(report, sort_keys=True, allow_nan=False) + "\n")
        atomic_json(self.output_dir / "length_monitor_state.json", self.state)
        atomic_json(self.output_dir / "LENGTH_STATUS.json", report)
        if report["alerts"]:
            atomic_json(self.output_dir / "LENGTH_ALERT.json", report)
            print(f"[LENGTH_ALERT] {json.dumps(report)}", flush=True)
        return report


def export_hf_atomically(export_fn, root, step, *, source_model, keep_latest=None):
    """Publish only complete collective exports; never prunes recovery MCore."""
    root = Path(root)
    if keep_latest is not None and int(keep_latest) < 1:
        raise ValueError("HF retention must retain at least one export")
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"step_{step}"
    if (destination / "EXPORT_COMPLETE.json").is_file():
        marker = json.loads((destination / "EXPORT_COMPLETE.json").read_text())
        if marker.get("step") != step or marker.get("source_model") != source_model:
            raise ValueError(f"Conflicting HF export identity: {destination}")
        return str(destination)
    if destination.exists():
        raise FileExistsError(f"Unmarked HF export exists: {destination}")
    temporary = root / f"tmp_step_{step}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    atomic_json(root / "HF_EXPORT_PENDING.json", {"step": step, "source_model": source_model})
    export_fn(str(temporary))
    shards = list(temporary.glob("*.safetensors"))
    if not (temporary / "config.json").is_file() or not shards or any(p.stat().st_size == 0 for p in shards):
        raise RuntimeError("HF export returned without config and safetensors weights")
    # The collective strict Bridge writer validates mapped keys. The index
    # additionally ensures every referenced shard was actually published.
    index = temporary / "model.safetensors.index.json"
    if index.exists():
        weights = json.loads(index.read_text()).get("weight_map", {})
        if not weights:
            raise RuntimeError("HF export has an empty weight map")
        for name in set(weights.values()):
            shard = temporary / name
            if shard.parent != temporary or not shard.is_file() or shard.stat().st_size == 0:
                raise RuntimeError(f"HF export is missing shard: {name}")
    atomic_json(temporary / "EXPORT_COMPLETE.json", {"step": step, "source_model": source_model, "time_epoch": time.time(), "validation": "strict Bridge export and shard presence; no model eval"})
    os.rename(temporary, destination)
    (root / "HF_EXPORT_PENDING.json").unlink(missing_ok=True)
    if keep_latest is not None:
        complete = sorted((p for p in root.glob("step_*") if p.name[5:].isdigit() and (p / "EXPORT_COMPLETE.json").is_file()), key=lambda p: int(p.name[5:]), reverse=True)
        for stale in complete[int(keep_latest) :]:
            shutil.rmtree(stale)
    return str(destination)


class IdleGuard:
    """Opt-in, job/run/source-bound reporting and safe-stop input consumer.

    Only completed events advance counters. Repeated waiting reports are not
    progress. This class never signals a process or invents an optimizer step.
    """
    def __init__(self, config, environ=None, clock=time.time):
        import hashlib
        import uuid
        self.clock = clock
        self.enabled = config.get('idle_watchdog_enabled', False)
        if type(self.enabled) is not bool:
            raise ValueError('idle_watchdog_enabled must be a boolean')
        if not self.enabled:
            env = os.environ if environ is None else environ
            if env.get('NVC_IDLE_GUARD_TOKEN') or 'NVC_IDLE_GUARD_JOB_ID' in env:
                raise ValueError('Host idle guard enabled but native config dropped its opt-in')
            return
        # Same reviewed range as scripts/gym_idle_watchdog.frozen_policy: the historical 20/25 min
        # policy up to 60/65 min, terminate 5..15 min after the request. Multi-turn agentic steps
        # (tau2 ~17 policy turns, simulator retries, gap-fill rounds) legitimately keep the learner
        # idle for more than 20 min (smoke job 35894567 was killed at 1261 s while gap-filling).
        request_seconds = config.get('idle_watchdog_request_seconds', 1200)
        terminate_seconds = config.get('idle_watchdog_terminate_seconds', 1500)
        if not (isinstance(request_seconds, int) and isinstance(terminate_seconds, int)):
            raise ValueError('Idle watchdog seconds must be integers')
        if not 1200 <= request_seconds <= 3600:
            raise ValueError('idle_watchdog_request_seconds must be within [1200, 3600]')
        if not request_seconds + 300 <= terminate_seconds <= request_seconds + 900:
            raise ValueError('idle_watchdog_terminate_seconds must be request+300 .. request+900')
        if not config.get('enabled', False):
            raise ValueError('Idle stop requires recovery checkpointing')
        env = os.environ if environ is None else environ
        # init_ray deliberately removes SLURM_* before constructing actor
        # runtime environments. The host supervisor carries its exact identity
        # in this dedicated variable; never recover it from a directory search
        # or restore Slurm launcher variables into the driver/actors.
        job_id = env.get('NVC_IDLE_GUARD_JOB_ID')
        if not isinstance(job_id, str) or re.fullmatch(r'[1-9][0-9]*', job_id) is None:
            raise ValueError('Missing or invalid NVC_IDLE_GUARD_JOB_ID')
        if 'SLURM_JOB_ID' in env and env['SLURM_JOB_ID'] != job_id:
            raise ValueError('Idle supervisor and Slurm job identities disagree')
        run = Path(env['NVC_DEBUG_RUN']).resolve(strict=True)
        self.directory = run / 'idle-guard' / job_id
        owner = json.loads((self.directory / 'OWNER.json').read_text())
        self.binding = {
            'schema': 'gym-idle-guard.v1', 'job_id': job_id,
            'run': str(run), 'source_sha256': (run / 'source.sha256').read_text().split()[0],
            'config_sha256': hashlib.sha256((run / 'config.yaml').read_bytes()).hexdigest(),
            'token': env['NVC_IDLE_GUARD_TOKEN'],
        }
        if not all(owner.get(k) == v for k, v in self.binding.items()):
            raise ValueError('Idle supervisor ownership/config binding mismatch')
        self.state = {**self.binding, 'driver_session': uuid.uuid4().hex,
                      'driver_pid': os.getpid(), 'driver_phase': env.get('NVC_GYM_DRIVER_PHASE'),
                      'semantic_seq': 0, 'learner_seq': 0}
        self.last_write = float('-inf')
        self.last_buffer = None
        self.rejected = None
        self.stage('startup', step=0)

    def stage(self, phase, *, step, buffered=None, progress=False, learner=False):
        if not self.enabled:
            return
        if phase not in {'startup', 'buffer_wait', 'training', 'refit', 'saving', 'stopping'}:
            raise ValueError('Invalid idle guard phase')
        changed = self.state.get('phase') != phase or self.state.get('step') != step
        if buffered is not None:
            key = (step, int(buffered))
            if self.last_buffer is None or self.last_buffer[0] != step:
                progress = progress or buffered > 0
            else:
                progress = progress or buffered > self.last_buffer[1]
            self.last_buffer = key
            self.state['buffer_entries'] = int(buffered)
        if progress:
            self.state['semantic_seq'] += 1
        if learner:
            self.state['learner_seq'] += 1
        self.state.update(phase=phase, step=int(step), reported_at_epoch=self.clock())
        if changed or progress or learner or self.clock() - self.last_write >= 30:
            atomic_json(self.directory / 'STATUS.json', self.state)
            self.last_write = self.clock()

    def stop_requested(self):
        if not self.enabled:
            return None
        path = self.directory / 'STOP_INPUT.json'
        if not path.is_file() or path.is_symlink():
            return None
        request = json.loads(path.read_text())
        now = self.clock()
        valid = all(request.get(k) == v for k, v in self.binding.items())
        issued, expires = request.get('issued_at_epoch'), request.get('not_after_epoch')
        valid = valid and type(issued) in (float, int) and type(expires) in (float, int)
        valid = valid and math.isfinite(issued) and math.isfinite(expires)
        valid = valid and issued <= now <= expires and 0 < expires - issued <= 300
        valid = valid and isinstance(request.get('request_id'), str) and len(request['request_id']) == 32
        if not valid:
            signature = json.dumps(request, sort_keys=True)
            if signature != self.rejected:
                atomic_json(self.directory / 'REJECTED_STOP.json', {
                    **self.binding, 'reason': 'stale_or_mismatched_stop_input', 'time_epoch': now,
                })
                self.rejected = signature
            return None
        return {'reason': 'idle_watchdog', 'idle_reason': request['reason'],
                'request_id': request['request_id'], 'time_epoch': now,
                'scope': 'safe optimizer boundary; no fabricated update'}

    def stopped(self, reason, step, checkpoint):
        if not self.enabled:
            return
        self.stage('stopping', step=step)
        atomic_json(self.directory / 'STOPPED.json', {
            **self.binding, **reason, 'completed_optimizer_step': int(step),
            'checkpoint': str(checkpoint) if checkpoint is not None else None,
            'fresh_initial_model_preserved': step == 0 and checkpoint is None,
            'status': 'stopped_at_safe_boundary',
        })
