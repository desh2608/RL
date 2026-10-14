"""Failure-injection tests for independent, one-snapshot Gym observation."""
import importlib.util
from pathlib import Path
import threading
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    'gym_shutdown_snapshot_subject', ROOT / 'nemo_rl/environments/gym_shutdown.py'
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def entry(pid, ppid, created):
    return types.SimpleNamespace(info={'pid': pid, 'ppid': ppid, 'create_time': created})


def guard_for(roots):
    guard = object.__new__(module.GymShutdownGuard)
    guard.run_helper = types.SimpleNamespace(_processes={
        alias: types.SimpleNamespace(pid=pid) for alias, pid in roots.items()
    })
    guard.observed = {}
    guard.root_create_times = {}
    guard.errors = []
    guard.lock = threading.Lock()
    return guard


def observe(guard, entries, live):
    def process(pid):
        value = live.get(pid)
        if value is None:
            raise module.psutil.NoSuchProcess(pid)
        if isinstance(value, Exception):
            raise value
        return types.SimpleNamespace(
            pid=pid, create_time=lambda: value,
            children=Mock(side_effect=AssertionError('per-root scanning is forbidden')),
            kill=Mock(side_effect=AssertionError('observer must not kill')),
            terminate=Mock(side_effect=AssertionError('observer must not signal')),
        )
    with patch.object(module.psutil, 'process_iter', return_value=iter(entries)) as iterator, \
            patch.object(module.psutil, 'Process', side_effect=process):
        guard._observe()
        iterator.assert_called_once_with(['pid', 'ppid', 'create_time'])


class SnapshotObservationTests(unittest.TestCase):
    def test_stable_tree_aliases_and_unrelated_processes(self):
        guard = guard_for({'a': 10, 'b': 20})
        rows = [entry(10, 1, 10), entry(11, 10, 11), entry(12, 11, 12),
                entry(20, 1, 20), entry(21, 20, 21), entry(80, 1, 1),
                entry(81, 10, 5)]  # Older than parent: possible reused PID.
        observe(guard, rows, {row.info['pid']: row.info['create_time'] for row in rows})
        self.assertEqual(guard.errors, [])
        self.assertEqual({pid for pid, _ in guard.observed}, {10, 11, 12, 20, 21})
        self.assertEqual(guard.observed[(12, 12)]['alias'], 'a')
        self.assertEqual(guard.observed[(21, 21)]['alias'], 'b')

    def test_sixty_six_roots_take_only_one_process_snapshot(self):
        guard = guard_for({f'alias_{pid}': pid for pid in range(100, 166)})
        rows = [entry(pid, 1, pid) for pid in range(100, 166)]
        rows += [entry(pid + 1000, pid, pid + 1) for pid in range(100, 166)]
        observe(guard, rows, {row.info['pid']: row.info['create_time'] for row in rows})
        self.assertEqual(len(guard.observed), 132)
        self.assertEqual(guard.errors, [])

    def test_observed_escape_is_retained_but_new_orphan_is_not_claimed(self):
        guard = guard_for({'a': 10})
        observe(guard, [entry(10, 1, 10), entry(11, 10, 11)], {10: 10, 11: 11})
        observe(guard, [entry(11, 1, 11), entry(12, 1, 12)], {11: 11, 12: 12})
        self.assertEqual(set(guard.observed), {(10, 10), (11, 11)})
        self.assertEqual(guard.observed[(11, 11)]['alias'], 'a')
        self.assertEqual(guard.errors, [])

    def test_root_pid_reuse_fails_closed_without_claiming_foreign_tree(self):
        guard = guard_for({'a': 10})
        observe(guard, [entry(10, 1, 10)], {10: 10})
        observe(guard, [entry(10, 1, 100), entry(99, 10, 101)], {10: 100, 99: 101})
        self.assertEqual(set(guard.observed), {(10, 10)})
        self.assertEqual(guard.errors, ['RuntimeError'])

    def test_root_changes_between_snapshot_and_live_check_fails_closed(self):
        guard = guard_for({'a': 10})
        observe(guard, [entry(10, 1, 10), entry(11, 10, 11)], {10: 100, 11: 11})
        self.assertEqual(guard.observed, {})
        self.assertEqual(guard.errors, ['RuntimeError'])

    def test_reused_descendant_does_not_claim_its_unknown_subtree(self):
        guard = guard_for({'a': 10})
        rows = [entry(10, 1, 10), entry(11, 10, 11), entry(12, 11, 12)]
        observe(guard, rows, {10: 10, 11: 100, 12: 12})
        self.assertEqual(set(guard.observed), {(10, 10)})
        self.assertEqual(guard.errors, [])

    def test_disappearance_during_snapshot_and_live_probe_is_benign(self):
        class Disappeared:
            @property
            def info(self):
                raise module.psutil.NoSuchProcess(90)
        guard = guard_for({'a': 10, 'gone_root': 30})
        rows = [Disappeared(), entry(10, 1, 10), entry(11, 10, 11),
                entry(12, 10, 12), entry(30, 1, 30)]
        observe(guard, rows, {10: 10, 12: 12})
        self.assertEqual(set(guard.observed), {(10, 10), (12, 12)})
        self.assertEqual(guard.errors, [])

    def test_unavailable_owned_identity_is_an_error_not_silent_success(self):
        guard = guard_for({'a': 10})
        observe(guard, [entry(10, 1, 10), entry(11, 10, None)], {10: 10})
        self.assertEqual(guard.errors, ['RuntimeError'])
        guard = guard_for({'a': 10})
        observe(guard, [entry(10, 1, 10)], {10: module.psutil.AccessDenied(10)})
        self.assertEqual(guard.errors, ['AccessDenied'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
