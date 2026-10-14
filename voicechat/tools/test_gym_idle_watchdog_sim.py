#!/usr/bin/env python3
"""gym_idle_watchdog.py with simulator nodes: job-bound sim receipts pin the idle clocks, driver-absent cap.

Stdlib unittest only; the watchdog gets a fake monotonic clock / wall clock and a fake main Popen.
Run inside the training container with the driver venv:
  /opt/nemotronvc/venvs/driver/bin/python scripts/test_gym_idle_watchdog_sim.py
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest

SCRIPTS = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('gym_idle_watchdog', SCRIPTS / 'gym_idle_watchdog.py')
watchdog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watchdog)

CONFIG = '''policy:
  model_name: /models/student
checkpointing:
  checkpoint_dir: /lineage/checkpoints
  idle_watchdog_enabled: true
  idle_watchdog_request_seconds: 1200
  idle_watchdog_terminate_seconds: 1500
'''
JOB = '4242'
HOST = 'batch-block5-00346'


class FakeMain:
    pid = 777

    def poll(self):
        return None


class Clock:
    def __init__(self, start=10000.0):
        self.now = start

    def __call__(self):
        return self.now


class WatchdogFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run = Path(self.tmp.name) / 'run'
        self.run.mkdir()
        (self.run / 'config.yaml').write_text(CONFIG)
        (self.run / 'source.sha256').write_text(hashlib.sha256(b'source').hexdigest() + '  source.tar.gz\n')
        self.clock = Clock()
        self.wall = Clock(1_700_000_000.0)
        self.main = FakeMain()

    def tearDown(self):
        self.tmp.cleanup()

    def make(self, allocation_elapsed=0.0, **kwargs):
        policy = watchdog.frozen_policy(self.run)
        return watchdog.IdleWatchdog(self.run, JOB, policy, clock=self.clock, wall=self.wall,
                                     allocation_start=self.wall() - allocation_elapsed, **kwargs)

    def sim_dir(self, job=JOB):
        path = self.run / 'sim' / job
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_hosts(self, job=JOB, hosts=(HOST,)):
        (self.sim_dir(job) / 'HOSTS').write_text(''.join(h + '\n' for h in hosts))

    def write_ready(self, job=JOB, host=HOST):
        (self.sim_dir(job) / f'READY-{host}.json').write_text(json.dumps(
            {'schema': 'nemotronvc.sim_role.v2', 'status': 'ready', 'job_id': job, 'host': host, 'port': 18100}) + '\n')

    def advance(self, seconds):
        self.clock.now += seconds
        self.wall.now += seconds

    def tick(self, dog, seconds=60):
        self.advance(seconds)
        return dog.tick(self.main)

    def status(self, dog):
        return json.loads((dog.directory / 'WATCHDOG_STATUS.json').read_text())

    def driver_status(self, dog, phase='startup', semantic=0, learner=0):
        watchdog.atomic_json(dog.directory / 'STATUS.json', {
            **dog.binding, 'phase': phase, 'semantic_seq': semantic, 'learner_seq': learner, 'driver_session': 's1'})


class FrozenPolicy(WatchdogFixture):
    def test_values_are_still_1200_1500(self):
        self.assertEqual(watchdog.frozen_policy(self.run), {'request_seconds': 1200, 'terminate_seconds': 1500})

    def test_policy_accepts_reviewed_range_and_rejects_outside(self):
        cfg = self.run / 'config.yaml'
        text = cfg.read_text()
        cfg.write_text(text.replace('idle_watchdog_request_seconds: 1200', 'idle_watchdog_request_seconds: 2400')
                           .replace('idle_watchdog_terminate_seconds: 1500', 'idle_watchdog_terminate_seconds: 3000'))
        self.assertEqual(watchdog.frozen_policy(self.run), {'request_seconds': 2400, 'terminate_seconds': 3000})
        cfg.write_text(text.replace('idle_watchdog_request_seconds: 1200', 'idle_watchdog_request_seconds: 600'))
        with self.assertRaises(ValueError):
            watchdog.frozen_policy(self.run)
        cfg.write_text(text.replace('idle_watchdog_terminate_seconds: 1500', 'idle_watchdog_terminate_seconds: 1300'))
        with self.assertRaises(ValueError):
            watchdog.frozen_policy(self.run)
        cfg.write_text(text)
        self.assertEqual(watchdog.DRIVER_ABSENT_SECONDS, 2400)

    def test_owner_records_driver_absent_cap_and_job_scoped_sim_dir(self):
        dog = self.make()
        owner = json.loads((dog.directory / 'OWNER.json').read_text())
        self.assertEqual(owner['driver_absent_seconds'], 2400)
        self.assertEqual(owner['sim_receipts'], str(self.run / 'sim' / JOB))
        self.assertEqual(owner['policy'], {'request_seconds': 1200, 'terminate_seconds': 1500})


class SimPendingHosts(WatchdogFixture):
    def test_no_hosts_file_means_nothing_pending(self):
        self.assertEqual(watchdog.sim_pending_hosts(self.run, JOB), [])
        self.sim_dir()
        self.assertEqual(watchdog.sim_pending_hosts(self.run, JOB), [])

    def test_only_this_jobs_directory_is_read(self):
        self.write_hosts(job='1111')
        self.assertEqual(watchdog.sim_pending_hosts(self.run, JOB), [])
        self.write_hosts()
        self.assertEqual(watchdog.sim_pending_hosts(self.run, JOB), [HOST])
        self.write_ready(job='1111')
        self.assertEqual(watchdog.sim_pending_hosts(self.run, JOB), [HOST], 'another job READY must not count')
        self.write_ready()
        self.assertEqual(watchdog.sim_pending_hosts(self.run, JOB), [])

    def test_empty_ready_file_is_still_pending(self):
        self.write_hosts()
        (self.sim_dir() / f'READY-{HOST}.json').write_text('')
        self.assertEqual(watchdog.sim_pending_hosts(self.run, JOB), [HOST])

    def test_unsafe_host_name_refused(self):
        self.write_hosts(hosts=('../evil',))
        with self.assertRaises(ValueError):
            watchdog.sim_pending_hosts(self.run, JOB)


class HistoricalLayout(WatchdogFixture):
    """No simulator nodes: behaviour is unchanged (latch at 1200 s, terminate at 1500 s from allocation start)."""

    def test_latch_and_terminate_from_allocation_start(self):
        dog = self.make(allocation_elapsed=1100)
        self.assertFalse(self.tick(dog, 60))          # 1160 s
        self.assertFalse((dog.directory / 'STOP_INPUT.json').exists())
        self.assertFalse(self.tick(dog, 60))          # 1220 s -> latched
        request = json.loads((dog.directory / 'STOP_INPUT.json').read_text())
        self.assertEqual(request['reason'], 'no_semantic_progress')
        self.assertIsNone(self.status(dog)['pin_reason'])
        self.assertFalse(self.tick(dog, 200))         # 1420 s
        self.assertTrue(self.tick(dog, 100))          # 1520 s -> terminate
        self.assertTrue((dog.directory / 'MAIN_TERM_INTENT.json').exists())


class BufferWaitProgress(WatchdogFixture):
    """Growing buffer entries in buffer_wait are progress: the learner-idle budget defers to the semantic clock."""

    def test_growing_buffer_never_latches_on_learner_budget(self):
        dog = self.make()
        self.driver_status(dog, phase='buffer_wait', semantic=1, learner=1)
        self.assertFalse(self.tick(dog, 60))
        semantic = 1
        for _ in range(8):                            # 4800 s of buffer growth, far past request/terminate
            semantic += 1
            self.driver_status(dog, phase='buffer_wait', semantic=semantic, learner=1)
            self.assertFalse(self.tick(dog, 600))
            self.assertFalse((dog.directory / 'STOP_INPUT.json').exists())
        snapshot = self.status(dog)
        self.assertLessEqual(snapshot['learner_idle_seconds'], snapshot['semantic_idle_seconds'])

    def test_stalled_buffer_still_latches_with_the_stall_reason(self):
        dog = self.make()
        self.driver_status(dog, phase='buffer_wait', semantic=1, learner=1)
        self.assertFalse(self.tick(dog, 60))
        self.driver_status(dog, phase='buffer_wait', semantic=2, learner=1)
        self.assertFalse(self.tick(dog, 600))
        for _ in range(6):                            # no new groups: the semantic budget latches
            self.tick(dog, 600)
            if (dog.directory / 'STOP_INPUT.json').exists():
                break
        request = json.loads((dog.directory / 'STOP_INPUT.json').read_text())
        self.assertEqual(request['reason'], 'no_semantic_progress')

    def test_learner_budget_still_applies_outside_buffer_wait(self):
        dog = self.make()
        self.driver_status(dog, phase='stopping', semantic=1, learner=1)
        self.assertFalse(self.tick(dog, 60))
        semantic = 1
        latched = False
        for _ in range(8):
            semantic += 1
            self.driver_status(dog, phase='stopping', semantic=semantic, learner=1)
            self.tick(dog, 600)
            if (dog.directory / 'STOP_INPUT.json').exists():
                latched = True
                break
        self.assertTrue(latched)
        request = json.loads((dog.directory / 'STOP_INPUT.json').read_text())
        self.assertEqual(request['reason'], 'learner_idle_budget')


class SimPendingPinsClocks(WatchdogFixture):
    def test_pending_receipt_pins_both_clocks_past_the_idle_budget(self):
        self.write_hosts()
        dog = self.make(allocation_elapsed=1100)
        self.assertFalse(self.tick(dog, 200))         # 1300 s since allocation: would have latched without the pin
        status = self.status(dog)
        self.assertEqual(status['pin_reason'], 'sim_pending')
        self.assertEqual(status['sim_pending_hosts'], [HOST])
        self.assertEqual((status['semantic_idle_seconds'], status['learner_idle_seconds']), (0.0, 0.0))
        self.assertAlmostEqual(status['driver_absent_seconds'], 1300.0)
        self.assertFalse((dog.directory / 'STOP_INPUT.json').exists())
        self.assertFalse(self.tick(dog, 600))         # 1900 s, still pinned
        self.assertFalse((dog.directory / 'STOP_INPUT.json').exists())

    def test_ready_receipt_unpins_and_idle_budget_restarts_from_there(self):
        self.write_hosts()
        dog = self.make(allocation_elapsed=200)
        self.assertFalse(self.tick(dog, 300))         # pinned at 500 s
        self.write_ready()
        self.assertFalse(self.tick(dog, 60))          # unpinned; ages count from the last pinned tick (60 s)
        status = self.status(dog)
        self.assertIsNone(status['pin_reason'])
        self.assertEqual(status['sim_pending_hosts'], [])
        self.assertAlmostEqual(status['semantic_idle_seconds'], 60.0)
        self.assertFalse(self.tick(dog, 1100))        # 1160 s idle (1660 s since allocation, below the 2400 s cap)
        self.assertFalse((dog.directory / 'STOP_INPUT.json').exists())
        self.assertFalse(self.tick(dog, 60))          # 1220 s idle -> latched
        self.assertEqual(json.loads((dog.directory / 'STOP_INPUT.json').read_text())['reason'], 'no_semantic_progress')
        self.assertAlmostEqual(self.status(dog)['driver_absent_seconds'], 1720.0)

    def test_other_jobs_receipts_do_not_pin(self):
        self.write_hosts(job='1111')                  # a previous allocation of the same run
        dog = self.make(allocation_elapsed=1100)
        self.assertFalse(self.tick(dog, 200))         # 1300 s -> latched because nothing of ours is pending
        self.assertIsNone(self.status(dog)['pin_reason'])
        self.assertTrue((dog.directory / 'STOP_INPUT.json').exists())

    def test_pin_stops_once_the_driver_has_published_status(self):
        self.write_hosts()
        dog = self.make()
        self.assertFalse(self.tick(dog, 60))
        self.driver_status(dog, phase='buffer_wait', semantic=1, learner=0)
        self.assertFalse(self.tick(dog, 60))          # driver seen: semantic progress -> clocks run normally
        self.assertIsNone(self.status(dog)['pin_reason'])
        self.assertIsNone(self.status(dog)['driver_absent_seconds'])
        (dog.directory / 'STATUS.json').unlink()      # HOSTS still has no READY, but the driver existed
        self.assertFalse(self.tick(dog, 60))
        self.assertIsNone(self.status(dog)['pin_reason'])
        self.assertAlmostEqual(self.status(dog)['semantic_idle_seconds'], 60.0)


class DriverAbsentCap(WatchdogFixture):
    def test_pinned_job_without_driver_dies_at_2400s(self):
        self.write_hosts()
        dog = self.make(allocation_elapsed=2300)
        self.assertFalse(self.tick(dog, 60))          # 2360 s, pinned
        self.assertEqual(self.status(dog)['pin_reason'], 'sim_pending')
        self.assertTrue(self.tick(dog, 60))           # 2420 s -> terminate on this tick
        request = json.loads((dog.directory / 'STOP_INPUT.json').read_text())
        self.assertEqual(request['reason'], 'driver_absent')
        intent = json.loads((dog.directory / 'MAIN_TERM_INTENT.json').read_text())
        self.assertEqual(intent['request_id'], request['request_id'])

    def test_cap_is_measured_from_allocation_start_without_sim_nodes(self):
        dog = self.make(allocation_elapsed=1150)
        # Idle latch fires first (reason no_semantic_progress); the cap never re-latches a latched request.
        self.assertFalse(self.tick(dog, 60))          # 1210 s -> latched, terminate at 1500 s
        self.assertEqual(json.loads((dog.directory / 'STOP_INPUT.json').read_text())['reason'], 'no_semantic_progress')
        self.assertAlmostEqual(self.status(dog)['driver_absent_seconds'], 1210.0)
        self.assertEqual(self.status(dog)['driver_absent_cap_seconds'], 2400)
        self.assertTrue(self.tick(dog, 1200))         # 2410 s: terminated by the idle latch, request unchanged
        self.assertEqual(json.loads((dog.directory / 'STOP_INPUT.json').read_text())['reason'], 'no_semantic_progress')

    def test_custom_cap_and_invalid_cap(self):
        self.write_hosts()
        dog = self.make(allocation_elapsed=100, driver_absent_seconds=200)
        self.assertFalse(self.tick(dog, 60))
        self.assertTrue(self.tick(dog, 60))
        with self.assertRaises(ValueError):
            self.make(driver_absent_seconds=0)

    def test_driver_present_is_never_capped(self):
        dog = self.make(allocation_elapsed=2500)
        self.driver_status(dog, phase='training', semantic=3, learner=2)
        self.assertFalse(self.tick(dog, 60))
        self.assertFalse((dog.directory / 'STOP_INPUT.json').exists())
        self.assertIsNone(self.status(dog)['driver_absent_seconds'])


if __name__ == '__main__':
    unittest.main()
