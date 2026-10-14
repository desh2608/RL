"""CPU regression for optional cap safety, using the production stdlib module."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

_path = Path(__file__).resolve().parents[3] / "nemo_rl/utils/training_guardrails.py"
_spec = importlib.util.spec_from_file_location("training_guardrails", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


class CapStopTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def monitor(self, **kwargs):
        return _module.LengthMonitor({"enabled": True, **kwargs}, self.tmp.name)

    def rows(self, n=100, caps=20):
        return [(65536, i < caps, False) for i in range(n)]

    def test_default_does_not_change_saturated_baseline_behavior(self):
        m = self.monitor()
        for step in range(1, 7):
            self.assertFalse(m.observe_rows({"if": self.rows(caps=100)}, step)["stop"])

    def test_cap_only_stop_does_not_require_relative_inflation_or_repetition(self):
        m = self.monitor(cap_hit_stop_fraction=0.2)
        for step in (1, 2):
            self.assertFalse(m.observe_rows({"if": self.rows()}, step)["stop"])
        r = m.observe_rows({"if": self.rows()}, 3)
        self.assertTrue(r["stop"])
        self.assertEqual(r["alerts"][0]["reason"], "sustained_cap_fraction")

    def test_nonconsecutive_caps_reset_streak(self):
        m = self.monitor(cap_hit_stop_fraction=0.2)
        for step, caps in enumerate((20, 20, 19, 20, 20), 1):
            self.assertFalse(m.observe_rows({"if": self.rows(caps=caps)}, step)["stop"])
        self.assertTrue(m.observe_rows({"if": self.rows()}, 6)["stop"])

    def test_sparse_domain_does_not_trigger_cap_only_stop(self):
        m = self.monitor(cap_hit_stop_fraction=0.2)
        for step in range(1, 7):
            self.assertFalse(m.observe_rows({"vif": self.rows(n=20, caps=20)}, step)["stop"])

    def test_domain_override_and_restored_streak(self):
        cfg = {"cap_hit_stop_fraction": 0.2, "cap_hit_stop_fraction_by_domain": {"dtau": 0.05}}
        m = self.monitor(**cfg)
        for step in (1, 2):
            r = m.observe_rows({"dtau": self.rows(caps=5), "if": self.rows(caps=5)}, step)
            self.assertFalse(r["stop"])
        restored = _module.LengthMonitor({"enabled": True, **cfg}, self.tmp.name, state=m.state)
        r = restored.observe_rows({"dtau": self.rows(caps=5), "if": self.rows(caps=5)}, 3)
        self.assertTrue(r["stop"])
        self.assertEqual([a["domain"] for a in r["alerts"]], ["dtau"])

    def test_invalid_thresholds_fail_before_training(self):
        for value in (0, -0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.monitor(cap_hit_stop_fraction=value)


if __name__ == "__main__":
    unittest.main()
