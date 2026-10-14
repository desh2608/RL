"""IdleGuard accepts the reviewed idle-policy range (mirrors scripts/gym_idle_watchdog.frozen_policy)."""

import unittest

from nemo_rl.utils.training_guardrails import IdleGuard


class IdleGuardPolicyRangeTest(unittest.TestCase):
    def _cfg(self, request, terminate):
        return {"idle_watchdog_enabled": True, "idle_watchdog_request_seconds": request,
                "idle_watchdog_terminate_seconds": terminate, "enabled": True}

    def test_disabled_guard_ignores_policy(self):
        guard = IdleGuard({"idle_watchdog_enabled": False}, environ={})
        self.assertFalse(guard.enabled)

    def test_range_is_validated_before_environment(self):
        # Inside the range the constructor proceeds to the environment binding and fails there.
        for request, terminate in ((1200, 1500), (2400, 3000), (3600, 4500)):
            with self.assertRaises(ValueError) as ctx:
                IdleGuard(self._cfg(request, terminate), environ={})
            self.assertIn("NVC_IDLE_GUARD_JOB_ID", str(ctx.exception))
        for request, terminate in ((600, 900), (3601, 4000), (2400, 2500), (2400, 3400), (1200, 1500.0)):
            with self.assertRaises(ValueError) as ctx:
                IdleGuard(self._cfg(request, terminate), environ={})
            self.assertNotIn("NVC_IDLE_GUARD_JOB_ID", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
