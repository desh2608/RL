import importlib.util
import json
import os
from pathlib import Path
import socket
import tempfile
import types
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("ray_ports", Path(__file__).with_name("ray_ports.py"))
ports = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ports)


class RayPorts(unittest.TestCase):
    def test_client_port_does_not_reuse_held_listener(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            chosen = ports.choose_client_port("127.0.0.1")
            self.assertNotEqual(chosen, listener.getsockname()[1])
            self.assertGreaterEqual(chosen, 1024)

    def test_wrong_host_or_zero_port_rejected(self):
        for address in ("127.0.0.2:12000", "127.0.0.1:0", "127.0.0.1:65536", "127.0.0.1:x", None):
            with self.assertRaises(ValueError):
                ports.validate_address(address, "127.0.0.1")

    def test_worker_must_join_published_head(self):
        with self.assertRaisesRegex(ValueError, "different Ray head"):
            ports.validate_address("127.0.0.1:32101", "127.0.0.1", "127.0.0.1:32100")

    def test_publishes_actual_gcs_address_not_a_fixed_guess(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            args = types.SimpleNamespace(run=run, job_id="42", rank=0, head_ip="127.0.0.1", supervisor_pid=os.getpid(), client_port=31200, expected_address=None, timeout=1)
            fake_ray = types.ModuleType("ray"); fake_ray.__version__ = "test"
            fake_utils = types.ModuleType("ray._private.utils")
            fake_utils.read_ray_address = lambda: "127.0.0.1:32123"
            with patch.dict("sys.modules", {"ray": fake_ray, "ray._private.utils": fake_utils}):
                ports.wait_and_record(args)
            self.assertEqual((run / "RAY_ADDRESS").read_text(), "127.0.0.1:32123\n")
            self.assertEqual(json.loads((run / "ray-port-rank-0.json").read_text())["gcs_address"], "127.0.0.1:32123")

    def test_dead_supervisor_cannot_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            args = types.SimpleNamespace(run=run, timeout=1, supervisor_pid=100)
            fake_ray = types.ModuleType("ray")
            fake_utils = types.ModuleType("ray._private.utils")
            fake_utils.read_ray_address = lambda: "127.0.0.1:12000"
            with patch.dict("sys.modules", {"ray": fake_ray, "ray._private.utils": fake_utils}), patch.object(os, "kill", side_effect=ProcessLookupError):
                with self.assertRaises(ProcessLookupError):
                    ports.wait_and_record(args)
            self.assertFalse((run / "RAY_ADDRESS").exists())


class LauncherContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path(__file__).with_name("ray_role.sh").read_text()

    def reject(self, source):
        with self.assertRaises(ValueError):
            ports.validate_launcher(source)

    def test_current_launcher_passes(self):
        self.assertEqual(ports.validate_launcher(self.source)["status"], "PASS")

    def test_fixed_or_missing_ports_rejected(self):
        for flag in ("min-worker-port", "max-worker-port", "node-manager-port", "object-manager-port",
                     "runtime-env-agent-port", "dashboard-agent-grpc-port", "dashboard-agent-listen-port",
                     "metrics-export-port", "port", "dashboard-port"):
            with self.subTest(flag=flag):
                self.reject(self.source.replace("--" + flag + "=0", "--" + flag + "=12102"))
                self.reject(self.source.replace("--" + flag + "=0", ""))
        self.reject(self.source.replace('--ray-client-server-port="$client_port"', '--ray-client-server-port=12001'))

    def test_later_overrides_on_either_start_rejected(self):
        for marker in ('--head --port=0', '--address="$head_address"'):
            with self.subTest(marker=marker):
                self.reject(self.source.replace(marker, marker + ' --node-manager-port=12102'))
                self.reject(self.source.replace(marker, marker + ' --node-manager-port=0'))
        for assignment in ('args+=(--node-manager-port=12102)', 'args[0]=--node-manager-port=12102',
                           'args=(--node-manager-port=12102)'):
            with self.subTest(assignment=assignment):
                marker = '\nif (( rank == 0 )); then'
                self.reject(self.source.replace(marker, '\n' + assignment + marker))

    def test_supervisor_block_and_pid_capture_required(self):
        for prefix in ('"$ray" start --block --head', '"$ray" start --block --address='):
            with self.subTest(prefix=prefix):
                self.reject(self.source.replace(prefix, prefix.replace("--block", "")))
                capture = self.source.index("ray_pid=$!", self.source.index(prefix))
                self.reject(self.source[:capture] + self.source[capture:].replace("ray_pid=$!", "ray_pid=$$", 1))

    def test_global_cleanup_rejected_but_comments_and_heredocs_ignored(self):
        for command in ('"$ray" stop --force', '/opt/nemotronvc/venvs/driver/bin/ray stop --force',
                        'pkill -f ray', 'killall raylet', '/usr/bin/pkill -f ray'):
            with self.subTest(command=command):
                self.reject(self.source.replace('cleanup() {', 'cleanup() {\n ' + command))
                ports.validate_launcher(self.source + '\n# ' + command + '\n')
                ports.validate_launcher(self.source + '\ncat <<\'DOC\'\n' + command + '\nDOC\n')

    def test_cleanup_cannot_signal_other_targets_or_omit_wait(self):
        for target in ('"$$"', '"-$ray_pid"', '"$other_pid"'):
            with self.subTest(target=target):
                self.reject(self.source.replace('kill -TERM "$ray_pid"', 'kill -TERM ' + target))
        self.reject(self.source.replace('wait "$ray_pid"', 'true'))

    def test_ready_must_follow_capture_and_verify_expected_cluster(self):
        self.reject(self.source.replace('ray_pid=$!\n', 'ray_pid=$!\n sleep 900\n', 1))
        self.reject(self.source.replace('--expected-address "$head_address"', ''))
        self.reject(self.source.replace('--supervisor-pid "$ray_pid"', '--supervisor-pid "$$"'))


if __name__ == "__main__":
    unittest.main()
