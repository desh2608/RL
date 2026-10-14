# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU Ray/endpoint regression for mapping saved GPU tags to physical nodes."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from nemo_rl.utils.logger import RayGpuMonitorLogger


class GPUNodeMappingTest(unittest.TestCase):
    nodes = [
        {"NodeID": "a", "NodeManagerAddress": "10.0.0.1", "NodeManagerHostname": "host-a", "MetricsExportPort": 12110},
        {"NodeID": "b", "NodeManagerAddress": "10.0.0.2", "NodeManagerHostname": "host-b", "MetricsExportPort": 12110},
    ]

    @staticmethod
    def response(url, timeout):
        assert timeout == 5.0
        value = 11 if "10.0.0.1:" in url else 22
        return SimpleNamespace(status_code=200, text=(
            'ray_node_gpus_utilization{GpuIndex="0",GpuDeviceName="fake"} ' + str(value) + '\n'))

    def test_first_and_changed_order_record_exact_parser_indices_without_repeating(self):
        with TemporaryDirectory() as tmp:
            monitor = RayGpuMonitorLogger(10, 10, "ray", "ray_step", SimpleNamespace(base_log_dir=tmp))
            with patch("nemo_rl.utils.logger.ray.is_initialized", return_value=True), \
                 patch("nemo_rl.utils.logger.ray.nodes", side_effect=[self.nodes, self.nodes, self.nodes[::-1]]), \
                 patch("nemo_rl.utils.logger.requests.get", side_effect=self.response) as requests, \
                 patch("nemo_rl.utils.logger.time.time", side_effect=[100.0, 200.0]):
                first = monitor._collect_metrics()
                second = monitor._collect_metrics()
                third = monitor._collect_metrics()
            self.assertEqual(first, {"node.0.gpu.0.util": 11.0, "node.1.gpu.0.util": 22.0})
            self.assertEqual(first, second)
            self.assertEqual(third, {"node.0.gpu.0.util": 22.0, "node.1.gpu.0.util": 11.0})
            records = [json.loads(line) for line in (Path(tmp)/"gpu_node_mapping.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 2)
            self.assertEqual([r["timestamp"] for r in records], [100.0, 200.0])
            self.assertEqual([n["node_id"] for n in records[0]["nodes"]], ["a", "b"])
            self.assertEqual([n["node_id"] for n in records[1]["nodes"]], ["b", "a"])
            self.assertEqual(records[1]["nodes"][0], {"node_index": 0, "node_id": "b", "ip": "10.0.0.2", "hostname": "host-b", "metrics_endpoint": "10.0.0.2:12110"})
            self.assertEqual([call.args[0] for call in requests.call_args_list],
                [f"http://{node['NodeManagerAddress']}:12110/metrics" for nodes in (self.nodes, self.nodes, self.nodes[::-1]) for node in nodes])

    def test_missing_parent_does_not_change_sampling(self):
        monitor = RayGpuMonitorLogger(10, 10, "ray", "ray_step")
        with patch("nemo_rl.utils.logger.ray.is_initialized", return_value=True), \
             patch("nemo_rl.utils.logger.ray.nodes", return_value=self.nodes), \
             patch("nemo_rl.utils.logger.requests.get", side_effect=self.response):
            self.assertEqual(monitor._collect_metrics()["node.0.gpu.0.util"], 11.0)

    def test_mapping_write_failure_does_not_disable_metrics(self):
        monitor = RayGpuMonitorLogger(10, 10, "ray", "ray_step", SimpleNamespace(base_log_dir="/unwritable"))
        with patch("nemo_rl.utils.logger.ray.is_initialized", return_value=True), \
             patch("nemo_rl.utils.logger.ray.nodes", return_value=self.nodes), \
             patch("nemo_rl.utils.logger.requests.get", side_effect=self.response), \
             patch("nemo_rl.utils.logger.os.makedirs", side_effect=PermissionError("test")):
            self.assertEqual(monitor._collect_metrics()["node.1.gpu.0.util"], 22.0)
        self.assertIsNone(monitor._last_node_mapping)


if __name__ == "__main__":
    unittest.main(verbosity=2)
