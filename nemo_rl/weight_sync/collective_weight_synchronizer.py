# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NCCL collective weight synchronizer for non-colocated deployments.

Handles weight transfer between policy and generation workers running on
separate GPU clusters using NCCL collective communication. The policy
broadcasts its weights, and generation workers receive them via the
established NCCL process group.

Lifecycle per sync:
  1. policy.broadcast_weights_for_collective()    -- send via NCCL
     generation.update_weights_from_collective()  -- receive via NCCL
  2. Verify transfer success

No offload/restore steps are needed since policy and generation run on
separate GPUs with dedicated memory.
"""

from contextlib import nullcontext
from typing import Any, Optional

import ray

from nemo_rl.utils.timer import Timer
from nemo_rl.weight_sync.interfaces import WeightSynchronizer


def pause_collective_refit_if_needed(generation: Any) -> bool:
    """Join the backend's optional rendezvous before any collective launches."""
    pause = getattr(generation, "pause_for_collective_refit", None)
    if pause is None:
        return False
    if not callable(pause):
        raise TypeError("Collective refit pause hook is not callable")
    paused = pause()
    if type(paused) is not bool:
        raise TypeError("Collective refit pause hook must return bool")
    if paused:
        for method in ("accept_collective_refit_results", "resume_after_collective_refit"):
            if not callable(getattr(generation, method, None)):
                raise RuntimeError(f"Paused collective refit is missing {method}")
    return paused


def collective_refit_update_succeeded(generation: Any, results: Any, paused: bool) -> bool:
    """Require strict backend ACKs when paused; retain legacy defaults otherwise."""
    if paused:
        accepted = generation.accept_collective_refit_results(results)
        if accepted is not True:
            raise RuntimeError("Paused collective refit did not accept every receiver")
        return True
    return all(result for result in results if result is not None)


class CollectiveWeightSynchronizer(WeightSynchronizer):
    """Weight synchronizer using NCCL collectives for non-colocated deployments.

    Policy and generation workers run on separate GPU clusters. Weights are
    synchronized via NCCL broadcast over a pre-established process group.

    Args:
        policy: Policy object implementing ColocatablePolicyInterface.
        generation: Generation object implementing GenerationInterface.
        train_cluster: RayVirtualCluster for the training workers, used to
            obtain the master address/port and world size for collective init.
        inference_cluster: RayVirtualCluster for the inference workers.
    """

    def __init__(
        self,
        policy: Any,
        generation: Any,
        train_cluster: Any,
        inference_cluster: Any,
    ):
        self._policy = policy
        self._generation = generation
        self._train_cluster = train_cluster
        self._inference_cluster = inference_cluster
        self._stale = True

    def sync_weights(
        self,
        *,
        timer: Optional[Timer] = None,
        kv_scales: Optional[dict[str, float]] = None,
    ) -> None:
        timer_context = (
            timer.time("prepare_for_generation/transfer_and_update_weights")
            if timer is not None
            else nullcontext()
        )
        with timer_context:
            # A failed transfer remains stale and must never resume partial weights.
            self._stale = True
            paused = pause_collective_refit_if_needed(self._generation)

            sender_spec = self._generation.get_collective_sender_spec()
            futures_train = self._policy.broadcast_weights_for_collective(
                kv_scales=kv_scales,
                buffer_size_bytes=sender_spec.buffer_size_bytes,
                num_buffers=sender_spec.num_buffers,
            )
            futures_inference = self._generation.update_weights_from_collective()

            ray.get(futures_train)
            results = ray.get(futures_inference)
            update_success = collective_refit_update_succeeded(
                self._generation, results, paused
            )

            if not update_success:
                raise RuntimeError(
                    "Weight transfer failed during NCCL collective sync. "
                    "This often indicates an issue with the NCCL process group "
                    "or the generation backend worker."
                )

            # Never resume from a failed/partial pause or transfer. Failure
            # propagates to the owning job lifecycle for teardown instead.
            if paused:
                self._generation.resume_after_collective_refit()

        self._stale = False

    @property
    def is_stale(self) -> bool:
        return self._stale

    def init_communicator(self) -> None:
        # prepare_refit_info is called before init_collective. This matches
        # distillation.py ordering. Neither call depends on the other today,
        # but we document this as the canonical ordering for future reference.
        state_dict_info = self._policy.prepare_refit_info()
        self._generation.prepare_refit_info(state_dict_info)

        ip, port = self._train_cluster.get_master_address_and_port()
        train_world_size = self._train_cluster.world_size()
        inference_world_size = self._generation.get_inference_world_size()
        if inference_world_size is None:
            inference_world_size = self._inference_cluster.world_size()
        world_size = train_world_size + inference_world_size

        sender_spec = self._generation.get_collective_sender_spec()
        futures_train = self._policy.init_collective(
            ip,
            port,
            world_size,
            train_world_size=train_world_size,
            nccl_peer=sender_spec.nccl_peer,
        )
        futures_inference = self._generation.init_collective(
            ip, port, world_size, train_world_size=train_world_size
        )
        ray.get(futures_train + futures_inference)

    def shutdown(self) -> None:
        # The NCCL process group lifecycle is managed by Ray actor teardown.
        # Explicit destroy_process_group() is not needed here because the
        # workers that own the group are destroyed when the cluster shuts down.
        pass
