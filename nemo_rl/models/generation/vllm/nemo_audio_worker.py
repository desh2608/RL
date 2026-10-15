# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Initialize exported Bridge audio adapters in each vLLM model process."""

from vllm.v1.worker.gpu_worker import Worker

from nemo_rl.models.generation.vllm.nemo_audio import install_nemo_audio


class NemoAudioVllmWorker(Worker):
    """Keep vLLM worker behavior while selecting Bridge's declared audio tower."""

    def load_model(self, *args, **kwargs):
        install_nemo_audio(self.model_config.model)
        return super().load_model(*args, **kwargs)
