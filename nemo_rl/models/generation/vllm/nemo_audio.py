# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Load Bridge-owned RoPE runtime assets in generation-only environments."""

from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module


def is_nemo_audio_config(config) -> bool:
    """Identify checkpoints declaring the Bridge NeMo audio contract."""
    sound = getattr(config, "sound_config", None)
    sound_type = (
        sound.get("model_type")
        if isinstance(sound, dict)
        else getattr(sound, "model_type", None)
    )
    return sound_type == "nemo_transformer"


def install_nemo_audio(model_name: str) -> bool:
    """Install the exported native-vLLM adapters before processor/model creation."""
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if not is_nemo_audio_config(config):
        return False
    install = get_class_from_dynamic_module(
        "nemo_audio_vllm.install_vllm_audio", model_name
    )
    install()
    return True


def wrap_nemo_audio_processor(processor, model_name: str):
    """Adapt HF processor outputs to Bridge's mel/valid-length learner inputs."""
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if not is_nemo_audio_config(config):
        return processor
    adapter = get_class_from_dynamic_module(
        "nemo_audio_processor.NemoAudioProcessorAdapter", model_name
    )
    return adapter(processor, config.sound_config)
