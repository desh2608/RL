# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU checks for loading Bridge assets without a vLLM/MCore dependency."""

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = (
    Path(os.environ["NEMO_RL_ROOT"])
    if "NEMO_RL_ROOT" in os.environ
    else Path(__file__).resolve().parents[3]
)
spec = importlib.util.spec_from_file_location(
    "_nemo_audio_helpers", ROOT / "nemo_rl/models/generation/vllm/nemo_audio.py"
)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


@pytest.mark.parametrize(
    "sound",
    [
        {"model_type": "nemo_transformer"},
        SimpleNamespace(model_type="nemo_transformer"),
    ],
)
def test_checkpoint_declares_audio_backend(sound):
    assert helpers.is_nemo_audio_config(SimpleNamespace(sound_config=sound))


def test_parakeet_avoids_runtime_install_and_processor_wrap(monkeypatch):
    config = SimpleNamespace(sound_config=SimpleNamespace(model_type="parakeet"))
    monkeypatch.setattr(
        helpers.AutoConfig, "from_pretrained", Mock(return_value=config)
    )
    load_runtime = Mock()
    monkeypatch.setattr(helpers, "get_class_from_dynamic_module", load_runtime)
    processor = object()
    assert helpers.install_nemo_audio("model") is False
    assert helpers.wrap_nemo_audio_processor(processor, "model") is processor
    load_runtime.assert_not_called()


def test_generation_calls_bridge_exported_installer(monkeypatch):
    config = SimpleNamespace(sound_config={"model_type": "nemo_transformer"})
    monkeypatch.setattr(
        helpers.AutoConfig, "from_pretrained", Mock(return_value=config)
    )
    installer = Mock()
    loader = Mock(return_value=installer)
    monkeypatch.setattr(helpers, "get_class_from_dynamic_module", loader)
    assert helpers.install_nemo_audio("model") is True
    loader.assert_called_once_with("nemo_audio_vllm.install_vllm_audio", "model")
    installer.assert_called_once_with()


def test_learner_calls_bridge_exported_processor_adapter(monkeypatch):
    config = SimpleNamespace(sound_config={"model_type": "nemo_transformer"})
    monkeypatch.setattr(
        helpers.AutoConfig, "from_pretrained", Mock(return_value=config)
    )
    adapter = Mock(return_value=object())
    monkeypatch.setattr(
        helpers, "get_class_from_dynamic_module", Mock(return_value=adapter)
    )
    processor = object()
    assert helpers.wrap_nemo_audio_processor(processor, "model") is adapter.return_value
    adapter.assert_called_once_with(processor, config.sound_config)


def test_audio_adapter_preserves_source_placeholder_contract():
    # Extract the small predicates to keep this check independent of GPU/Ray
    # imports from the larger data module.
    import ast

    source = ROOT / "nemo_rl/data/multimodal_utils.py"
    tree = ast.parse(source.read_text())
    names = {"_processor_type_name", "uses_image_placeholder", "uses_fixed_tile_image_processor"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {
        "Any": object,
        "_PLACEHOLDER_STYLE_PROCESSOR_NAMES": {"NemotronH_Nano_Omni_Reasoning_V3Processor"},
        "_FIXED_TILE_IMAGE_PROCESSOR_NAMES": {"NemotronNanoVLV2Processor"},
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    original = type("NemotronH_Nano_Omni_Reasoning_V3Processor", (), {})()
    adapter = type("NemoAudioProcessorAdapter", (), {} )()
    adapter.processor = original
    assert namespace["uses_image_placeholder"](adapter)
    assert not namespace["uses_fixed_tile_image_processor"](adapter)
