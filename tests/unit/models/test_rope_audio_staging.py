# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""CPU checks for the thin Bridge packaging entrypoint."""

import importlib.util
import json
import os
from pathlib import Path

import pytest

ROOT = (
    Path(os.environ["NEMO_RL_ROOT"])
    if "NEMO_RL_ROOT" in os.environ
    else Path(__file__).resolve().parents[3]
)
spec = importlib.util.spec_from_file_location(
    "rope_staging", ROOT / "voicechat/tools/stage_rope_audio_hf.py"
)
staging = importlib.util.module_from_spec(spec)
spec.loader.exec_module(staging)
BRIDGE = ROOT / "3rdparty/Megatron-Bridge-workspace/Megatron-Bridge"


def checkpoint(path):
    path.mkdir()
    config = {
        "auto_map": {
            "AutoConfig": "configuration.TestConfig",
            "AutoModelForCausalLM": "modeling.TestModel",
        },
        "sound_config": {
            "model_type": "nemo_transformer",
            "hidden_size": 16,
            "encoder_config": {
                "architecture": "rope_transformer",
                "self_attention_model": "rope",
                "pre_encode": "stacking",
                "n_mels": 4,
                "d_model": 16,
                "n_heads": 2,
                "n_layers": 2,
                "subsampling_factor": 4,
            },
            "preprocessor_config": {
                "features": 4,
                "normalize": "all_features",
                "dither": 0.0,
            },
        },
    }
    (path / "config.json").write_text(json.dumps(config))
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "weights.safetensors"}})
    )
    (path / "weights.safetensors").write_bytes(b"opaque unchanged weight fixture")
    (path / "configuration.py").write_text(
        "class SoundConfig:\n    def to_dict(self): return {}\nclass TestConfig: pass\n"
    )
    (path / "modeling.py").write_text("class TestModel: pass\n")
    (path / "audio_model.py").write_text("class OldEncoder: pass\n")
    return config


def test_staging_calls_real_bridge_and_preserves_source(tmp_path):
    source, dest = tmp_path / "source", tmp_path / "staged"
    config = checkpoint(source)
    before = {path.name: path.read_bytes() for path in source.iterdir()}
    staging.stage(source, dest, BRIDGE)
    assert {path.name: path.read_bytes() for path in source.iterdir()} == before
    assert (dest / "weights.safetensors").is_symlink()
    assert (dest / "weights.safetensors").resolve() == source / "weights.safetensors"
    assert (dest / "nemo_audio_vllm.py").is_file()
    assert (dest / "STAGING_COMPLETE.json").is_file()
    assert (
        json.loads((dest / "config.json").read_text())["sound_config"]
        == config["sound_config"]
    )
    assert (
        "from .nemo_audio_runtime import NemoTransformerSoundEncoder"
        in (dest / "audio_model.py").read_text()
    )


def test_staging_rejects_source_destination(tmp_path):
    source = tmp_path / "source"
    checkpoint(source)
    with pytest.raises(ValueError, match="outside"):
        staging.stage(source, source / "nested", BRIDGE)
    assert not (source / "nested").exists()


def test_staging_rejects_other_encoder(tmp_path):
    source = tmp_path / "source"
    config = checkpoint(source)
    config["sound_config"]["encoder_config"]["architecture"] = "transformer"
    (source / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="rope_transformer"):
        staging.stage(source, tmp_path / "staged", BRIDGE)
    assert not (tmp_path / "staged").exists()
