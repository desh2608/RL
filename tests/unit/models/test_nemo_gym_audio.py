# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""CPU regression checks for Gym rollout/learner audio parity."""

import importlib.util
import base64
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

ROOT = (
    Path(os.environ["NEMO_RL_ROOT"])
    if "NEMO_RL_ROOT" in os.environ
    else Path(__file__).resolve().parents[3]
)
spec = importlib.util.spec_from_file_location(
    "gym_audio", ROOT / "nemo_rl/environments/nemo_gym_audio.py"
)
audio = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audio)


def test_text_only_keeps_old_path():
    assert audio.gym_audio_model_inputs({}, None, [1, 2]) == {}


@pytest.mark.parametrize(
    "key,value",
    [
        ("audio_path", "/clip.wav"),
        ("audio_data", "data:audio/wav;base64,AA=="),
        ("audio_paths", ["/a.wav", "/b.wav"]),
    ],
)
def test_sidechannel_decodes_and_requires_matching_clip_runs(monkeypatch, key, value):
    sources = value if isinstance(value, list) else [value]
    waves = {source: list(range(16 + i * 8)) for i, source in enumerate(sources)}
    load = Mock(side_effect=lambda source, sr: waves[source])
    monkeypatch.setattr(audio, "_load_waveform", load)
    extractor = Mock(return_value={"sound_clips": "mels", "sound_length": "lengths"})
    extractor.sampling_rate = 16000
    extractor.audio_token_count = lambda n: n // 8
    processor = SimpleNamespace(feature_extractor=extractor, audio_token_id=99)
    tokens = [1, 99, 99, 2] + ([99, 99, 99, 3] if len(sources) == 2 else [])
    row = {"responses_create_params": {"metadata": {key: value}}}
    assert audio.gym_audio_model_inputs(row, processor, tokens) == {
        "sound_clips": "mels",
        "sound_length": "lengths",
    }
    extractor.assert_called_once_with(list(waves.values()))
    assert load.call_count == len(sources)
    with pytest.raises(ValueError, match="token mismatch"):
        audio.gym_audio_model_inputs(row, processor, tokens + [99])


def test_ambiguous_metadata_rejected():
    row = {
        "responses_create_params": {
            "metadata": {"audio_path": "/x.wav", "audio_data": "data:x"}
        }
    }
    with pytest.raises(ValueError, match="exactly one"):
        audio.gym_audio_model_inputs(row, None, [])


def test_missing_bridge_processor_rejected():
    row = {"responses_create_params": {"metadata": {"audio_path": "/x.wav"}}}
    with pytest.raises(ValueError, match="Bridge NeMo"):
        audio.gym_audio_model_inputs(row, None, [])


def test_waveform_decode_matches_pcm_for_paths_and_inline_audio(tmp_path):
    import soundfile

    path = tmp_path / "speech.wav"
    signal = np.array([0, 1024, -32768, 32767], dtype=np.int16)
    soundfile.write(path, signal, 16000, subtype="PCM_16")
    expected = signal.astype(np.float32) / 32768
    np.testing.assert_array_equal(audio._load_waveform(str(path), 16000), expected)
    uri = "data:audio/wav;base64," + base64.b64encode(path.read_bytes()).decode()
    np.testing.assert_array_equal(audio._load_waveform(uri, 16000), expected)


def test_waveform_decode_rejects_different_sampling_rate(tmp_path):
    import soundfile

    path = tmp_path / "wrong_rate.wav"
    soundfile.write(path, np.zeros(1600, dtype=np.float32), 8000)
    with pytest.raises(ValueError, match="requires 16000 Hz"):
        audio._load_waveform(str(path), 16000)
