# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Attach the Gym audio sidechannel to learner trajectories."""

import base64
import io
from pathlib import Path


def _load_waveform(source, sampling_rate):
    import soundfile

    if source.startswith("data:audio/"):
        header, payload = source.split(",", 1)
        if not header.endswith(";base64"):
            raise ValueError("Gym audio data must be a base64 audio URI")
        source = io.BytesIO(base64.b64decode(payload, validate=True))
    elif not Path(source).is_absolute():
        raise ValueError("Gym learner audio_path must be absolute")
    waveform, actual_rate = soundfile.read(source, dtype="float32")
    if actual_rate != sampling_rate:
        raise ValueError(
            f"Gym audio training requires {sampling_rate} Hz audio; got {actual_rate}. "
            "Resample the shared input file once before generation and training."
        )
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    return waveform


def gym_audio_model_inputs(row, processor, prompt_token_ids):
    """Decode the same metadata audio that Gym sends to the policy server.

    This is the initial audio sidechannel, attached once on the first trainable
    turn. Later prompt deltas retain the initial media in the message log.
    Require exact per-clip placeholder counts before admitting learner inputs.
    Image-only and text-only requests keep their existing path.
    """
    metadata = row.get("responses_create_params", {}).get("metadata") or {}
    keys = [
        key for key in ("audio_data", "audio_path", "audio_paths") if metadata.get(key)
    ]
    if not keys:
        return {}
    if len(keys) != 1:
        raise ValueError("Set exactly one Gym audio metadata field")
    extractor = getattr(processor, "feature_extractor", None)
    if not callable(getattr(extractor, "audio_token_count", None)):
        raise ValueError("Gym audio training requires the Bridge NeMo audio processor")
    value = metadata[keys[0]]
    sources = value if keys[0] == "audio_paths" else [value]
    if not isinstance(sources, list) or not all(
        isinstance(source, str) for source in sources
    ):
        raise ValueError("Gym audio_paths must be a list of paths")
    waves = [_load_waveform(source, extractor.sampling_rate) for source in sources]
    expected = [extractor.audio_token_count(len(wave)) for wave in waves]
    observed = []
    in_run = False
    for token in prompt_token_ids:
        if int(token) == processor.audio_token_id:
            if not observed or not in_run:
                observed.append(0)
            observed[-1] += 1
            in_run = True
        else:
            in_run = False
    if observed != expected:
        raise ValueError(
            f"Gym audio rollout/learner token mismatch: {observed=} {expected=}"
        )
    return extractor(waves)
