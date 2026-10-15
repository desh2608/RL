#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Link unchanged HF weights and ask Bridge to package the RoPE runtime."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import shutil
import sys
import types


def stage(source: Path, destination: Path, bridge_root: Path) -> None:
    source, destination = source.resolve(), destination.resolve()
    if destination == source or source in destination.parents:
        raise ValueError("Choose a destination outside the source checkpoint")
    config = json.loads((source / "config.json").read_text())
    sound = config.get("sound_config") or {}
    if (sound.get("encoder_config") or {}).get("architecture") != "rope_transformer":
        raise ValueError("Expected the saved rope_transformer encoder configuration")
    index = json.loads((source / "model.safetensors.index.json").read_text())
    weights = set(index["weight_map"].values())
    for name in weights:
        if Path(name).name != name or not (source / name).is_file():
            raise ValueError(f"Expected an existing flat HF weight shard: {name!r}")
    family = bridge_root.resolve() / "src/megatron/bridge/models/nemotron_omni"
    if not (family / "nemo_audio_export.py").is_file():
        raise FileNotFoundError("Fetch Bridge with --rope-audio before staging")
    destination.mkdir(parents=True, exist_ok=False)
    for path in source.iterdir():
        if not path.is_file():
            continue
        if path.name in weights:
            (destination / path.name).symlink_to(path.resolve())
        elif path.suffix in {
            ".json",
            ".py",
            ".txt",
            ".model",
            ".jinja",
            ".spm",
            ".tiktoken",
            ".bpe",
            ".yaml",
            ".yml",
        }:
            shutil.copy2(path, destination / path.name)
    # Loading this package directly avoids importing the full GPU training stack.
    # The packaging and encoder contract validation remain owned by Bridge.
    package = types.ModuleType("_voicechat_bridge_audio_staging")
    package.__path__ = [str(family)]
    sys.modules[package.__name__] = package
    helper = importlib.import_module(package.__name__ + ".nemo_audio_export")
    model_module = config["auto_map"]["AutoModelForCausalLM"].rsplit(".", 1)[0]
    helper.export_nemo_audio_assets(
        destination,
        destination / (model_module + ".py"),
        sound_model_type="nemo_transformer",
    )
    (destination / "STAGING_COMPLETE.json").write_text(
        json.dumps(
            {
                "source_hf": str(source),
                "bridge_root": str(bridge_root.resolve()),
                "weight_shards_linked": len(weights),
                "weight_conversion": False,
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument(
        "--bridge-root",
        type=Path,
        default=(
            Path(__file__).resolve().parents[2]
            / "3rdparty/Megatron-Bridge-workspace/Megatron-Bridge"
        ),
    )
    args = parser.parse_args()
    if sys.version_info < (3, 10):
        parser.error("Use the container driver Python or Python 3.10+ with Torch")
    stage(args.source, args.destination, args.bridge_root)
    print(args.destination.resolve())


if __name__ == "__main__":
    main()
