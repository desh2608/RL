# Third-party trees

| Dependency | In git? | Base | Local changes |
|---|---|---|---|
| Gym (`3rdparty/Gym-workspace/Gym`) | yes (sources, no `.git`; upstream test files over 3 MB listed in `SKIPPED_LARGE_FILES.txt`) | NVIDIA-NeMo/Gym `5dd5408` (super-v3.5-posttraining) | in the repo code; see the file list at the end of `../docs/BUGS_FIXED.md` |
| Megatron-Bridge (`3rdparty/Megatron-Bridge-workspace/Megatron-Bridge`) | no (~170 MB without `.git`, incl. nested Megatron-LM) | NVIDIA-NeMo/Megatron-Bridge `3961f399` | `megatron-bridge-patches/` (4): Omni packing layout, two-way CP text packing, verified HDO restore, pinned MCore fix |
| Megatron-LM (Megatron-Bridge `3rdparty/Megatron-LM`) | no | `f2f0f7bfd` | `megatron-lm-patches/` (3): HDO master restore, FP32 CPU copy restore, keep restored step |
| Automodel | not used | — | — |

```bash
bash voicechat/third_party/fetch_third_party.sh               # default: copy the exact tree used for training from the shared cs-oci path (no .git)
bash voicechat/third_party/fetch_third_party.sh --upstream    # clone the upstream base commits and apply the patches (needs network and public base commits)
bash voicechat/third_party/fetch_third_party.sh --large-files # also restore the Gym files skipped by size
```

`scripts/runtime_env.sh` puts `/workspace/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src` and its nested
`3rdparty/Megatron-LM` on PYTHONPATH inside the container, so the tree must exist. Both directories are git-ignored.
