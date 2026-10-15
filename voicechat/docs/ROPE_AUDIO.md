# RoPE audio through Megatron Bridge

The learner imports `NemoTransformerAudioModel` from MCore. Megatron Bridge owns
HF config/weight conversion and the standalone HF/vLLM audio runtime. NeMo-RL
loads those exported assets for audio preprocessing and native vLLM generation;
its existing Bridge refit stream supplies updated HF weights. Evaluation
launchers and their conversion workflow are outside this change.

## Dependency setup

```bash
bash voicechat/third_party/fetch_third_party.sh --upstream --rope-audio
```

`--rope-audio` is opt-in. The validated Bridge `3961f399` and MCore `f2f0f7bfd`
base pins and all four Bridge / three MCore bug patches remain in use. Two
additional bundles extend only the Omni/audio code. The MCore bundle contains
upstream audio source from `a98aed553e25b4e86c59f50be54ac0ea4227b442`, already
containing RoPE merge `ca8429d421` (#7066). The standalone Bridge branch is based
on upstream main `2c75856da`; its own submodule already contains that code.

Both additive patches are checked before either is applied. A content-hash marker
makes repeat fetches idempotent. Changed patch bundles require a freshly fetched
third-party tree; do not edit a submitted run's source.

## Checkpoint contract

The HF model must declare `sound_config.model_type = "nemo_transformer"` and
contain nonempty `encoder_config` and `preprocessor_config` dictionaries copied
from the SFT checkpoint. Retain all saved dimensions, normalization, RoPE,
subsampling, and attention-mode fields. For RoPE, the normalized encoder config
requires:

- `architecture = "rope_transformer"`
- `self_attention_model = "rope"`
- `pre_encode = "stacking"` (`feature_stacking` is trunk's compatibility alias)
- `stacking_pad_mode = "zeros"`
- `attn_impl = "te"`

The encoder's `n_mels` must match the frontend's `features`, and `hidden_size`
must match `d_model`. Sampling rates must agree. Preserve `causal_mask` and
`left_context` from the checkpoint; RL does not randomly change the encoder's
attention mode. Incompatible architecture/frontend settings fail explicitly.

The encoder mapping is replicated across TP:

```text
MCore/Bridge sound_model.audio_model.encoder.*
HF           sound_encoder.audio_model.encoder.*
```

The existing sound projection is reused, with checkpoint-controlled bias and
projection epsilon (HF default `1e-5`). The historical Tau-Voice V2 HF architecture
label routes explicitly declared NeMo audio through the Omni provider/mappings.
Parakeet configs keep their existing path.

Bridge's HF export hook installs `nemo_audio_runtime.py`, its trunk-derived
encoder/frontend modules, `nemo_audio_processor.py`, and `nemo_audio_vllm.py`.
It also makes relative dependencies visible for a fresh Transformers module
cache. A pre-existing HF model directory needs the same asset packaging before
NeMo-RL can load its new adapters. The model's original Omni modeling,
configuration, tokenizer, and processor artifacts remain the source of the
outer model definition.

## Learner and generation inputs

Set `policy.is_vlm=true` for audio training. With Chen's generic
`examples/run_grpo.py` entrypoint, also copy `policy.tokenizer` into
`env.nemo_gym.tokenizer_config`: that entrypoint does not call the specialized
Gym runner's `setup_nemo_gym_config` helper. The explicit actor configuration
constructs the processor needed to attach learner audio tensors. A text-only
recipe without it cannot train on audio.

The learner adapter returns float32 `sound_clips` shaped `[clips, mel_frames,
mel_bins]` and integral `sound_length` with valid frame counts. Variable frame
widths are padded when assembling batches. The encoder returns `[clips,
encoder_steps, hidden]` plus valid output lengths, matching the Parakeet-facing
Bridge interface.

The frontend runs in eval mode, preserving the checkpoint's mel normalization.
Its window/filter buffers remain float32 even when the HF model uses BF16.
Waveforms are materialized individually before mel batching, matching MCore and
preserving saved fixed global normalization vectors (B22).
Frame stacking counts a final partial stack as one token and pads it with zeros.
A frozen learner tower stays in eval mode when its parent enters training mode.

For the fork's pinned native vLLM implementation, the exported adapter replaces
only audio architecture dispatch: it uses the same frontend and encoder keys,
then the existing vLLM sound projection. One full waveform is one clip, matching
the learner's unsegmented waveform processing. Dispatch is installed in the
engine process and in each model worker before construction. The runtime loads
Bridge's HF names during initial load and subsequent refit, with no checkpoint
converter in NeMo-RL or Megatron-LM. Generation environments need TE but do not
need Megatron-LM or Megatron Bridge imports to load the standalone audio files.
Gym audio sidechannels use SoundFile; supply absolute paths or audio data URIs
with audio already normalized to the model sampling rate (16kHz for the tested
checkpoint). The learner checks each clip’s placeholder count before training.

## Differences from the previous Tau-Voice HF encoder

| Aspect | Previous HF implementation | Trunk-aligned implementation |
| --- | --- | --- |
| Architecture selection | Always the legacy NeMo transformer | Explicit legacy / RoPE selection from saved config |
| Attention state layout | Legacy attention modules | RoPE fused `w_qkv`, optional per-head Q/K norms, trunk RoPE parameters |
| Normalization | Legacy transformer layout | Trunk embedding, block, and final normalization layout |
| Frame stacking | Legacy learned padding / `proj_out` | RoPE zero padding / `proj`; valid lengths use per-sample ceiling division |
| Forward execution | Dense legacy path | True packed audio execution and dense/packed public interfaces |
| Attention mode | Fixed legacy configuration | Trunk full-context/causal controls, including per-audio policy support |
| Runtime dependency | Legacy supported SDPA | RoPE uses TE, exactly as trunk requires |
| Audio preprocessing | Older vendored frontend; model dtype could affect buffers | Trunk frontend with float32 buffer preservation |

The wrapper-level API is small; the two encoder architectures have different
state layouts. Their weights must be loaded with their own saved architecture
configuration.

## Stage an existing HF checkpoint

After fetching the opt-in dependencies, run in the driver environment (Python
3.10+ with Torch). The destination must be new:

```bash
/opt/nemotronvc/venvs/driver/bin/python voicechat/tools/stage_rope_audio_hf.py \
  --source /absolute/path/to/original/hf \
  --destination /absolute/path/to/new/rl-hf
```

This copies small assets, links the original weight shards, and calls Bridge's
asset exporter. It leaves the source checkpoint unchanged. Retain the source
shards for as long as the staged view is used.

## Validated offline RL recipe

IAD job **14049277 completed with exit 0**: three real Chen offline GRPO updates,
HF export at step3, strict checkpoint resume, and a fourth real update. The
96 spoken static-v10 pivots retain Chen's original tools, gold and verifier.
Native vLLM rollouts, rewards, filtering, learner updates, Bridge weight refits,
cache resets, checkpoint save and optimizer restore all executed. All eight
generation ranks acknowledged initial and updated weights. No GRPO objective
or reward code was changed.

The tested model is `tau_nano35_lc6k_s3_jkyi40k_r5_gbs512_retry1` step7000.
The three-node A100 recipe uses:

- Generation: one node, vLLM0.27.1, TP1, node DP8, EP8.
- Learner: two nodes, TP4, CP1, EP8, ETP1, DP4.
- `policy.sequence_packing.enabled=false`; CP2 packed language attention had
  no supported backend on these A100s. The RoPE audio encoder retains trunk TE.
- P4 × K16, response global batch64, sequence limit8192, max new tokens2048.
- `policy.is_vlm=true` and explicit `env.nemo_gym.tokenizer_config` as above.
- `policy.generation.vllm_kwargs.max_num_seqs=16` and
  `policy.generation.async_global_ready_queue.max_inflight_per_replica=128`.
  Queue capacity must not exceed max sequences × node DP (16 × 8 here).

Chen's `smoke_check.py` accepted all three initial dumps. The fourth dump also
has64 active rows, four groups of16 distinct generations, mixed verifier
rewards, finite advantages and correct prompt/response masks. Gradient norms
were1.0314,0.7926,0.8777,0.7585. All32 optimizer restore receipts (16 ranks,
load and pre-update phases) report exact master weights and moments at step3.
Both sandbox ownership/exit gates passed. Resume generation needed the native
shutdown timeout's force-termination fallback; the outer job exited0.

The strict HF export retains all7,125 tensor keys. All490 audio tensors
(1,424,360,992 bytes) are bit-identical to the source; selected language weights
changed. Encoder/frontend configs are exactly preserved. Independent job
**14050505** cold-loaded that trained HF export in native vLLM and generated on
four real audio prompts. This establishes execution, not reward improvement or
online-RL acceptance. Evaluation launchers remain deferred.

### IAD reference assets and reproducibility

All following paths are relative to
`/lustre/fsw/portfolios/nemotron/projects/nemotron_speech_chat/users/deshr`:

```text
containers/voicechat/NemotronVC-Gym34-vllm0271-v1.sqsh
containers/voicechat/sandbox-x86_64-v1.sqsh
results/tau-voice-rl/chen-offline-e2e-20261015/
  preflight-20261015T163720Z/project/runs/smoke-v1-20261015T174804Z/
    SMOKE_PASS.json
    hf/step_3/
    checkpoints/step_4/
  native-audit-20261015T180800Z/AUDIT-14050072.json
  trained-hf-reload-20261015T182000Z/VLLM_PASS.json
  runtime/te-audio-tiler1-20261015T154200Z/site/
```

The frozen training source SHA256 is
`084e941f55da7fe9e0dffe177e49aadd6b5fdcd7927e150468ddc69581ac8a41`.
Inspect that run's frozen recipe and environment for the exact submission.
Use Chen's [offline workflow](OFFLINE_PIVOT_RL.md) and `make_run.py --project`
to freeze a fresh smoke for a colleague's checkout. The staging CLI and final
handoff notes were added after the accepted source freeze; training/runtime
code was unchanged by those additions.

The shared image has Torch2.11 in the learner and Torch2.13 in generation.
Generation needs the external overlay above: SoundFile and TE2.15 built from
source revision `42b840051647eef89761a16dfdff87e82bb253ab` against Torch2.13.
Retain its predecessor profiles referenced by symlinks. Put that site directory
first in generation's PYTHONPATH after sourcing the image environment, and
preload the matching image cuBLAS libraries **in generation only**:

```bash
export LD_PRELOAD=/opt/nemotronvc/venvs/mcore/lib/python3.13/site-packages/nvidia/cu13/lib/libcublasLt.so.13:/opt/nemotronvc/venvs/mcore/lib/python3.13/site-packages/nvidia/cu13/lib/libcublas.so.13
```

Do not share compiled TE extensions across Torch ABIs. These dependencies are
external assets, not a newly baked container. Native Mamba uses deterministic
float32 states and the image's fused path. The original image/fork version
mismatch handling is retained; no dependency fingerprint was rewritten.

Additional coverage: bit-for-bit native MCore versus standalone HF RoPE encoder
parity on actual checkpoint CUDA/TE outputs, 34 focused CPU contracts (Bridge,
RL, Gym and staging), and12 launcher checks. Earlier diagnostic jobs14019801 /
14020350 used fixed signed advantages and are separate from the native pass.
