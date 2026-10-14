#!/usr/bin/env python3
"""Tune mamba_ssm SSD Triton kernels ONCE on the target GPU with the per-rank Mamba shapes of our student, and write the
pin table read by nemo_rl/models/megatron/mamba_numerics.py (megatron_cfg.mamba_triton_pin_table).

Per-rank shapes (TP4 x CP2 -> Megatron's Mamba CP splits heads): nheads 64/8 = 8, ngroups 8/8 = 1, headdim 64,
d_state 128, chunk 128, conv width 4, BF16 activations, FP32 states (state_dtype), packed sequences (seq_idx).
Both Megatron paths are exercised forward + backward: mamba_split_conv1d_scan_combined (mem-eff, training default) and
mamba_chunk_scan_combined (unfused). Afterwards every Autotuner's chosen config is recorded; a pinned re-run checks
that outputs are bitwise identical to the autotuned run.
Run with the mcore venv python inside the training image on one GPU.
Usage: tune_ssd.py --out TABLE.json [--nheads 8 --ngroups 1 --headdim 64 --dstate 128 --chunk 128 --seqlen 16384]
       tune_ssd.py --probe REPORT.json [--pin TABLE.json]   (one process per GPU: chosen configs + output hash, to
       compare across GPUs/ranks with and without pinning)
"""
import argparse
import importlib.util
import json
import os

import torch


def load_numerics():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "..", "nemo_rl", "models", "megatron", "mamba_numerics.py")
    spec = importlib.util.spec_from_file_location("mamba_numerics", os.path.realpath(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_inputs(a, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev, bf = "cuda", torch.bfloat16
    h, p, n, ng, L = a.nheads, a.headdim, a.dstate, a.ngroups, a.seqlen
    d_inner = h * p
    lens = [L // 4, L // 8, L // 2, L - L // 4 - L // 8 - L // 2]
    seq_idx = torch.cat([torch.full((x,), i, dtype=torch.int32) for i, x in enumerate(lens)]).to(dev)[None]
    r = lambda *s, scale=1.0, dt=bf: (torch.randn(*s, generator=g, device=dev) * scale).to(dt)
    unfused = dict(x=r(1, L, h, p), dt=r(1, L, h, scale=0.5), A=-torch.rand(h, generator=g, device=dev).float() * 4 - 0.5,
                   B=r(1, L, ng, n, scale=0.3), C=r(1, L, ng, n, scale=0.3), D=r(h, dt=torch.float32),
                   z=r(1, L, h, p), dt_bias=r(h, scale=0.1, dt=torch.float32), seq_idx=seq_idx)
    dim = d_inner + 2 * ng * n
    fused = dict(zxbcdt=r(1, L, 2 * d_inner + 2 * ng * n + h, scale=0.5), conv1d_weight=r(dim, 4, scale=0.3),
                 conv1d_bias=r(dim, scale=0.1), dt_bias=r(h, scale=0.1, dt=torch.float32),
                 A=-torch.rand(h, generator=g, device=dev).float() * 4 - 0.5, D=r(h, dt=torch.float32), seq_idx=seq_idx)
    for t in list(unfused.values()) + list(fused.values()):
        if t.is_floating_point():
            t.requires_grad_(True)
    return unfused, fused


def run_once(a, seed):
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined

    u, f = make_inputs(a, seed)
    y1 = mamba_chunk_scan_combined(u["x"], u["dt"], u["A"], u["B"], u["C"], a.chunk, D=u["D"], z=u["z"],
                                   dt_bias=u["dt_bias"], seq_idx=u["seq_idx"], dt_softplus=True,
                                   state_dtype=torch.float32)
    y1.float().square().mean().backward()
    y2 = mamba_split_conv1d_scan_combined(f["zxbcdt"], f["conv1d_weight"], f["conv1d_bias"], f["dt_bias"], f["A"],
                                          f["D"], a.chunk, seq_idx=f["seq_idx"], activation="silu", headdim=a.headdim,
                                          ngroups=a.ngroups, norm_before_gate=False, state_dtype=torch.float32)
    y2.float().square().mean().backward()
    torch.cuda.synchronize()
    fwd = [y1.detach(), y2.detach()]
    grads = [t.grad.detach() for t in list(u.values()) + list(f.values())
             if isinstance(t, torch.Tensor) and t.grad is not None]
    return fwd, grads


def max_rel(xs, ys):
    return max(((x.float() - y.float()).norm() / (y.float().norm() + 1e-12)).item() for x, y in zip(xs, ys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--probe")
    ap.add_argument("--pin")
    ap.add_argument("--nheads", type=int, default=8)
    ap.add_argument("--ngroups", type=int, default=1)
    ap.add_argument("--headdim", type=int, default=64)
    ap.add_argument("--dstate", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--seqlen", type=int, default=16384)
    a = ap.parse_args()
    import triton
    import mamba_ssm

    num = load_numerics()
    if a.probe:
        import hashlib
        if a.pin:
            num.pin_triton_configs(a.pin)
            num.check_pinned()
        import time
        fwd, grads = run_once(a, 0)             # first call (autotunes unless deterministic mode pinned configs)
        times = []
        for i in range(3):
            t0 = time.perf_counter()
            run_once(a, 100 + i)
            times.append(time.perf_counter() - t0)
        h, hg = hashlib.sha256(), hashlib.sha256()
        for t in fwd:
            h.update(t.float().cpu().numpy().tobytes())
        for t in grads:
            hg.update(t.float().cpu().numpy().tobytes())
        torch.save({"fwd": [t.cpu() for t in fwd], "grads": [t.cpu() for t in grads]}, a.probe + ".pt")
        chosen = {name: str(sorted(tuner.cache.items(), key=lambda kv: str(kv[0]))[0][1]) if getattr(tuner, "cache", None)
                  else (str(tuner.configs[0]) if len(tuner.configs) == 1 else None) for name, tuner in num.iter_autotuners()}
        from mamba_ssm.utils.determinism import use_deterministic_mode
        multi = [name for name, tuner in num.iter_autotuners() if len(tuner.configs) > 1]
        json.dump({"gpu_index": torch.cuda.current_device(), "pinned": bool(a.pin),
                   "deterministic_mode": use_deterministic_mode(), "autotuners_with_several_configs": len(multi),
                   "seconds_per_fwd_bwd_median": sorted(times)[1], "forward_sha256": h.hexdigest(), "grads_sha256": hg.hexdigest(),
                   "chosen": chosen}, open(a.probe, "w"), indent=1)
        return
    tuned = run_once(a, 0)          # autotunes every kernel on this GPU
    run_once(a, 1)
    kernels, untouched = {}, []
    for name, tuner in num.iter_autotuners():
        cache = getattr(tuner, "cache", {}) or {}
        if not cache:
            untouched.append(name)
            continue
        if len(set(map(str, cache.values()))) > 1:
            print("[tune] note: %s chose different configs for keys %s" % (name, list(cache)))
        key, cfg = sorted(cache.items(), key=lambda kv: str(kv[0]))[0]
        kernels[name] = {"config": num.config_to_dict(cfg), "key": [str(x) for x in key], "n_candidates": len(tuner.configs)}
    table = {"kernels": kernels, "untouched": untouched, "triton": triton.__version__, "mamba_ssm": mamba_ssm.__version__,
             "gpu": torch.cuda.get_device_name(), "dims": vars(a)}
    with open(a.out, "w") as fh:
        json.dump(table, fh, indent=1)
    # pinned re-run must reproduce the autotuned run bit for bit (same configs, same inputs)
    num.pin_triton_configs(a.out)
    num.check_pinned()
    pinned = run_once(a, 0)
    fwd_same = all(torch.equal(x, y) for x, y in zip(tuned[0], pinned[0]))
    grad_same = all(torch.equal(x, y) for x, y in zip(tuned[1], pinned[1]))
    print(json.dumps({"tuned_kernels": len(kernels), "untouched": untouched, "pinned_rerun_forward_bitwise_equal": fwd_same,
                      "pinned_rerun_grads_bitwise_equal": grad_same, "grads_max_rel_diff": max_rel(pinned[1], tuned[1]),
                      "gpu": table["gpu"], "triton": table["triton"], "mamba_ssm": table["mamba_ssm"]}))
    if not fwd_same:
        raise SystemExit("pinned re-run forward differs from the autotuned run")


if __name__ == "__main__":
    main()
