#!/usr/bin/env python
"""GPU gradient-sync probe for the NeMo-RL Megatron policy worker (frozen run source).

Runs the real ``MegatronPolicyWorkerImpl`` (no Ray: ``ray.get_gpu_ids`` is shimmed) under torchrun,
builds it from a frozen run's ``config.yaml`` exactly like ``examples/run_grpo.py`` + ``lm_policy.Policy``,
and calls ``worker.train(fixed_batch, ClippedPGLossFn)`` once. Instruments:

* identity: the config object the mcore schedule reads (``get_model_config(worker.model)``) vs the
  Bridge provider (``worker.megatron_cfg.model``), the Float16 wrapper config and the inner HybridModel config;
  whether ``finalize_model_grads_func`` is set on each;
* a counting wrapper around the bound ``finalize_model_grads_func`` (calls per ``train()``);
* PRE (inside the wrapper, before the real finalize): linear sketches <g_local, r_k> over every trainable
  param's full local ``main_grad``, summed over the group finalize is supposed to reduce over
  (dense: DP x CP; SP-tagged dense: TP x DP x CP; expert: expert-DP)  -> expected post-sync sketch;
* POST (entry of ``optimizer.step()``, i.e. what the optimizer consumes): sketches and squared norm over
  the exact shard each rank's DistributedOptimizer reads (``_get_model_param_range_map(p)['param']``),
  summed over that optimizer's ``data_parallel_group``  -> full reduced local gradient sketch;
* TP consistency for TP-replicated dense params (identical full grads expected on all TP ranks).

Modes: ``normal`` (production code path) and ``nofinalize`` (negative control: the schedule's callback
is set to None before train, so no DP/SP reduction runs).  Cross-layout comparison (DP1 reference vs
DP4 production) is done offline by ``compare_gradsync.py`` from the per-rank stats files.
"""
import argparse, hashlib, json, os, sys, time

K = 4  # random projections per parameter


def log(msg):
    r = os.environ.get("RANK", "?")
    print(f"[gradsync rank={r}] {msg}", flush=True)


def seed_for(*parts):
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "little") & ((1 << 62) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=["normal", "nofinalize"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-seqs", type=int, default=96)
    ap.add_argument("--min-len", type=int, default=2048)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--data-seed", type=int, default=20261006)
    args = ap.parse_args()

    import ray
    local_rank = int(os.environ["LOCAL_RANK"])
    ray.get_gpu_ids = lambda: [local_rank]  # worker reads its GPU from Ray; torchrun gives LOCAL_RANK

    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from nemo_rl.utils.config import load_config
    from nemo_rl.algorithms.grpo import MasterConfig
    from nemo_rl.algorithms.utils import get_tokenizer
    from nemo_rl.models.generation import configure_generation_config
    from nemo_rl.algorithms.loss.loss_functions import ClippedPGLossFn
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from nemo_rl.distributed.named_sharding import NamedSharding
    from nemo_rl.models.policy.workers.megatron_policy_worker import MegatronPolicyWorkerImpl

    cfg = load_config(args.config)
    cfg = MasterConfig(**OmegaConf.to_container(cfg, resolve=True))
    policy_cfg = cfg.policy
    tokenizer = get_tokenizer(policy_cfg["tokenizer"])
    policy_cfg["generation"] = configure_generation_config(
        policy_cfg["generation"], tokenizer,
        has_refit_draft_weights=bool(policy_cfg["draft"]["enabled"]),
        trains_mtp=bool((policy_cfg.get("megatron_cfg") or {}).get("mtp_num_layers")),
    )
    mcfg = policy_cfg["megatron_cfg"]
    world = int(os.environ["WORLD_SIZE"])
    tp, cp, pp = mcfg["tensor_model_parallel_size"], mcfg["context_parallel_size"], mcfg["pipeline_model_parallel_size"]
    sharding = NamedSharding(layout=np.arange(world).reshape(pp, -1, cp, tp),
                             names=["pipeline_parallel", "data_parallel", "context_parallel", "tensor_parallel"])
    if cfg.loss_fn.force_on_policy_ratio:
        os.environ["NRL_IGNORE_TP_ACCURACY_CHECK"] = "1"  # grpo.setup does the same

    t0 = time.time()
    worker = MegatronPolicyWorkerImpl(policy_cfg, tokenizer, init_optimizer=True,
                                      init_reference_model=cfg.loss_fn.reference_policy_kl_penalty > 0,
                                      worker_sharding_annotations=sharding)
    log(f"worker init {time.time()-t0:.0f}s")

    from megatron.core import parallel_state as ps
    from megatron.core.utils import get_model_config
    rank = torch.distributed.get_rank()
    tp_rank, ep_rank = ps.get_tensor_model_parallel_rank(), ps.get_expert_model_parallel_rank()
    dp_size = worker.dp_size
    dp_rank = ps.get_data_parallel_rank()

    # ---- identity evidence
    sched_cfg = get_model_config(worker.model)
    provider = worker.megatron_cfg.model
    f16_cfg = worker._get_model_config()
    inner = worker.model
    while hasattr(inner, "module"):
        inner = inner.module
    lm = getattr(inner, "language_model", None)
    lm_cfg = getattr(lm, "config", None)
    ident = {
        "rank": rank, "world": world, "dp_size": dp_size, "tp_rank": tp_rank, "ep_rank": ep_rank,
        "model_type": type(worker.model).__name__, "inner_type": type(inner).__name__,
        "sched_cfg_is_provider": sched_cfg is provider,
        "sched_cfg_is_float16_cfg": sched_cfg is f16_cfg,
        "sched_cfg_is_language_model_cfg": (sched_cfg is lm_cfg) if lm_cfg is not None else None,
        "sched_finalize_set": sched_cfg.finalize_model_grads_func is not None,
        "sched_finalize_repr": repr(sched_cfg.finalize_model_grads_func)[:200],
        "provider_finalize_set": getattr(provider, "finalize_model_grads_func", None) is not None,
        "sched_no_sync_func": repr(sched_cfg.no_sync_func), "sched_grad_sync_func": repr(sched_cfg.grad_sync_func),
        "overlap_grad_reduce": worker.megatron_cfg.ddp.overlap_grad_reduce,
        "use_distributed_optimizer": worker.megatron_cfg.ddp.use_distributed_optimizer,
        "calculate_per_token_loss": sched_cfg.calculate_per_token_loss,
        "sequence_parallel": sched_cfg.sequence_parallel,
        "dp_cp_size": ps.get_data_parallel_world_size(with_context_parallel=True),
        "expt_dp_size": ps.get_expert_data_parallel_world_size(),
        "mode": args.mode,
    }

    # ---- parameter classification + random-projection seeds
    params = [(n, p) for n, p in worker.model.named_parameters() if p.requires_grad]
    tp_group = ps.get_tensor_model_parallel_group()
    dpcp_group = ps.get_data_parallel_group(with_context_parallel=True)
    tpdpcp_group = ps.get_tensor_and_data_parallel_group(with_context_parallel=True)
    edp_group = ps.get_expert_data_parallel_group()
    info = []
    for n, p in params:
        expert = not getattr(p, "allreduce", True)
        sp = bool(sched_cfg.sequence_parallel and getattr(p, "sequence_parallel", False))
        tp_part = bool(getattr(p, "tensor_model_parallel", False))
        if expert:
            coord, seed, grp = f"ep{ep_rank}", seed_for(n, "ep", ep_rank), edp_group
        elif tp_part:
            coord, seed, grp = f"tp{tp_rank}", seed_for(n, "tp", tp_rank), (tpdpcp_group if sp else dpcp_group)
        else:  # TP-replicated dense param: same projection on every TP rank
            coord, seed, grp = "rep", seed_for(n, "rep"), (tpdpcp_group if sp else dpcp_group)
        info.append(dict(name=n, p=p, expert=expert, sp=sp, tp_part=tp_part, coord=coord, seed=seed, pre_group=grp,
                          numel=p.numel()))
    ident["n_trainable"] = len(info)
    ident["n_expert"] = sum(i["expert"] for i in info)
    ident["n_sp"] = sum(i["sp"] for i in info)
    ident["n_tp_replicated"] = sum((not i["expert"]) and (not i["tp_part"]) for i in info)

    def proj(i, flat, lo=0, hi=None):
        """<flat[lo:hi], r_k[lo:hi]> for k<K (fp64 accumulate), r regenerated from the per-param seed."""
        hi = flat.numel() if hi is None else hi
        out = torch.zeros(K, dtype=torch.float64, device="cuda")
        if hi <= lo:
            return out
        g = torch.Generator(device="cuda")
        seg = flat[lo:hi].float()
        for k in range(K):
            g.manual_seed((i["seed"] + 7919 * k) & ((1 << 62) - 1))
            r = torch.randn(flat.numel(), generator=g, device="cuda", dtype=torch.float32)[lo:hi]
            out[k] = torch.dot(seg, r).double()
        return out

    calls = {"n": 0}
    pre = {}

    def wrap(orig):
        def probe_finalize(*a, **kw):
            calls["n"] += 1
            torch.cuda.synchronize()
            for idx, i in enumerate(info):
                mg = getattr(i["p"], "main_grad", None)
                pre[idx] = (proj(i, mg.view(-1)).cpu().tolist() if mg is not None else None)
            return orig(*a, **kw)
        return probe_finalize

    if args.mode == "normal":
        assert sched_cfg.finalize_model_grads_func is not None, "finalize_model_grads_func is None on the schedule config"
        sched_cfg.finalize_model_grads_func = wrap(sched_cfg.finalize_model_grads_func)
    else:
        ident["nofinalize_removed"] = repr(sched_cfg.finalize_model_grads_func)[:120]
        sched_cfg.finalize_model_grads_func = None

    # ---- POST capture at optimizer.step entry (exactly what the optimizer consumes)
    opt = worker.optimizer
    subopts = list(getattr(opt, "chained_optimizers", [opt]))
    owner = {}
    for so in subopts:
        if hasattr(so, "model_param_gbuf_map"):
            for p in so.model_param_gbuf_map:
                owner[p] = so
    post = {}

    orig_step = opt.step

    def probe_step(*a, **kw):
        torch.cuda.synchronize()
        for idx, i in enumerate(info):
            so = owner.get(i["p"])
            mg = getattr(i["p"], "main_grad", None)
            if so is None or mg is None:
                post[idx] = None
                continue
            rng = so._get_model_param_range_map(i["p"])["param"]
            flat = mg.view(-1)
            s_ = proj(i, flat, rng.start, rng.end)
            nsq = flat[rng.start:rng.end].float().pow(2).sum().double().reshape(1)
            post[idx] = (subopts.index(so), torch.cat([s_, nsq]).cpu().tolist(), int(rng.end - rng.start))
        torch.cuda.synchronize()
        res = orig_step(*a, **kw)
        post["_step_return"] = [float(x) if x is not None and not isinstance(x, bool) else x for x in res]
        return res

    opt.step = probe_step

    # ---- fixed synthetic batch, identical global batch in every layout; DP rank d takes a contiguous slice
    gen = torch.Generator().manual_seed(args.data_seed)
    N = args.num_seqs
    assert N % dp_size == 0
    lens = torch.randint(args.min_len, args.max_len + 1, (N,), generator=gen)
    S = int(lens.max())
    vocab = len(tokenizer)
    ids = torch.randint(1000, min(vocab, 130000) - 1000, (N, S), generator=gen)
    adv_seq = torch.where(torch.rand(N, generator=gen) < 0.5, -1.0, 1.0)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    token_mask = torch.zeros(N, S)
    for b in range(N):
        L = int(lens[b]); ids[b, L:] = pad
        token_mask[b, int(L * 0.6):L] = 1.0
    full = dict(input_ids=ids, input_lengths=lens, token_mask=token_mask, sample_mask=torch.ones(N),
                advantages=adv_seq[:, None] * token_mask, generation_logprobs=torch.zeros(N, S),
                prev_logprobs=torch.zeros(N, S), reference_policy_logprobs=torch.zeros(N, S),
                rewards=(adv_seq > 0).float())
    per = N // dp_size
    # Same sharding + sequence-packing assignment as lm_policy.Policy._shard_for_train
    sp_cfg = policy_cfg["sequence_packing"]
    assert sp_cfg["enabled"] and not policy_cfg["dynamic_batching"]["enabled"]
    packing_args = {"algorithm": sp_cfg["algorithm"], "input_key": "input_ids", "input_lengths_key": "input_lengths",
                    "sequence_length_pad_multiple": policy_cfg["make_sequence_length_divisible_by"],
                    "max_tokens_per_microbatch": sp_cfg["train_mb_tokens"]}
    if sp_cfg.get("microbatch_order") is not None:
        packing_args["microbatch_order"] = sp_cfg["microbatch_order"]
    shards, _ = BatchedDataDict(full).shard_by_batch_size(dp_size, batch_size=N, sequence_packing_args=packing_args)
    data = shards[dp_rank]
    ident["microbatches_this_dp_rank"] = len(data.micro_batch_indices[0]) if data.micro_batch_indices else None
    ident["batch"] = {"N": N, "per_dp": per, "max_len": S, "total_tokens": int(lens.sum()),
                      "response_tokens": int(token_mask.sum())}

    loss_fn = ClippedPGLossFn(cfg.loss_fn)
    torch.distributed.barrier()
    t1 = time.time()
    results = worker.train(data, loss_fn, gbs=N, mbs=1)
    ident["train_seconds"] = round(time.time() - t1, 1)
    ident["finalize_calls"] = calls["n"]
    ident["step_return"] = post.pop("_step_return", None)
    try:
        ident["train_loss"] = [float(x) for x in torch.as_tensor(results["global_loss"]).float().view(-1)]
        ident["grad_norm"] = [float(x) for x in torch.as_tensor(results["grad_norm"]).float().view(-1)]
    except Exception as e:  # metrics are informational only
        ident["metrics_error"] = repr(e)[:200]

    # ---- aggregation: one all_gather_object per group, keyed by (param name, coord) -> order/asymmetry safe
    def gather_sum(local, group):
        objs = [None] * torch.distributed.get_world_size(group)
        torch.distributed.all_gather_object(objs, local, group=group)
        tot, cnt = {}, {}
        for o in objs:
            for k, v in o.items():
                if v is None:
                    continue
                if k in tot:
                    tot[k] = [a + b for a, b in zip(tot[k], v)]
                else:
                    tot[k] = list(v)
                cnt[k] = cnt.get(k, 0) + 1
        return tot, cnt, len(objs)

    key = lambda i: i["name"] + "@" + i["coord"]
    # POST: per sub-optimizer, over its data_parallel_group (what the optimizer shards over)
    post_tot, post_cnt, post_gsize, owned_elems, post_opt = {}, {}, {}, {}, {}
    for oi, so in enumerate(subopts):
        local = {key(i): post[idx][1] for idx, i in enumerate(info)
                 if post.get(idx) is not None and post[idx][0] == oi}
        loc_el = {key(i): post[idx][2] for idx, i in enumerate(info)
                  if post.get(idx) is not None and post[idx][0] == oi}
        tot, cnt, gsz = gather_sum(local, so.data_parallel_group)
        el, _, _ = gather_sum({k: [v] for k, v in loc_el.items()}, so.data_parallel_group)
        for k in tot:
            post_tot[k], post_cnt[k], post_gsize[k], owned_elems[k] = tot[k], cnt[k], gsz, int(el[k][0])
            post_opt[k] = oi
    ident["post_subopt_param_counts"] = [sum(1 for idx in range(len(info)) if post.get(idx) is not None and post[idx][0] == oi)
                                         for oi in range(len(subopts))]
    ident["post_subopt_group_sizes"] = [torch.distributed.get_world_size(so.data_parallel_group) for so in subopts]
    # PRE expected: sum over the group finalize should reduce over
    pre_tot, pre_cnt = {}, {}
    if args.mode == "normal":
        for gname, grp in (("dpcp", dpcp_group), ("tpdpcp", tpdpcp_group), ("edp", edp_group)):
            local = {key(i): pre.get(idx) for idx, i in enumerate(info) if i["pre_group"] is grp}
            tot, cnt, _ = gather_sum(local, grp)
            pre_tot.update(tot); pre_cnt.update(cnt)
    # TP consistency for TP-replicated dense params
    rep_local = {key(i): post_tot.get(key(i)) for i in info if not i["expert"] and not i["tp_part"]}
    objs = [None] * torch.distributed.get_world_size(tp_group)
    torch.distributed.all_gather_object(objs, rep_local, group=tp_group)
    tp_dev = {}
    for k, v0 in rep_local.items():
        if v0 is None:
            continue
        vs = [o.get(k) for o in objs]
        if any(v is None for v in vs):
            tp_dev[k] = float("nan"); continue
        ref = max(abs(x) for x in v0[:K]) or 1e-30
        tp_dev[k] = max(abs(a - b) for v in vs for a, b in zip(v[:K], v0[:K])) / ref

    # write: one writer per reduction group (rank 0 of the owning optimizer's group); 'rep' only on TP rank 0
    rows = []
    for idx, i in enumerate(info):
        k = key(i)
        if k not in post_tot:  # reduced key is known on every group member, owned shard or not
            continue
        so = subopts[post_opt[k]]
        if torch.distributed.get_rank(group=so.data_parallel_group) != 0:
            continue
        if i["coord"] == "rep" and tp_rank != 0:
            continue
        v = post_tot[k]
        rows.append({"name": i["name"], "coord": i["coord"], "expert": i["expert"], "sp": i["sp"],
                     "tp_part": i["tp_part"], "numel": i["numel"], "owned_elems_total": owned_elems.get(k),
                     "post_members": post_cnt.get(k), "post_group_size": post_gsize.get(k),
                     "post_proj": v[:K], "post_sqnorm": v[K],
                     "pre_expected_proj": pre_tot.get(k), "pre_members": pre_cnt.get(k),
                     "tp_rel_dev": tp_dev.get(k)})
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"stats-rank{rank:03d}.json"), "w") as f:
        json.dump({"ident": ident, "rows": rows}, f)
    torch.distributed.barrier()
    if rank == 0:
        with open(os.path.join(args.out, "DONE.json"), "w") as f:
            json.dump({"ident_rank0": ident, "time": time.time()}, f, indent=1)
    log(f"done: finalize_calls={calls['n']} rows_written={len(rows)} train_s={ident['train_seconds']}")


if __name__ == "__main__":
    main()
