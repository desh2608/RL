#!/usr/bin/env python3
"""Compare gradsync_probe.py outputs: reference (DP1) vs production layout (DP4) vs no-finalize control.

Random-projection estimator: for r ~ N(0, I), E[<d, r>^2] = ||d||^2, so with K projections
  rel_err(prod, ref) ~= sqrt(mean_k (s_prod_k - s_ref_k)^2) / ||g_ref||   (relative L2 error of the full local grad)
Usage: compare_gradsync.py REF_DIR PROD_DIR [NOFIN_DIR] [--json OUT]
"""
import glob, json, math, os, statistics, sys


def load(d):
    rows, idents = {}, []
    for f in sorted(glob.glob(os.path.join(d, "stats-rank*.json"))):
        j = json.load(open(f))
        idents.append(j["ident"])
        for r in j["rows"]:
            rows[(r["name"], r["coord"])] = r
    return rows, idents


def cls(r):
    if r["expert"]:
        return "expert"
    if r["coord"] == "rep":
        return "tp-replicated SP" if r["sp"] else "tp-replicated non-SP"
    return "tp-sharded dense"


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(p * (len(v) - 1))))] if v else float("nan")


def cross(a, b):
    """per-key relative L2 error of a vs b (b = reference)."""
    out = {}
    for k, rb in b.items():
        ra = a.get(k)
        if ra is None:
            continue
        K = len(rb["post_proj"])
        d2 = sum((x - y) ** 2 for x, y in zip(ra["post_proj"], rb["post_proj"])) / K
        nb = rb["post_sqnorm"]
        out[k] = dict(cls=cls(rb), rel=math.sqrt(d2) / math.sqrt(nb) if nb > 0 else float("nan"),
                      norm_ratio=math.sqrt(ra["post_sqnorm"] / nb) if nb > 0 else float("nan"),
                      d2=d2, nb=nb, numel=rb["numel"], name=k[0], coord=k[1])
    return out


def summarize(c):
    res = {}
    by = {}
    for v in c.values():
        by.setdefault(v["cls"], []).append(v)
    for name, vs in sorted(by.items()):
        rel = [v["rel"] for v in vs if not math.isnan(v["rel"])]
        nr = [v["norm_ratio"] for v in vs if not math.isnan(v["norm_ratio"])]
        res[name] = dict(n=len(vs), rel_median=q(rel, .5), rel_p90=q(rel, .9), rel_max=max(rel) if rel else None,
                         norm_ratio_median=q(nr, .5), norm_ratio_min=min(nr) if nr else None, norm_ratio_max=max(nr) if nr else None,
                         weighted_rel=math.sqrt(sum(v["d2"] for v in vs) / sum(v["nb"] for v in vs)))
    allv = list(c.values())
    res["ALL"] = dict(n=len(allv), weighted_rel=math.sqrt(sum(v["d2"] for v in allv) / sum(v["nb"] for v in allv)))
    worst = sorted(allv, key=lambda v: -v["rel"])[:8]
    res["worst"] = [dict(name=v["name"], coord=v["coord"], cls=v["cls"], rel=round(v["rel"], 4),
                         norm_ratio=round(v["norm_ratio"], 4), numel=v["numel"]) for v in worst]
    return res


def internal(rows, idents):
    pre = [r for r in rows.values() if r.get("pre_expected_proj") is not None]
    rel = []
    for r in pre:
        e, p = r["pre_expected_proj"], r["post_proj"]
        den = math.sqrt(sum(x * x for x in e) / len(e)) or 1e-30
        rel.append(math.sqrt(sum((x - y) ** 2 for x, y in zip(p, e)) / len(e)) / den)
    tpd = [r["tp_rel_dev"] for r in rows.values() if r.get("tp_rel_dev") is not None]
    keys = ["sched_cfg_is_provider", "sched_cfg_is_float16_cfg", "sched_cfg_is_language_model_cfg",
            "sched_finalize_set", "provider_finalize_set", "finalize_calls", "overlap_grad_reduce",
            "use_distributed_optimizer", "calculate_per_token_loss", "sequence_parallel", "dp_size", "dp_cp_size",
            "expt_dp_size", "world", "model_type", "inner_type", "n_trainable", "n_expert", "n_sp", "n_tp_replicated"]
    agree = {k: sorted({json.dumps(i.get(k)) for i in idents}) for k in keys}
    gn = sorted({json.dumps(i.get("grad_norm")) for i in idents})
    return dict(ranks=len(idents), ident=agree, grad_norm_reported=gn,
                train_loss=sorted({json.dumps(i.get("train_loss")) for i in idents})[:2],
                train_seconds=max((i.get("train_seconds") or 0) for i in idents),
                batch=idents[0].get("batch") if idents else None,
                pre_post_rel_median=q(rel, .5) if rel else None, pre_post_rel_max=max(rel) if rel else None,
                pre_post_n=len(rel), tp_replicated_rel_dev_max=max(tpd) if tpd else None, tp_replicated_n=len(tpd),
                harness_global_grad_norm=math.sqrt(sum(r["post_sqnorm"] for r in rows.values())))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out_json = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    if out_json in args:
        args.remove(out_json)
    ref_rows, ref_id = load(args[0])
    prod_rows, prod_id = load(args[1])
    report = {"reference": internal(ref_rows, ref_id), "production": internal(prod_rows, prod_id)}
    c = cross(prod_rows, ref_rows)
    report["prod_vs_ref"] = summarize(c)
    report["key_coverage"] = dict(ref=len(ref_rows), prod=len(prod_rows), common=len(c))
    if len(args) > 2:
        nf_rows, nf_id = load(args[2])
        report["nofinalize"] = internal(nf_rows, nf_id)
        report["nofinalize_vs_ref"] = summarize(cross(nf_rows, ref_rows))
    print(json.dumps(report, indent=1))
    if out_json:
        json.dump(report, open(out_json, "w"), indent=1)


if __name__ == "__main__":
    main()
