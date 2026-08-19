"""Analyze trajectories: convergence metrics, semantic drift, LLM-judge quality,
statistics, and figures. Reads results/trajectories/*.json, writes
results/*.json/.csv and figures/*.png."""
import os
import json
import glob
import csv
import hashlib

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import llm
import prompts
import metrics as M

ROOT = os.path.join(os.path.dirname(__file__), "..")
TRAJ = os.path.join(ROOT, "results", "trajectories")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

JUDGE_MODEL = "gpt-4o"   # different generation from reviser gpt-4.1 (self-bias guard)


CACHE = os.path.join(RES, "cache")


def _cache_key(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def version_truncation_flags(tr):
    """Reconstruct, for each *version* V_1..V_T, whether the revise call that
    produced it was cut off at the token limit (finish_reason == 'length').
    V_0 is the seed and is never truncated. Uses the on-disk cache (free)."""
    cfg = tr["config"]
    vs = tr["versions"]
    rvs = tr["reviews"]
    tmpl = (prompts.REVISE_USER if cfg.get("preserve_length", True)
            else prompts.REVISE_USER_UNCONSTRAINED)
    flags = [False]  # V_0
    for t in range(len(vs) - 1):
        if cfg["mode"] == "two_stage":
            user = tmpl.format(version=vs[t], review=rvs[t])
            systxt = prompts.REVISE_SYSTEM
        else:
            user = prompts.SELFREFINE_USER.format(version=vs[t])
            systxt = prompts.SELFREFINE_SYSTEM
        payload = {"model": cfg["reviser_model"], "system": systxt, "user": user,
                   "temperature": cfg["temperature"], "max_tokens": 2048,
                   "seed": cfg["seed"]}
        path = os.path.join(CACHE, "chat_" + _cache_key(payload) + ".json")
        fr = None
        if os.path.exists(path):
            fr = json.load(open(path)).get("finish_reason")
        flags.append(fr == "length")
    return flags


def parse_judge(txt):
    try:
        s = txt[txt.index("{"): txt.rindex("}") + 1]
        return json.loads(s)
    except Exception:  # noqa: BLE001
        return None


def judge_version(v):
    txt = llm.chat(prompts.JUDGE_SYSTEM, prompts.JUDGE_USER.format(version=v),
                   model=JUDGE_MODEL, temperature=0.0, max_tokens=100)
    return parse_judge(txt) or {}


def judge_meaning(orig, rev):
    txt = llm.chat(prompts.MEANING_SYSTEM,
                   prompts.MEANING_USER.format(original=orig, revised=rev),
                   model=JUDGE_MODEL, temperature=0.0, max_tokens=100)
    return parse_judge(txt) or {}


def analyze_trajectory(tr, do_judge=True, do_embed=True):
    vs = tr["versions"]
    deltas = M.step_deltas(vs)
    fit = M.fit_exponential(deltas)
    # truncation-aware: a delta d(V_t, V_{t+1}) is "clean" iff neither endpoint
    # version was cut off at the token limit.
    tflags = version_truncation_flags(tr)
    clean_mask = [(not tflags[t]) and (not tflags[t + 1])
                  for t in range(len(vs) - 1)]
    clean_deltas = [deltas[t] for t in range(len(deltas)) if clean_mask[t]]
    # contiguous truncation-free prefix (steps before ANY version is truncated)
    first_trunc_ver = next((i for i, f in enumerate(tflags) if f), None)
    prefix_deltas = (deltas if first_trunc_ver is None
                     else deltas[:max(first_trunc_ver - 1, 0)])
    rec = {
        "condition": tr.get("condition"),
        "doc_id": tr["doc_id"],
        "n_versions": len(vs),
        "char_lengths": [len(v) for v in vs],
        "word_lengths": [len(v.split()) for v in vs],
        "deltas": deltas,
        "final_delta": deltas[-1] if deltas else None,
        "min_delta": min(deltas) if deltas else None,
        "exact_fp": M.first_exact_fp(vs),
        "approx_fp": M.first_approx_fp(vs),
        "fit_k": fit["k"], "fit_c": fit["c"], "fit_r2": fit["r2"],
        "n_trunc_versions": int(sum(tflags)),
        "first_trunc_ver": first_trunc_ver,
        "clean_deltas": clean_deltas,
        "clean_final_delta": clean_deltas[-1] if clean_deltas else None,
        "clean_min_delta": min(clean_deltas) if clean_deltas else None,
        "prefix_last_delta": prefix_deltas[-1] if prefix_deltas else None,
        "n_clean_steps": len(clean_deltas),
    }
    # semantic drift: cosine(V0, Vt)
    if do_embed:
        embs = [llm.embed(v) for v in vs]
        rec["drift_v0"] = [M.cosine(embs[0], e) for e in embs]
        rec["consec_cos"] = [M.cosine(embs[t], embs[t + 1])
                             for t in range(len(embs) - 1)]
    # quality trajectory (V0, mid, final) + meaning preservation of final
    if do_judge:
        idxs = sorted(set([0, len(vs) // 2, len(vs) - 1]))
        rec["judge"] = {i: judge_version(vs[i]) for i in idxs}
        rec["meaning_final"] = judge_meaning(vs[0], vs[-1])
    return rec


def aggregate(records):
    """Group per condition and summarize."""
    conds = {}
    for r in records:
        conds.setdefault(r["condition"], []).append(r)
    summary = {}
    for c, rs in conds.items():
        fds = [r["final_delta"] for r in rs if r["final_delta"] is not None]
        mds = [r["min_delta"] for r in rs if r["min_delta"] is not None]
        approx = [r["approx_fp"] for r in rs]
        exact = [r["exact_fp"] for r in rs]
        cfloor = [r["fit_c"] for r in rs if r["fit_c"] is not None]
        r2s = [r["fit_r2"] for r in rs if r["fit_r2"] is not None]
        # mean delta curve (align by step; ragged only if n_iters differ)
        maxlen = max(len(r["deltas"]) for r in rs)
        curve = []
        for t in range(maxlen):
            vals = [r["deltas"][t] for r in rs if t < len(r["deltas"])]
            curve.append(float(np.mean(vals)))
        len_growth = [float(np.mean([r["char_lengths"][t] for r in rs
                                     if t < len(r["char_lengths"])]))
                      for t in range(max(len(r["char_lengths"]) for r in rs))]
        cfd = [r["clean_final_delta"] for r in rs if r["clean_final_delta"] is not None]
        cmd = [r["clean_min_delta"] for r in rs if r["clean_min_delta"] is not None]
        ntr = [r["n_trunc_versions"] for r in rs]
        summary[c] = {
            "n_docs": len(rs),
            "mean_final_delta": float(np.mean(fds)) if fds else None,
            "sd_final_delta": float(np.std(fds)) if fds else None,
            "mean_min_delta": float(np.mean(mds)) if mds else None,
            "mean_clean_final_delta": float(np.mean(cfd)) if cfd else None,
            "mean_clean_min_delta": float(np.mean(cmd)) if cmd else None,
            "mean_trunc_versions": float(np.mean(ntr)),
            "pct_exact_fp": float(np.mean([e is not None for e in exact])),
            "pct_approx_fp": float(np.mean([a is not None for a in approx])),
            "mean_approx_fp_time": (float(np.mean([a for a in approx if a is not None]))
                                    if any(a is not None for a in approx) else None),
            "mean_fit_c": float(np.mean(cfloor)) if cfloor else None,
            "mean_fit_r2": float(np.mean(r2s)) if r2s else None,
            "mean_delta_curve": curve,
            "mean_len_curve": len_growth,
        }
    return summary


def main():
    files = sorted(glob.glob(os.path.join(TRAJ, "*.json")))
    all_recs = []
    for fp in files:
        with open(fp) as f:
            trs = json.load(f)
        for tr in trs:
            tr.setdefault("condition", os.path.basename(fp)[:-5])
            all_recs.append(analyze_trajectory(tr))
        print("analyzed", os.path.basename(fp), len(trs))

    with open(os.path.join(RES, "per_trajectory.json"), "w") as f:
        json.dump(all_recs, f, indent=2)

    summary = aggregate(all_recs)
    with open(os.path.join(RES, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---- CSV of per-trajectory key metrics ----
    with open(os.path.join(RES, "per_trajectory.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "doc_id", "final_delta", "min_delta",
                    "exact_fp", "approx_fp", "fit_k", "fit_c", "fit_r2",
                    "len0", "len_final", "drift_final"])
        for r in all_recs:
            w.writerow([r["condition"], r["doc_id"], r["final_delta"],
                        r["min_delta"], r["exact_fp"], r["approx_fp"],
                        r["fit_k"], r["fit_c"], r["fit_r2"],
                        r["char_lengths"][0], r["char_lengths"][-1],
                        r.get("drift_v0", [None])[-1]])

    # ---- statistics ----
    stats_out = run_stats(all_recs)
    with open(os.path.join(RES, "stats.json"), "w") as f:
        json.dump(stats_out, f, indent=2)

    make_figures(all_recs, summary)
    print("COST:", json.dumps(llm.cost_report()))
    print("Wrote summary.json, per_trajectory.{json,csv}, stats.json, figures/")


def _by_cond(recs):
    d = {}
    for r in recs:
        d.setdefault(r["condition"], []).append(r)
    return d


def run_stats(recs):
    d = _by_cond(recs)
    out = {}

    def paired(ca, cb, key="final_delta"):
        if ca not in d or cb not in d:
            return None
        ma = {r["doc_id"]: r[key] for r in d[ca]}
        mb = {r["doc_id"]: r[key] for r in d[cb]}
        ids = [i for i in ma if i in mb and ma[i] is not None and mb[i] is not None]
        if len(ids) < 3:
            return None
        a = [ma[i] for i in ids]
        b = [mb[i] for i in ids]
        try:
            w = stats.wilcoxon(a, b)
            wp = float(w.pvalue)
        except Exception:  # noqa: BLE001
            wp = None
        diff = np.array(a) - np.array(b)
        d_eff = float(np.mean(diff) / (np.std(diff) + 1e-9))
        return {"n": len(ids), "mean_a": float(np.mean(a)),
                "mean_b": float(np.mean(b)), "wilcoxon_p": wp,
                "cohen_dz": d_eff}

    # H0: two-stage review->revise does not reach a fixed point.
    # test final_delta vs 0 (one-sample) for D1_main
    if "D1_main" in d:
        fds = [r["final_delta"] for r in d["D1_main"] if r["final_delta"] is not None]
        t = stats.ttest_1samp(fds, 0.0)
        out["D1_final_delta_vs_zero"] = {
            "mean": float(np.mean(fds)), "sd": float(np.std(fds)),
            "t": float(t.statistic), "p": float(t.pvalue), "n": len(fds)}

    # robustness: same conclusion on the truncation-free "clean" min delta
    if "D1_main" in d:
        cmd = [r["clean_min_delta"] for r in d["D1_main"]
               if r["clean_min_delta"] is not None]
        if cmd:
            t = stats.ttest_1samp(cmd, M.EPS_EDIT)
            out["D1_clean_min_delta_vs_approxFP_thr"] = {
                "mean": float(np.mean(cmd)), "sd": float(np.std(cmd)),
                "thr": M.EPS_EDIT, "t": float(t.statistic),
                "p": float(t.pvalue), "n": len(cmd)}
    out["single_vs_two_clean_min_delta"] = paired(
        "D2_single_stage", "D1_main", key="clean_min_delta")
    out["single_vs_two_stage_final_delta"] = paired("D2_single_stage", "D1_main")
    out["temp_default_vs_temp0_final_delta"] = paired("D2_temp_default", "D1_main")
    out["cross_vs_self_reviewer_final_delta"] = paired("D2_cross_model", "D1_main")
    out["harsh_vs_structured_final_delta"] = paired("D2_framing_harsh", "D1_main")
    out["lenient_vs_structured_final_delta"] = paired("D2_framing_lenient", "D1_main")
    out["unconstrained_vs_constrained_final_delta"] = paired("D2_unconstrained", "D1_main")
    return out


def make_figures(recs, summary):
    d = _by_cond(recs)
    order = ["D1_main", "D2_single_stage", "D2_temp_default", "D2_cross_model",
             "D2_framing_harsh", "D2_framing_lenient", "D2_genre_prose",
             "D2_unconstrained"]
    order = [c for c in order if c in summary]

    # Fig 1: mean delta curves
    plt.figure(figsize=(8, 5))
    for c in order:
        curve = summary[c]["mean_delta_curve"]
        plt.plot(range(1, len(curve) + 1), curve, marker="o", label=c)
    plt.axhline(M.EPS_EDIT, ls="--", color="gray", label=f"approx-FP thr={M.EPS_EDIT}")
    plt.xlabel("iteration t (step V_{t-1}->V_t)")
    plt.ylabel("mean normalized edit distance  Δt")
    plt.title("Convergence dynamics: per-step change by condition")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "delta_curves.png"), dpi=130)
    plt.close()

    # Fig 2: length growth
    plt.figure(figsize=(8, 5))
    for c in order:
        curve = summary[c]["mean_len_curve"]
        plt.plot(range(len(curve)), curve, marker="s", label=c)
    plt.xlabel("iteration t (version index)")
    plt.ylabel("mean document length (chars)")
    plt.title("Document length trajectory by condition")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "length_growth.png"), dpi=130)
    plt.close()

    # Fig 3: final delta boxplot
    plt.figure(figsize=(9, 5))
    data = [[r["final_delta"] for r in d[c] if r["final_delta"] is not None]
            for c in order]
    plt.boxplot(data, labels=[c.replace("D2_", "").replace("D1_", "")
                              for c in order])
    plt.axhline(M.EPS_EDIT, ls="--", color="red", label=f"approx-FP thr")
    plt.ylabel("final-step Δt (V_{T-1}->V_T)")
    plt.title("Final-step change: is the loop at a fixed point?")
    plt.xticks(rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "final_delta_box.png"), dpi=130)
    plt.close()

    # Fig 4: semantic drift (cosine V0 vs Vt) for main + prose + single
    plt.figure(figsize=(8, 5))
    for c in ["D1_main", "D2_single_stage", "D2_genre_prose", "D2_unconstrained"]:
        if c not in d:
            continue
        curves = [r["drift_v0"] for r in d[c] if "drift_v0" in r]
        if not curves:
            continue
        maxlen = max(len(x) for x in curves)
        mean = [float(np.mean([x[t] for x in curves if t < len(x)]))
                for t in range(maxlen)]
        plt.plot(range(maxlen), mean, marker="o", label=c)
    plt.xlabel("iteration t (version index)")
    plt.ylabel("cosine(V0, Vt)  — meaning preservation")
    plt.title("Semantic drift from the original")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "semantic_drift.png"), dpi=130)
    plt.close()

    # Fig 5: quality trajectory (judge) for a few conditions
    plt.figure(figsize=(9, 5))
    crit = ["clarity", "conciseness", "technical_precision"]
    conds_q = ["D1_main", "D2_single_stage", "D2_unconstrained"]
    width = 0.12
    xbase = np.arange(3)  # V0, Vmid, Vfinal
    ci = 0
    for c in conds_q:
        if c not in d:
            continue
        rs = [r for r in d[c] if "judge" in r]
        if not rs:
            continue
        # average across docs, average across criteria -> overall score per stage
        stage_keys = sorted({int(k) for r in rs for k in r["judge"]})
        stages = stage_keys[:3]
        vals = []
        for sidx in stages:
            sc = []
            for r in rs:
                jr = r["judge"].get(str(sidx)) or r["judge"].get(sidx)
                if jr:
                    sc.append(np.mean([jr.get(k, np.nan) for k in crit]))
            vals.append(float(np.nanmean(sc)))
        plt.bar(xbase + ci * width, vals, width, label=c)
        ci += 1
    plt.xticks(xbase + width, ["V0 (orig)", "V_mid", "V_final"])
    plt.ylabel("mean judge score (1-10, avg of 3 criteria)")
    plt.title("Quality trajectory (LLM-judge = gpt-4o)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "quality_trajectory.png"), dpi=130)
    plt.close()

    # Fig 6: truncation-robustness -- clean (untruncated) minimum Δt per condition
    plt.figure(figsize=(9, 5))
    data = [[r["clean_min_delta"] for r in d[c]
             if r.get("clean_min_delta") is not None] for c in order]
    data = [x if x else [np.nan] for x in data]
    plt.boxplot(data, labels=[c.replace("D2_", "").replace("D1_", "")
                              for c in order])
    plt.axhline(M.EPS_EDIT, ls="--", color="red",
                label=f"approx-FP thr={M.EPS_EDIT}")
    plt.ylabel("min Δt over truncation-free steps")
    plt.title("Robustness: convergence is not reached even on untruncated steps")
    plt.xticks(rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "clean_min_delta_box.png"), dpi=130)
    plt.close()


if __name__ == "__main__":
    main()
