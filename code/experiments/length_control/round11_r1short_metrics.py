"""Round 11 P1B: compute MI/AUROC/TPR/source-paired ratio for R1-short
versus R0 (canonical pool) versus R1-original (canonical pool).

Canonical setting: pi=0.5, B=500 source-paired bootstrap, 20 equal-width
bins, clipped Miller-Madow, seed 20260513, peak strength delta=5.

Sources:
- R1-short z-scores: compute via fast_z on data/candidates_r1short.jsonl
- R0 / R1-original z-scores: read from data/zscore_index_round6.jsonl
  (variant=A, peak strength, syn corpus)
"""
import json, sys, math
from pathlib import Path
import numpy as np

try: sys.stdout.reconfigure(encoding="utf-8")
except: pass

sys.path.insert(0, "./data/scripts")
from fast_z import fast_detect_z

VOCAB = 152064; GAMMA = 0.25; KEY = 20260513
PEAK = 5.0
SEED = 20260513
N_BOOT = 500
N_BINS = 20


def mm_h(c):
    n = c.sum()
    if n == 0: return 0.0
    p = c / n; p = p[p > 0]
    H = -np.sum(p * np.log2(p))
    K = (p > 0).sum()
    return H + (K - 1) / (2 * n * math.log(2))


def mi(z_wm, z_no, n_bins=N_BINS):
    if len(z_wm) < 5 or len(z_no) < 5: return 0.0
    all_z = np.concatenate([z_wm, z_no])
    lo, hi = float(all_z.min()), float(all_z.max())
    if hi - lo < 1e-9: return 0.0
    edges = np.linspace(lo, hi + 1e-9, n_bins + 1)
    c_a = np.histogram(all_z, bins=edges)[0]
    c_w = np.histogram(z_wm, bins=edges)[0]
    c_n = np.histogram(z_no, bins=edges)[0]
    nt, nw, nn = len(all_z), len(z_wm), len(z_no)
    pw = np.array([nn, nw]) / nt
    raw = mm_h(c_a) - (pw[1] * mm_h(c_w) + pw[0] * mm_h(c_n))
    return float(np.clip(raw, 0, 1))


def auroc(pos, neg):
    pos = np.asarray(pos); neg = np.asarray(neg)
    n_p, n_n = len(pos), len(neg)
    if n_p == 0 or n_n == 0: return float("nan")
    all_s = np.concatenate([pos, neg])
    ranks = np.argsort(np.argsort(all_s)) + 1
    R_p = float(ranks[:n_p].sum())
    return (R_p - n_p * (n_p + 1) / 2.0) / (n_p * n_n)


def tpr_at(pos, neg, fpr):
    pos = np.asarray(pos); neg = np.asarray(neg)
    if len(neg) == 0: return float("nan")
    sorted_neg = np.sort(neg)[::-1]
    k = max(int(round(fpr * len(neg))) - 1, 0)
    thr = float(sorted_neg[k])
    return float(np.mean(pos >= thr))


def boot(z_wm, z_no, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n_w, n_n = len(z_wm), len(z_no)
    if n_w < 5 or n_n < 5:
        return {"mi": float("nan"), "mi_ci": [float("nan")]*2,
                "auroc": float("nan"), "tpr1": float("nan"), "tpr5": float("nan")}
    mi_p = mi(z_wm, z_no); au_p = auroc(z_wm, z_no)
    t1 = tpr_at(z_wm, z_no, 0.01); t5 = tpr_at(z_wm, z_no, 0.05)
    mi_b = np.empty(n_boot)
    for b in range(n_boot):
        iw = rng.integers(0, n_w, size=n_w); iN = rng.integers(0, n_n, size=n_n)
        mi_b[b] = mi(z_wm[iw], z_no[iN])
    return {"mi": mi_p, "mi_ci": [float(np.percentile(mi_b, 2.5)),
                                   float(np.percentile(mi_b, 97.5))],
            "auroc": au_p, "tpr1": t1, "tpr5": t5}


def boot_ratio(wm0_by_id, no0, wm1_by_id, no1, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    common = sorted(set(wm0_by_id) & set(wm1_by_id))
    if len(common) < 5:
        return float("nan"), float("nan"), float("nan"), len(common)
    z_w0 = np.array([wm0_by_id[i] for i in common])
    z_w1 = np.array([wm1_by_id[i] for i in common])
    z_n0 = np.asarray(no0); z_n1 = np.asarray(no1)
    p0 = mi(z_w0, z_n0); p1 = mi(z_w1, z_n1)
    point = p1 / max(p0, 1e-6)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        ip = rng.integers(0, len(common), size=len(common))
        i0 = rng.integers(0, len(z_n0), size=len(z_n0))
        i1 = rng.integers(0, len(z_n1), size=len(z_n1))
        m0 = mi(z_w0[ip], z_n0[i0]); m1 = mi(z_w1[ip], z_n1[i1])
        boots[b] = m1 / max(m0, 1e-6)
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), len(common)


def main():
    print("loading R1-short candidates and computing z-scores...")
    r1s = [json.loads(l) for l in open(
        "./data/data/candidates_r1short.jsonl",
        encoding="utf-8")]
    print(f"  {len(r1s)} R1-short candidates")
    for r in r1s:
        toks = r.get("token_ids") or []
        if r["family"] == "no_wm":
            r["z_kgw"] = fast_detect_z("kgw", toks, VOCAB, GAMMA, KEY)
            r["z_uw"]  = fast_detect_z("uw",  toks, VOCAB, GAMMA, KEY)
        else:
            r["z_self"] = fast_detect_z(r["family"], toks, VOCAB, GAMMA, KEY)

    print("loading canonical R0/R1 z-score index...")
    zidx = [json.loads(l) for l in open(
        "./data/data/zscore_index_round6.jsonl",
        encoding="utf-8")]
    def corp(r): return "nat" if "natural" in r.get("src_file","") else "syn"

    def collect_canonical(regime, fam):
        wm = [r for r in zidx if corp(r)=="syn" and r["regime"]==regime
              and r["family"]==fam and r["strength"]==PEAK
              and r.get("variant","A")=="A"]
        no = [r for r in zidx if corp(r)=="syn" and r["regime"]==regime
              and r["family"]=="no_wm" and r.get("variant","A")=="A"]
        # downsample no to wm count (Round-7 canonical convention)
        rng = np.random.default_rng(SEED)
        target = min(len(wm), len(no))
        if len(wm) > target:
            idx = rng.choice(len(wm), size=target, replace=False)
            wm = [wm[i] for i in sorted(idx)]
        if len(no) > target:
            idx = rng.choice(len(no), size=target, replace=False)
            no = [no[i] for i in sorted(idx)]
        return wm, no

    out = {"cells": [], "ratios_source_paired": [], "n_boot": N_BOOT,
           "seed": SEED, "n_bins": N_BINS, "peak": PEAK,
           "sample_description": ("R0/R1-original from canonical Round-7 "
            "balanced sample (variant=A, peak=5.0, n=200/200). R1-short "
            "from new generation runs/round11; balanced via downsample to "
            "min(wm,no_wm) per family.")}

    print("\n=== R1-short cell-level metrics (balanced) ===")
    for fam in ("kgw", "uw"):
        wm = [r for r in r1s if r["family"]==fam]
        no = [r for r in r1s if r["family"]=="no_wm"]
        rng = np.random.default_rng(SEED)
        target = min(len(wm), len(no))
        if len(wm) > target:
            idx = rng.choice(len(wm), size=target, replace=False)
            wm = [wm[i] for i in sorted(idx)]
        if len(no) > target:
            idx = rng.choice(len(no), size=target, replace=False)
            no = [no[i] for i in sorted(idx)]
        z_wm = np.array([r["z_self"] for r in wm])
        z_no = np.array([r[f"z_{fam}"] for r in no])
        m = boot(z_wm, z_no)
        mean_len_wm = float(np.mean([r["n_tokens"] for r in wm]))
        mean_len_no = float(np.mean([r["n_tokens"] for r in no]))
        out["cells"].append({"family": fam, "cell": "R1-short",
                             "n_wm": len(z_wm), "n_no": len(z_no),
                             "mean_len_wm": mean_len_wm,
                             "mean_len_no": mean_len_no, **m})
        print(f"  {fam:3s} R1-short  n={len(z_wm)}/{len(z_no)}  "
              f"mean_len_wm={mean_len_wm:.1f}  mean_len_no={mean_len_no:.1f}  "
              f"MI={m['mi']:.3f} CI=[{m['mi_ci'][0]:.3f},{m['mi_ci'][1]:.3f}]  "
              f"AUROC={m['auroc']:.3f}  TPR1={m['tpr1']:.3f}  TPR5={m['tpr5']:.3f}")

    print("\n=== Canonical R0 / R1-original metrics (for cross-comparison) ===")
    for fam in ("kgw", "uw"):
        for regime in ("R0", "R1"):
            wm, no = collect_canonical(regime, fam)
            z_wm = np.array([r["z_self"] for r in wm])
            z_no = np.array([r[f"z_{fam}"] for r in no])
            m = boot(z_wm, z_no)
            mean_len_wm = float(np.mean([r["n_tokens"] for r in wm]))
            cell = "R0" if regime == "R0" else "R1-original"
            out["cells"].append({"family": fam, "cell": cell,
                                 "n_wm": len(z_wm), "n_no": len(z_no),
                                 "mean_len_wm": mean_len_wm, **m})
            print(f"  {fam:3s} {cell:11s}  n={len(z_wm)}/{len(z_no)}  "
                  f"mean_len_wm={mean_len_wm:.1f}  MI={m['mi']:.3f}  "
                  f"CI=[{m['mi_ci'][0]:.3f},{m['mi_ci'][1]:.3f}]  "
                  f"AUROC={m['auroc']:.3f}")

    print("\n=== Source-paired ratios ===")
    # R1-short / R0 ratio
    for fam in ("kgw", "uw"):
        # R0 paired pool: variant A R0 wm + no_wm
        r0_wm = {r["item_id"]: r["z_self"] for r in zidx
                 if corp(r)=="syn" and r["regime"]=="R0" and r["family"]==fam
                 and r["strength"]==PEAK and r.get("variant","A")=="A"}
        r0_no = [r[f"z_{fam}"] for r in zidx if corp(r)=="syn" and r["regime"]=="R0"
                 and r["family"]=="no_wm" and r.get("variant","A")=="A"]
        # R1-short paired pool
        r1s_wm = {r["item_id"]: r["z_self"] for r in r1s if r["family"]==fam}
        r1s_no = [r[f"z_{fam}"] for r in r1s if r["family"]=="no_wm"]
        # R1-original paired pool
        r1o_wm = {r["item_id"]: r["z_self"] for r in zidx
                  if corp(r)=="syn" and r["regime"]=="R1" and r["family"]==fam
                  and r["strength"]==PEAK and r.get("variant","A")=="A"}
        r1o_no = [r[f"z_{fam}"] for r in zidx if corp(r)=="syn" and r["regime"]=="R1"
                  and r["family"]=="no_wm" and r.get("variant","A")=="A"]

        for label, wmL_by_id, noL in [("R1-short", r1s_wm, r1s_no),
                                        ("R1-original", r1o_wm, r1o_no)]:
            ratio, lo, hi, npairs = boot_ratio(r0_wm, np.array(r0_no),
                                                wmL_by_id, np.array(noL))
            out["ratios_source_paired"].append({
                "family": fam, "ratio_label": f"{label}/R0",
                "ratio": ratio, "ci_lo": lo, "ci_hi": hi,
                "n_pairs": npairs})
            print(f"  {fam:3s} {label}/R0 ratio={ratio:.2f}x [{lo:.2f},{hi:.2f}]  "
                  f"n_pairs={npairs}")

    Path("./data/runs/round11").mkdir(parents=True, exist_ok=True)
    json.dump(out,
              open("./data/runs/round11/r1short_metrics.json",
                   "w", encoding="utf-8"), indent=2, default=str)
    print("\nsaved runs/round11/r1short_metrics.json")


if __name__ == "__main__":
    main()
