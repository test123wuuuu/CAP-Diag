"""Round 10 R10-7: Length-constrained R1 disentanglement.

Goal: test whether the R0 vs R1 detector-statistic-accessible MI gap remains
when length is matched in a narrow band (35-45 tokens). Three cells:
  - R0-band: R0 records with n_tokens in [35, 45]
  - R1-short: R1 records with n_tokens in [35, 45]
  - R1-original: full R1 distribution
Same canonical estimator/bootstrap as Tables 1-4: clipped Miller-Madow,
20 equal-width bins, B=500 source-paired bootstrap, seed 20260513.
Empirical prior pi=0.5: n_wm=n_no per cell after downsampling.

For each (family in {kgw, uw}):
  - report MI, CI, AUROC, TPR1, TPR5, mean length, n_wm, n_no
  - source-paired R1-band/R0-band ratio (where item_ids overlap)
  - source-paired R1-original/R0-band ratio for reference

Output:
  experiments/idea5/runs/round10/length_disentanglement.json
"""
import json, sys, math
from pathlib import Path
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SEED = 20260513
N_BOOT = 500
PEAK = 5.0
BAND = (35, 45)


def mm_h(c):
    n = c.sum()
    if n == 0: return 0.0
    p = c / n; p = p[p > 0]
    H = -np.sum(p * np.log2(p))
    K = (p > 0).sum()
    return H + (K - 1) / (2 * n * math.log(2))


def mi(z_wm, z_no, n_bins=20):
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


def boot_ci(z_wm, z_no, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n_w, n_n = len(z_wm), len(z_no)
    if n_w < 5 or n_n < 5:
        return {"mi": float("nan"), "mi_ci": [float("nan")] * 2,
                "auroc": float("nan"), "tpr1": float("nan"), "tpr5": float("nan")}
    mi_p = mi(z_wm, z_no); au_p = auroc(z_wm, z_no)
    t1 = tpr_at(z_wm, z_no, 0.01); t5 = tpr_at(z_wm, z_no, 0.05)
    mi_b = np.empty(n_boot)
    for b in range(n_boot):
        iw = rng.integers(0, n_w, size=n_w)
        iN = rng.integers(0, n_n, size=n_n)
        mi_b[b] = mi(z_wm[iw], z_no[iN])
    return {"mi": mi_p,
            "mi_ci": [float(np.percentile(mi_b, 2.5)), float(np.percentile(mi_b, 97.5))],
            "auroc": au_p, "tpr1": t1, "tpr5": t5}


def boot_paired_ratio(wm0_by_id, no0, wm1_by_id, no1, n_boot=N_BOOT, seed=SEED):
    """Source-paired ratio MI(R1)/MI(R0)."""
    rng = np.random.default_rng(seed)
    common = sorted(set(wm0_by_id) & set(wm1_by_id))
    if len(common) < 10:
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
    rows = [json.loads(l) for l in open("./data/data/zscore_index_round6.jsonl",
                                         encoding="utf-8")]

    def corp(r):
        return "nat" if "natural" in r.get("src_file", "") else "syn"

    def pick(regime, family, length_band=None):
        recs = [r for r in rows
                if corp(r) == "syn" and r["regime"] == regime
                and r["family"] == family
                and r.get("strength", 0) in (PEAK, 0.0)
                and r.get("variant", "A") == "A"]
        if length_band is not None:
            lo, hi = length_band
            recs = [r for r in recs if lo <= r["n_tokens"] <= hi]
        return recs

    result = {"band": list(BAND), "n_boot": N_BOOT, "seed": SEED, "peak": PEAK,
              "cells": [], "ratios_source_paired": []}

    print(f"=== Length-constrained R1 disentanglement, band {BAND} ===")
    for fam in ("kgw", "uw"):
        # R0-band: R0 records in band
        r0_wm_band = pick("R0", fam, length_band=BAND)
        r0_no_band = pick("R0", "no_wm", length_band=BAND)
        # R1-short: R1 records in band
        r1s_wm_band = pick("R1", fam, length_band=BAND)
        r1s_no_band = pick("R1", "no_wm", length_band=BAND)
        # R1-original: full R1
        r1o_wm = pick("R1", fam)
        r1o_no = pick("R1", "no_wm")

        # Downsample no_wm to match wm count for pi=0.5 per cell
        def balance(wm_list, no_list, label):
            rng = np.random.default_rng(SEED)
            target = min(len(wm_list), len(no_list))
            if len(wm_list) > target:
                idx = rng.choice(len(wm_list), size=target, replace=False)
                wm_list = [wm_list[i] for i in sorted(idx)]
            if len(no_list) > target:
                idx = rng.choice(len(no_list), size=target, replace=False)
                no_list = [no_list[i] for i in sorted(idx)]
            return wm_list, no_list

        r0_wm_b, r0_no_b = balance(r0_wm_band, r0_no_band, "R0-band")
        r1s_wm_b, r1s_no_b = balance(r1s_wm_band, r1s_no_band, "R1-short")
        r1o_wm_b, r1o_no_b = balance(r1o_wm, r1o_no, "R1-original")

        for label, wm, no, no_field in [
            ("R0-band", r0_wm_b, r0_no_b, f"z_{fam}"),
            ("R1-short", r1s_wm_b, r1s_no_b, f"z_{fam}"),
            ("R1-original", r1o_wm_b, r1o_no_b, f"z_{fam}"),
        ]:
            if not wm or not no: continue
            z_wm = np.array([r["z_self"] for r in wm])
            z_no = np.array([r[no_field] for r in no])
            m = boot_ci(z_wm, z_no)
            mean_len_wm = float(np.mean([r["n_tokens"] for r in wm]))
            mean_len_no = float(np.mean([r["n_tokens"] for r in no]))
            row = {"family": fam, "cell": label,
                   "n_wm": len(z_wm), "n_no": len(z_no),
                   "mean_len_wm": mean_len_wm, "mean_len_no": mean_len_no,
                   **m}
            result["cells"].append(row)
            print(f"  {fam:3s} {label:13s}  n={len(z_wm):>3d}/{len(z_no):>3d}  "
                  f"mean_len_wm={mean_len_wm:.1f}  mean_len_no={mean_len_no:.1f}  "
                  f"MI={m['mi']:.3f}  CI=[{m['mi_ci'][0]:.3f},{m['mi_ci'][1]:.3f}]  "
                  f"AUROC={m['auroc']:.3f}  TPR1={m['tpr1']:.3f}  TPR5={m['tpr5']:.3f}")

        # Source-paired ratios R1-short / R0-band and R1-original / R0-band
        wm0_by_id = {r["item_id"]: r["z_self"] for r in r0_wm_band}
        no0 = np.array([r[f"z_{fam}"] for r in r0_no_band])
        wm1s_by_id = {r["item_id"]: r["z_self"] for r in r1s_wm_band}
        no1s = np.array([r[f"z_{fam}"] for r in r1s_no_band])
        wm1o_by_id = {r["item_id"]: r["z_self"] for r in r1o_wm}
        no1o = np.array([r[f"z_{fam}"] for r in r1o_no])

        ratio_s, r_lo_s, r_hi_s, n_pairs_s = boot_paired_ratio(
            wm0_by_id, no0, wm1s_by_id, no1s)
        ratio_o, r_lo_o, r_hi_o, n_pairs_o = boot_paired_ratio(
            wm0_by_id, no0, wm1o_by_id, no1o)
        result["ratios_source_paired"].append({
            "family": fam,
            "ratio_R1short_over_R0band": ratio_s,
            "ratio_R1short_ci": [r_lo_s, r_hi_s],
            "n_pairs_R1short": n_pairs_s,
            "ratio_R1original_over_R0band": ratio_o,
            "ratio_R1original_ci": [r_lo_o, r_hi_o],
            "n_pairs_R1original": n_pairs_o,
        })
        print(f"  {fam:3s} source-paired ratio R1-short / R0-band = {ratio_s:.2f}x "
              f"[{r_lo_s:.2f}, {r_hi_s:.2f}]  n_pairs={n_pairs_s}")
        print(f"  {fam:3s} source-paired ratio R1-original / R0-band = {ratio_o:.2f}x "
              f"[{r_lo_o:.2f}, {r_hi_o:.2f}]  n_pairs={n_pairs_o}")

    Path("./data/runs/round10").mkdir(parents=True, exist_ok=True)
    json.dump(result,
              open("./data/runs/round10/length_disentanglement.json",
                   "w", encoding="utf-8"), indent=2, default=str)
    print("\nsaved")


if __name__ == "__main__":
    main()
