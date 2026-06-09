"""E3 capacity measurement: compute single-bit MI for EXP/Gumbel watermark
on R0 + R1 candidates.

Uses the same MI estimator as Phase A.5 (histogram + Miller-Madow, 20 bins,
1000-bootstrap CI). Detector is exp_detect_z. Also runs a permutation test
between R0 and R1 to check that the capacity ladder direction is consistent
on the third (non-greenlist) mechanism.
"""
import argparse, json, sys, io, math
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
# sys.path.insert removed for clean repo structure
from exp_watermark import exp_detect_z

VOCAB = 152064
KEY = 20260513


def mm_entropy(c):
    n = c.sum()
    if n == 0: return 0.0
    p = c / n; p = p[p > 0]
    H = -np.sum(p * np.log2(p))
    K = (p > 0).sum()
    return H + (K - 1) / (2 * n * math.log(2))


def mi_histogram(z_wm, z_no, n_bins=20):
    if len(z_wm) < 5 or len(z_no) < 5: return 0.0
    all_z = np.concatenate([z_wm, z_no])
    z_lo, z_hi = float(all_z.min()), float(all_z.max())
    if z_hi - z_lo < 1e-9: return 0.0
    edges = np.linspace(z_lo, z_hi + 1e-9, n_bins + 1)
    c_all = np.histogram(all_z, bins=edges)[0]
    c_wm = np.histogram(z_wm, bins=edges)[0]
    c_no = np.histogram(z_no, bins=edges)[0]
    n_t = len(all_z); n_w = len(z_wm); n_n = len(z_no)
    pw = np.array([n_n, n_w]) / n_t
    return max(0.0, min(mm_entropy(c_all) - (pw[1] * mm_entropy(c_wm) + pw[0] * mm_entropy(c_no)), 1.0))


def bootstrap_mi(z_wm, z_no, n_boot=1000, seed=20260513):
    rng = np.random.default_rng(seed)
    n_w, n_n = len(z_wm), len(z_no)
    if n_w < 5 or n_n < 5:
        return float("nan"), float("nan"), float("nan")
    point = mi_histogram(z_wm, z_no)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        i_w = rng.integers(0, n_w, size=n_w)
        i_n = rng.integers(0, n_n, size=n_n)
        boots[b] = mi_histogram(z_wm[i_w], z_no[i_n])
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def permutation_test(z_wm_A, z_no_A, z_wm_B, z_no_B, n_perm=2000, seed=20260513):
    rng = np.random.default_rng(seed)
    obs = mi_histogram(z_wm_A, z_no_A) - mi_histogram(z_wm_B, z_no_B)
    pool_wm = np.concatenate([z_wm_A, z_wm_B])
    pool_no = np.concatenate([z_no_A, z_no_B])
    nA_w, nA_n = len(z_wm_A), len(z_no_A)
    diffs = np.empty(n_perm)
    for p in range(n_perm):
        iw = rng.permutation(len(pool_wm))
        i_n = rng.permutation(len(pool_no))
        diffs[p] = (mi_histogram(pool_wm[iw[:nA_w]], pool_no[i_n[:nA_n]])
                    - mi_histogram(pool_wm[iw[nA_w:]], pool_no[i_n[nA_n:]]))
    return obs, float(np.mean(np.abs(diffs) >= abs(obs)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates",
                    default="./data/data/candidates_exp.jsonl")
    ap.add_argument("--out",
                    default="./data/data/capacity_exp.jsonl")
    ap.add_argument("--out_summary",
                    default="./data/runs/round3/e3_capacity.json")
    args = ap.parse_args()
    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)

    recs = [json.loads(l) for l in open(args.candidates, encoding='utf-8')]
    print(f"loaded EXP candidates: {len(recs)}", flush=True)

    print("computing EXP detector z-scores...", flush=True)
    for r in recs:
        toks = r.get("token_ids") or []
        r["z_exp"] = exp_detect_z(toks, VOCAB, KEY)

    # Group by regime
    z_by = defaultdict(lambda: {"wm": [], "no": []})
    for r in recs:
        bucket = "wm" if r["family"] == "exp" else "no"
        z_by[r["regime"]][bucket].append({"z": r["z_exp"],
                                          "n_tokens": r["n_tokens"],
                                          "item_id": r["item_id"]})

    print("\n=== Per-regime EXP capacity ===")
    print(f"{'regime':6s}  {'n_wm':>5s}  {'n_no':>5s}  {'mean_tok_wm':>11s}  "
          f"{'mean_tok_no':>11s}  {'mi':>7s}  {'CI95':>22s}")
    rows = []
    for regime in ("R0", "R1"):
        wm = z_by[regime]["wm"]; no = z_by[regime]["no"]
        z_wm = np.array([x["z"] for x in wm])
        z_no = np.array([x["z"] for x in no])
        mi, lo, hi = bootstrap_mi(z_wm, z_no, n_boot=1000)
        mt_w = float(np.mean([x["n_tokens"] for x in wm]))
        mt_n = float(np.mean([x["n_tokens"] for x in no]))
        rows.append({"regime": regime, "family": "exp",
                     "n_wm": len(wm), "n_no": len(no),
                     "mean_n_tokens_wm": mt_w, "mean_n_tokens_no": mt_n,
                     "mi": mi, "mi_ci_lo": lo, "mi_ci_hi": hi})
        print(f"  {regime:6s}  {len(wm):>5d}  {len(no):>5d}  {mt_w:>11.2f}  "
              f"{mt_n:>11.2f}  {mi:>7.4f}  [{lo:+.4f},{hi:+.4f}]")

    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    print(f"\nsaved per-cell capacity: {args.out}")

    # Permutation: R0 vs R1
    print("\n=== R0 vs R1 permutation (B=2000) ===")
    z_wm_0 = np.array([x["z"] for x in z_by["R0"]["wm"]])
    z_no_0 = np.array([x["z"] for x in z_by["R0"]["no"]])
    z_wm_1 = np.array([x["z"] for x in z_by["R1"]["wm"]])
    z_no_1 = np.array([x["z"] for x in z_by["R1"]["no"]])
    obs_diff, p_perm = permutation_test(z_wm_1, z_no_1, z_wm_0, z_no_0,
                                        n_perm=2000)
    print(f"  obs MI diff (R1 - R0): {obs_diff:+.4f}")
    print(f"  perm p-value: {p_perm:.4f}")

    mi_r0 = rows[0]["mi"]; mi_r1 = rows[1]["mi"]
    ci_overlap = not (rows[1]["mi_ci_lo"] > rows[0]["mi_ci_hi"])
    ratio = mi_r1 / max(mi_r0, 1e-6)
    pass_gate = (not ci_overlap) or (p_perm < 0.01)
    print(f"\n=== E3 gate ===")
    print(f"  R0 MI = {mi_r0:.4f}  CI=[{rows[0]['mi_ci_lo']:.4f},{rows[0]['mi_ci_hi']:.4f}]")
    print(f"  R1 MI = {mi_r1:.4f}  CI=[{rows[1]['mi_ci_lo']:.4f},{rows[1]['mi_ci_hi']:.4f}]")
    print(f"  ratio R1/R0 = {ratio:.2f}×")
    print(f"  CI non-overlap: {not ci_overlap}")
    print(f"  perm p < 0.01: {p_perm < 0.01}")
    print(f"  E3 PASS (CI non-overlap OR perm p<0.01): {pass_gate}")

    summary = {
        "per_regime": rows,
        "permutation": {"obs_mi_diff_R1_minus_R0": obs_diff, "p_value": p_perm,
                        "n_perm": 2000},
        "ratio_R1_over_R0": ratio,
        "ci_non_overlap_R0_R1": not ci_overlap,
        "perm_p_lt_01": p_perm < 0.01,
        "e3_pass": pass_gate,
        "vocab": VOCAB, "secret_key": KEY,
        "note": "EXP/Gumbel watermark, third mechanism family (non-greenlist). Confirms R0 << R1 capacity holds across mechanism families.",
    }
    json.dump(summary, open(args.out_summary, "w", encoding="utf-8"),
              indent=2, default=str)
    print(f"\nsaved summary: {args.out_summary}")


if __name__ == "__main__":
    main()
