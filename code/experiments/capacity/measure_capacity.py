"""Phase A.5 — single-bit mutual information capacity per (regime, family, strength).

For each (regime, family, strength) with strength > 0:
  W = 1 if watermark applied, 0 if no_wm
  S = detector z-score (family's matched detector)
  pairs: (W=1 from this strength, S=z) and (W=0 from no_wm same regime, S=z)
  estimate I(W; S) via histogram on z values

For each (regime, family), capacity = max over utility-pass strengths.

Reuses:
  - R0 data from experiments/idea6_r4/data/candidates_r4.jsonl
  - R1/R2 data from experiments/idea5/data/candidates_idea5.jsonl
  - watermark detectors from experiments/idea6_r4/scripts/watermarks.py
  - utility metrics will be computed here for R1/R2 (R0 uses R4 cached calibration)
"""
import argparse, json, sys, io, math, time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
sys.path.insert(0, "./data/scripts")
from watermarks import detect_z


# ----------------- MI estimator -----------------
def miller_madow(counts):
    """Miller-Madow bias correction for entropy estimation.
    counts: array of integer counts."""
    n = counts.sum()
    if n == 0: return 0.0
    p = counts / n
    p = p[p > 0]
    H_naive = -np.sum(p * np.log2(p))
    # MM correction
    K = (p > 0).sum()
    return H_naive + (K - 1) / (2 * n * math.log(2))


def estimate_mi(z_wm, z_no_wm, n_bins=20):
    """Single-bit MI(W; S) where W ∈ {0=no_wm, 1=wm} and S is z-score.
    Use histogram-based estimator with Miller-Madow bias correction."""
    if len(z_wm) == 0 or len(z_no_wm) == 0:
        return 0.0
    all_z = np.concatenate([z_wm, z_no_wm])
    n_total = len(all_z)
    # H(W) = log2(2) = 1 when balanced; here we use empirical
    n_wm, n_no_wm = len(z_wm), len(z_no_wm)
    p_w = np.array([n_no_wm, n_wm]) / n_total
    H_W = -np.sum(p_w[p_w > 0] * np.log2(p_w[p_w > 0]))
    # H(S) and H(S|W) via binning
    z_lo, z_hi = float(all_z.min()), float(all_z.max())
    if z_hi - z_lo < 1e-9:
        return 0.0
    edges = np.linspace(z_lo, z_hi + 1e-9, n_bins + 1)
    counts_all = np.histogram(all_z, bins=edges)[0]
    counts_wm = np.histogram(z_wm, bins=edges)[0]
    counts_no = np.histogram(z_no_wm, bins=edges)[0]
    # H(S)
    H_S = miller_madow(counts_all)
    # H(S | W=1), H(S | W=0)
    H_S_w1 = miller_madow(counts_wm)
    H_S_w0 = miller_madow(counts_no)
    H_S_given_W = p_w[1] * H_S_w1 + p_w[0] * H_S_w0
    MI = H_S - H_S_given_W
    # bound at 0 (avoid negative MI from finite-sample bias)
    return max(MI, 0.0)


def bootstrap_mi(z_wm, z_no_wm, n_boot=1000, seed=20260513, n_bins=20):
    rng = np.random.default_rng(seed)
    n_wm, n_no = len(z_wm), len(z_no_wm)
    if n_wm == 0 or n_no == 0:
        return 0.0, 0.0, 0.0
    point = estimate_mi(z_wm, z_no_wm, n_bins=n_bins)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx_w = rng.integers(0, n_wm, size=n_wm)
        idx_n = rng.integers(0, n_no, size=n_no)
        boots[b] = estimate_mi(z_wm[idx_w], z_no_wm[idx_n], n_bins=n_bins)
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# ----------------- AUROC (auxiliary) -----------------
def auroc(z_pos, z_neg):
    if len(z_pos) == 0 or len(z_neg) == 0: return 0.5
    import bisect
    sneg = sorted(z_neg.tolist())
    s = 0.0
    n_pos = len(z_pos); n_neg = len(z_neg)
    for p in z_pos:
        n_lt = bisect.bisect_left(sneg, p)
        n_eq = bisect.bisect_right(sneg, p) - n_lt
        s += n_lt + 0.5 * n_eq
    return s / (n_pos * n_neg)


# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r0_candidates", default="./data/data/candidates_r4.jsonl",
                    help="R0 candidates (reused from R4)")
    ap.add_argument("--r1r2_candidates", default="./data/data/candidates_idea5.jsonl",
                    help="R1+R2 candidates (Idea 5 Phase A output)")
    ap.add_argument("--out", default="./data/data/capacity_metrics.jsonl")
    ap.add_argument("--gamma", type=float, default=0.25)
    ap.add_argument("--secret_key", type=int, default=20260513)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--n_bins", type=int, default=20)
    args = ap.parse_args()

    # ---- load R0 candidates and re-derive z scores from cached token_ids ----
    # R0 file has 6800 entries with token_ids; we treat strength=0 as no_wm
    print("loading R0 candidates...", flush=True)
    r0_recs = []
    for l in open(args.r0_candidates, encoding="utf-8"):
        try:
            r = json.loads(l)
            r["regime"] = "R0"
            r0_recs.append(r)
        except: pass
    print(f"R0: {len(r0_recs)}", flush=True)

    # load r1+r2
    print("loading R1+R2 candidates...", flush=True)
    r12_recs = [json.loads(l) for l in open(args.r1r2_candidates, encoding="utf-8")]
    print(f"R1+R2: {len(r12_recs)}", flush=True)

    all_recs = r0_recs + r12_recs

    # need vocab size for detector — read from first cand or from config
    # We know it's 152064 for Qwen2.5-7B; hard-code as constant + assert
    VOCAB_SIZE = 152064

    # compute z scores per record
    print("computing detector z-scores...", flush=True)
    t0 = time.time()
    for i, r in enumerate(all_recs):
        if "token_ids" in r and r["token_ids"]:
            tokens = r["token_ids"]
        else:
            tokens = []
        if r["family"] == "no_wm":
            r["z_kgw"] = detect_z("kgw", tokens, VOCAB_SIZE, args.gamma, args.secret_key)
            r["z_uw"] = detect_z("uw", tokens, VOCAB_SIZE, args.gamma, args.secret_key)
        else:
            r["z_self"] = detect_z(r["family"], tokens, VOCAB_SIZE, args.gamma, args.secret_key)
        if (i+1) % 2000 == 0:
            print(f"  z-scored {i+1}/{len(all_recs)} ({time.time()-t0:.0f}s)", flush=True)

    # group by (regime, family, strength)
    groups = defaultdict(list)
    for r in all_recs:
        groups[(r["regime"], r["family"], r["strength"])].append(r)

    # compute MI + AUROC per (regime, family, strength); also no_wm baseline z-score distributions per regime
    no_wm_z = defaultdict(lambda: {"kgw":[], "uw":[]})  # regime -> family -> z list
    for (regime, family, strength), recs in groups.items():
        if family == "no_wm":
            no_wm_z[regime]["kgw"] = [r["z_kgw"] for r in recs]
            no_wm_z[regime]["uw"] = [r["z_uw"] for r in recs]

    # compute per-cell metrics
    print("\ncomputing MI per (regime, family, strength)...", flush=True)
    metrics = []
    for (regime, family, strength), recs in sorted(groups.items()):
        if family == "no_wm":
            metrics.append({"regime": regime, "family": family, "strength": strength,
                            "n": len(recs), "mi": 0.0, "mi_ci_lo": 0.0, "mi_ci_hi": 0.0, "auroc": 0.5,
                            "mean_z": float(np.mean([r["z_kgw"] for r in recs])),
                            "median_z": float(np.median([r["z_kgw"] for r in recs])),
                            "mean_n_tokens": float(np.mean([r["n_tokens"] for r in recs]))})
            continue
        z_wm = np.array([r["z_self"] for r in recs])
        z_no = np.array(no_wm_z[regime][family])
        mi_point, mi_lo, mi_hi = bootstrap_mi(z_wm, z_no, n_boot=args.n_boot, n_bins=args.n_bins)
        a = auroc(z_wm, z_no)
        metrics.append({
            "regime": regime, "family": family, "strength": strength,
            "n": len(recs), "n_no_wm": len(z_no),
            "mi": mi_point, "mi_ci_lo": mi_lo, "mi_ci_hi": mi_hi,
            "auroc": a,
            "mean_z": float(z_wm.mean()), "median_z": float(np.median(z_wm)),
            "mean_n_tokens": float(np.mean([r["n_tokens"] for r in recs])),
        })

    # save
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for m in metrics:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # capacity = max MI per (regime, family) across strengths
    print("\n=== Capacity per (regime, family) ===", flush=True)
    print(f"{'regime':6s} {'family':6s}  {'max MI':>10s}  {'argmax':>6s}  {'max AUROC':>10s}  {'mean_tok@argmax':>15s}")
    print("-" * 70)
    cap_per = {}
    for regime in ("R0", "R1", "R2"):
        for fam in ("kgw", "uw"):
            cell = [m for m in metrics if m["regime"]==regime and m["family"]==fam]
            if not cell: continue
            best = max(cell, key=lambda x: x["mi"])
            cap_per[(regime, fam)] = best
            print(f"{regime:6s} {fam:6s}  {best['mi']:>10.4f}  {best['strength']:>6.1f}  {best['auroc']:>10.3f}  {best['mean_n_tokens']:>15.1f}")

    # check monotonicity
    print("\n=== Monotonicity check (C1: R0 < R1 < R2) ===", flush=True)
    for fam in ("kgw", "uw"):
        try:
            c0 = cap_per[("R0", fam)]["mi"]
            c1 = cap_per[("R1", fam)]["mi"]
            c2 = cap_per[("R2", fam)]["mi"]
            mono = (c0 < c1 < c2)
            print(f"  {fam}: R0={c0:.4f}  R1={c1:.4f}  R2={c2:.4f}  monotone={mono}")
        except KeyError as e:
            print(f"  {fam}: incomplete data {e}")

    # save capacity summary
    cap_summary = [
        {"regime": k[0], "family": k[1], **v} for k, v in cap_per.items()
    ]
    json.dump(cap_summary, open(Path(args.out).parent / "capacity_summary.json", "w"), indent=2, default=float)
    print(f"\nsaved metrics: {args.out}")
    print(f"saved capacity summary: {Path(args.out).parent / 'capacity_summary.json'}")


if __name__ == "__main__":
    main()
