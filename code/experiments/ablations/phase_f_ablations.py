"""Phase F — Ablations.

A1 Regime split (already in Phase D-Idea5 / E — replicate per family)
A2 Proxy leave-one-component-out: drop count, drop radius, drop ppl → recompute correlation
A3 Length-control: stratify by length quartile across all regimes
A4 Family split: KGW-only vs UW-only fit (already done in Phase E)
A5 Utility-gate sensitivity: relax NLI threshold; see capacity at less strict gate
"""
import argparse, json, sys, io
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")


def spearman_rho(x, y):
    n = len(x)
    if n < 2: return float('nan')
    rx = np.argsort(np.argsort(np.array(x))).astype(float)
    ry = np.argsort(np.argsort(np.array(y))).astype(float)
    mx, my = rx.mean(), ry.mean()
    num = ((rx - mx) * (ry - my)).sum()
    den = np.sqrt(((rx - mx)**2).sum() * ((ry - my)**2).sum())
    return float(num / den) if den > 0 else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capacity", default="./data/data/capacity_metrics.jsonl")
    ap.add_argument("--h_proxy", default="./data/data/h_proxy.jsonl")
    ap.add_argument("--candidates", default="./data/data/candidates_idea5.jsonl")
    ap.add_argument("--r0_candidates", default="./data/data/candidates_r4.jsonl")
    ap.add_argument("--out_dir", default="./data/runs/phase_f")
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # capacity
    metrics = [json.loads(l) for l in open(args.capacity, encoding='utf-8')]
    by_rf = defaultdict(list)
    for r in metrics:
        if r["family"] == "no_wm": continue
        by_rf[(r["regime"], r["family"])].append(r)
    capacity_per = {k: max(v, key=lambda x: x["mi"]) for k, v in by_rf.items()}

    # h_proxy
    proxies = [json.loads(l) for l in open(args.h_proxy, encoding='utf-8')]
    by_r = defaultdict(list)
    for r in proxies: by_r[r["regime"]].append(r)

    EPS = 1e-3
    def h_full(r):
        c, rad, p = r["feasible_count"], r["embedding_radius"], r["ppl_ratio"]
        ppl_t = 0.0 if (p != p or p == float('inf')) else np.log(max(p, EPS))
        return np.log(c + 1) + rad - ppl_t
    def h_no_count(r):
        rad, p = r["embedding_radius"], r["ppl_ratio"]
        ppl_t = 0.0 if (p != p or p == float('inf')) else np.log(max(p, EPS))
        return rad - ppl_t
    def h_no_radius(r):
        c, p = r["feasible_count"], r["ppl_ratio"]
        ppl_t = 0.0 if (p != p or p == float('inf')) else np.log(max(p, EPS))
        return np.log(c + 1) - ppl_t
    def h_no_ppl(r):
        c, rad = r["feasible_count"], r["embedding_radius"]
        return np.log(c + 1) + rad

    # =========================================================
    # A2 Proxy leave-one-component-out
    # =========================================================
    print("\n=== A2: proxy leave-one-component-out ===")
    a2_rows = []
    for tag, fn in [("full", h_full), ("no_count", h_no_count),
                    ("no_radius", h_no_radius), ("no_ppl", h_no_ppl)]:
        H = {r: float(np.mean([fn(x) for x in by_r[r]])) for r in by_r}
        x = [H[regime] for (regime, fam) in capacity_per.keys()]
        y = [v["mi"] for v in capacity_per.values()]
        rho = spearman_rho(x, y)
        a2_rows.append({"variant": tag, "spearman_rho": rho, "H_per_regime": H})
        print(f"  {tag:10s}  ρ={rho:+.4f}  H(R0)={H['R0']:.3f}  H(R1)={H['R1']:.3f}  H(R2)={H['R2']:.3f}")

    # =========================================================
    # A3 Length-control: redo capacity at constant length subsets
    # =========================================================
    # Stratify candidates by token-count buckets across all regimes
    print("\n=== A3: length-control (does R0 < R1 < R2 persist at same length?) ===")
    # Load candidates from both files to know lengths and z-scores
    # Use existing capacity_metrics z-scores instead by re-walking the data
    # Simplification: report token-count distribution by regime to verify length differs
    r0_cands = [json.loads(l) for l in open(args.r0_candidates, encoding='utf-8')]
    r12_cands = [json.loads(l) for l in open(args.candidates, encoding='utf-8')]
    for r in r0_cands: r["regime"] = "R0"
    all_cands = r0_cands + r12_cands
    by_reg_tok = defaultdict(list)
    for r in all_cands:
        if r["family"] in ("no_wm",): continue
        by_reg_tok[r["regime"]].append(r["n_tokens"])
    print("  Token count distribution per regime:")
    for r in ("R0","R1","R2"):
        if r in by_reg_tok:
            xs = by_reg_tok[r]
            print(f"    {r}: mean={np.mean(xs):.1f}, median={np.median(xs):.0f}, p10={np.percentile(xs,10):.0f}, p90={np.percentile(xs,90):.0f}")

    # Note: full length-control would re-estimate capacity on common-length subset; this is
    # left as a methodological observation here since the regimes have non-overlapping length
    # distributions by construction (R0 ~32, R1 ~50, R2 ~110). True length-control would
    # require generating same-length text at different semantic constraints, which is the
    # whole point of R0 vs R1 prompt design — they ARE the length-controlled cases at the
    # token-count *intersection*. We report distributions and note overlap.
    overlap_r0_r1 = (max(np.percentile(by_reg_tok["R0"],90), np.percentile(by_reg_tok["R1"],10)),
                     min(np.percentile(by_reg_tok["R0"],10), np.percentile(by_reg_tok["R1"],90)))
    print(f"  Note: R0/R1 length distributions are mostly disjoint by construction.")
    print(f"  Length-control conclusion: regime effect ≠ pure length effect requires per-source within-length analysis.")
    print(f"  This is reported as a known limitation; H_proxy contains length-aware components (radius, ppl).")

    # =========================================================
    # A4 Family split — already shown in C1 / Phase E; print summary
    # =========================================================
    print("\n=== A4: family split ===")
    for fam in ("kgw","uw"):
        c0 = capacity_per[("R0", fam)]["mi"]
        c1 = capacity_per[("R1", fam)]["mi"]
        c2 = capacity_per[("R2", fam)]["mi"]
        print(f"  {fam}: R0={c0:.4f}  R1={c1:.4f}  R2={c2:.4f}  monotone={c0<c1<c2}")

    # =========================================================
    # save
    # =========================================================
    summary = {
        "A2_proxy_loocv": a2_rows,
        "A3_length_dist": {r: {"mean": float(np.mean(by_reg_tok[r])),
                               "median": float(np.median(by_reg_tok[r])),
                               "p10": float(np.percentile(by_reg_tok[r],10)),
                               "p90": float(np.percentile(by_reg_tok[r],90))}
                           for r in by_reg_tok},
        "A4_per_family": {fam: {"R0": capacity_per[("R0",fam)]["mi"],
                                "R1": capacity_per[("R1",fam)]["mi"],
                                "R2": capacity_per[("R2",fam)]["mi"]}
                          for fam in ("kgw","uw")},
    }
    json.dump(summary, open(out_dir / "phase_f_analysis.json", "w", encoding="utf-8"),
              indent=2, default=str)

    md = ["# Phase F Ablations", "",
          "## A2: Proxy leave-one-component-out", "",
          "| variant | Spearman ρ |",
          "|---|---:|"]
    for row in a2_rows:
        md.append(f"| {row['variant']} | {row['spearman_rho']:+.4f} |")
    md += ["", "## A3: Length distribution per regime", "",
           "| regime | mean | median | p10 | p90 |",
           "|---|---:|---:|---:|---:|"]
    for r in ("R0","R1","R2"):
        if r in by_reg_tok:
            xs = by_reg_tok[r]
            md.append(f"| {r} | {np.mean(xs):.1f} | {np.median(xs):.0f} | {np.percentile(xs,10):.0f} | {np.percentile(xs,90):.0f} |")
    md += ["",
           "Note: R0/R1/R2 length distributions are intentionally non-overlapping by prompt design.",
           "Within-length capacity comparison requires either matched-length truncation or per-source MI",
           "estimation — deferred to Phase G if pilot positive.",
           "", "## A4: Per-family monotonicity", ""]
    for fam in ("kgw","uw"):
        c0 = capacity_per[("R0", fam)]["mi"]
        c1 = capacity_per[("R1", fam)]["mi"]
        c2 = capacity_per[("R2", fam)]["mi"]
        md.append(f"- {fam}: R0={c0:.4f} → R1={c1:.4f} → R2={c2:.4f}; monotone = {c0<c1<c2}")
    (out_dir / "phase_f_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nsaved: {out_dir / 'phase_f_summary.md'}")


if __name__ == "__main__":
    main()
