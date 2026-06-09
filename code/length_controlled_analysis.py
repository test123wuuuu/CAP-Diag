"""Length-controlled MI analysis to disentangle length effects from regime effects.

Generates R1 texts constrained to match R0 length distribution (30-50 tokens)
to test whether the MI gap persists when length is controlled.
"""
import argparse
import json
import numpy as np
from collections import defaultdict


def compute_length_matched_mi(r0_data, r1_data, r1_short_data, target_band=(30, 50)):
    """Compare MI across length-matched regimes.

    Args:
        r0_data: R0 records with detection statistics
        r1_data: R1 records (full length distribution)
        r1_short_data: R1 records constrained to target_band
        target_band: Length range (min, max) tokens

    Returns:
        Dict with MI estimates and comparisons
    """
    from compute_mi_estimates import compute_mi, bootstrap_ci

    # Filter R0 to target band
    r0_band = [r for r in r0_data
               if target_band[0] <= r.get("length", 0) <= target_band[1]]

    # Filter R1-short to target band
    r1_short_band = [r for r in r1_short_data
                     if target_band[0] <= r.get("length", 0) <= target_band[1]]

    # Extract statistics
    def extract_stats(records):
        z_wm = [r["z_score"] for r in records if r.get("watermarked", True)]
        z_no = [r["z_score"] for r in records if not r.get("watermarked", True)]
        ids_wm = [r["item_id"] for r in records if r.get("watermarked", True)]
        ids_no = [r["item_id"] for r in records if not r.get("watermarked", True)]
        return np.array(z_wm), np.array(z_no), ids_wm, ids_no

    # Compute MI for each condition
    r0_stats = extract_stats(r0_band)
    r1_short_stats = extract_stats(r1_short_band)

    mi_r0, ci_r0_low, ci_r0_high = bootstrap_ci(*r0_stats)
    mi_r1_short, ci_r1s_low, ci_r1s_high = bootstrap_ci(*r1_short_stats)

    # Compute paired ratio for common sources
    r0_by_id = defaultdict(list)
    r1s_by_id = defaultdict(list)

    for r in r0_band:
        if r.get("watermarked"):
            r0_by_id[r["item_id"]].append(r["z_score"])

    for r in r1_short_band:
        if r.get("watermarked"):
            r1s_by_id[r["item_id"]].append(r["z_score"])

    common_ids = sorted(set(r0_by_id.keys()) & set(r1s_by_id.keys()))

    # Source-paired MI ratio
    ratios = []
    for item_id in common_ids:
        # Simplified paired comparison (actual implementation uses bootstrap)
        if len(r0_by_id[item_id]) > 0 and len(r1s_by_id[item_id]) > 0:
            ratios.append(np.mean(r1s_by_id[item_id]) / (np.mean(r0_by_id[item_id]) + 1e-9))

    ratio_median = np.median(ratios) if ratios else 1.0
    ratio_ci = (np.percentile(ratios, 2.5), np.percentile(ratios, 97.5)) if ratios else (1.0, 1.0)

    result = {
        "r0_band": {
            "mi": float(mi_r0),
            "ci": [float(ci_r0_low), float(ci_r0_high)],
            "n_pairs": len(r0_band),
            "mean_length": np.mean([r.get("length", 0) for r in r0_band])
        },
        "r1_short_band": {
            "mi": float(mi_r1_short),
            "ci": [float(ci_r1s_low), float(ci_r1s_high)],
            "n_pairs": len(r1_short_band),
            "mean_length": np.mean([r.get("length", 0) for r in r1_short_band])
        },
        "ratio": {
            "r1_short_over_r0": float(ratio_median),
            "ci": [float(ratio_ci[0]), float(ratio_ci[1])],
            "n_common_sources": len(common_ids)
        }
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Length-controlled MI analysis")
    parser.add_argument("--r0_data", required=True, help="R0 data JSONL")
    parser.add_argument("--r1_data", required=True, help="R1 data JSONL")
    parser.add_argument("--r1_short_data", required=True, help="R1-short data JSONL")
    parser.add_argument("--output", required=True, help="Output JSON")
    parser.add_argument("--target_band", nargs=2, type=int, default=[30, 50],
                        help="Target length band (min max)")

    args = parser.parse_args()

    # Load data
    with open(args.r0_data) as f:
        r0_data = [json.loads(line) for line in f]

    with open(args.r1_data) as f:
        r1_data = [json.loads(line) for line in f]

    with open(args.r1_short_data) as f:
        r1_short_data = [json.loads(line) for line in f]

    # Compute length-matched comparison
    result = compute_length_matched_mi(
        r0_data, r1_data, r1_short_data,
        target_band=tuple(args.target_band)
    )

    # Save result
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"R0 MI: {result['r0_band']['mi']:.3f}")
    print(f"R1-short MI: {result['r1_short_band']['mi']:.3f}")
    print(f"Ratio: {result['ratio']['r1_short_over_r0']:.2f}x")


if __name__ == "__main__":
    main()
