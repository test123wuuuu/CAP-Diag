"""Compute mutual information estimates between watermark flag and detector statistic.

Implements clipped Miller-Madow MI estimator with source-paired bootstrap
confidence intervals as described in the paper.
"""
import argparse
import json
import math
from collections import defaultdict

import numpy as np


def miller_madow_entropy(counts):
    """Compute Miller-Madow bias-corrected entropy estimate.

    Args:
        counts: Array of bin counts

    Returns:
        Entropy in bits
    """
    n = counts.sum()
    if n == 0:
        return 0.0

    probs = counts / n
    probs = probs[probs > 0]

    # Shannon entropy
    H = -np.sum(probs * np.log2(probs))

    # Miller-Madow bias correction
    K = (probs > 0).sum()
    bias_correction = (K - 1) / (2 * n * math.log(2))

    return H + bias_correction


def compute_mi(z_watermarked, z_unwatermarked, n_bins=20):
    """Compute mutual information I(W; S) between watermark flag and statistic.

    Args:
        z_watermarked: Detection statistics for watermarked texts
        z_unwatermarked: Detection statistics for unwatermarked texts
        n_bins: Number of histogram bins

    Returns:
        MI estimate in bits
    """
    if len(z_watermarked) < 5 or len(z_unwatermarked) < 5:
        return 0.0

    all_z = np.concatenate([z_watermarked, z_unwatermarked])
    z_min, z_max = float(all_z.min()), float(all_z.max())

    if z_max - z_min < 1e-9:
        return 0.0

    # Create equal-width bins
    edges = np.linspace(z_min, z_max + 1e-9, n_bins + 1)

    # Joint distribution counts
    counts_wm = np.histogram(z_watermarked, bins=edges)[0]
    counts_no = np.histogram(z_unwatermarked, bins=edges)[0]
    counts_joint = np.column_stack([counts_wm, counts_no])

    # Marginal distribution counts
    counts_all = np.histogram(all_z, bins=edges)[0]

    # I(W;S) = H(S) - H(S|W)
    H_S = miller_madow_entropy(counts_all)

    n_total = counts_joint.sum()
    H_S_given_W = 0.0
    for w_idx in range(2):
        p_w = counts_joint[:, w_idx].sum() / n_total
        if p_w > 0:
            H_S_given_W += p_w * miller_madow_entropy(counts_joint[:, w_idx])

    mi = H_S - H_S_given_W
    return max(0.0, mi)  # Clip negative values due to finite sample


def bootstrap_ci(z_wm, z_no, item_ids_wm, item_ids_no, n_bootstrap=500, seed=42):
    """Compute source-paired bootstrap confidence interval for MI.

    Args:
        z_wm: Watermarked statistics
        z_no: Unwatermarked statistics
        item_ids_wm: Source IDs for watermarked texts
        item_ids_no: Source IDs for unwatermarked texts
        n_bootstrap: Number of bootstrap samples
        seed: Random seed

    Returns:
        (mi_estimate, lower_95ci, upper_95ci)
    """
    rng = np.random.RandomState(seed)

    # Group by source ID for paired resampling
    wm_by_id = defaultdict(list)
    no_by_id = defaultdict(list)

    for z, item_id in zip(z_wm, item_ids_wm):
        wm_by_id[item_id].append(z)

    for z, item_id in zip(z_no, item_ids_no):
        no_by_id[item_id].append(z)

    # Find common source IDs
    common_ids = sorted(set(wm_by_id.keys()) & set(no_by_id.keys()))

    if len(common_ids) < 10:
        # Fall back to unpaired bootstrap if too few common sources
        common_ids = None

    # Compute bootstrap distribution
    mi_boot = []

    for _ in range(n_bootstrap):
        if common_ids:
            # Resample source IDs (paired)
            resampled_ids = rng.choice(common_ids, size=len(common_ids), replace=True)
            boot_wm = np.concatenate([wm_by_id[i] for i in resampled_ids])
            boot_no = np.concatenate([no_by_id[i] for i in resampled_ids])
        else:
            # Unpaired resampling
            boot_wm = rng.choice(z_wm, size=len(z_wm), replace=True)
            boot_no = rng.choice(z_no, size=len(z_no), replace=True)

        mi_boot.append(compute_mi(boot_wm, boot_no))

    mi_boot = np.array(mi_boot)

    # Point estimate and 95% CI
    mi_estimate = compute_mi(z_wm, z_no)
    ci_lower = np.percentile(mi_boot, 2.5)
    ci_upper = np.percentile(mi_boot, 97.5)

    return mi_estimate, ci_lower, ci_upper


def main():
    parser = argparse.ArgumentParser(description="Compute MI estimates")
    parser.add_argument("--input", required=True,
                        help="Input JSONL with detection statistics")
    parser.add_argument("--output", required=True,
                        help="Output JSON path")
    parser.add_argument("--n_bins", type=int, default=20,
                        help="Number of histogram bins")
    parser.add_argument("--n_bootstrap", type=int, default=500,
                        help="Number of bootstrap samples")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    # Load detection statistics
    with open(args.input) as f:
        data = [json.loads(line) for line in f]

    # Separate by watermark flag
    z_wm = []
    z_no = []
    ids_wm = []
    ids_no = []

    for record in data:
        z_score = record["z_score"]
        watermarked = record.get("watermarked", record.get("watermark") != "none")
        item_id = record.get("item_id", record.get("id"))

        if watermarked:
            z_wm.append(z_score)
            ids_wm.append(item_id)
        else:
            z_no.append(z_score)
            ids_no.append(item_id)

    z_wm = np.array(z_wm)
    z_no = np.array(z_no)

    # Compute MI with bootstrap CI
    mi, ci_lower, ci_upper = bootstrap_ci(
        z_wm, z_no, ids_wm, ids_no,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed
    )

    result = {
        "mi_estimate": float(mi),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "n_watermarked": len(z_wm),
        "n_unwatermarked": len(z_no),
        "n_bins": args.n_bins,
        "n_bootstrap": args.n_bootstrap,
    }

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"MI: {mi:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")


if __name__ == "__main__":
    main()
