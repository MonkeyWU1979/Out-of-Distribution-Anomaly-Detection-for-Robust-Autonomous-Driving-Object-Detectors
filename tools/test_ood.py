"""Full OOD evaluation: aggregate metrics + dual-pathway warning quality.

Steps:
1. Load a fitted Mahalanobis model.
2. Score ID validation features and OOD test features.
3. Calibrate dual-pathway thresholds on the ID scores (target FPR).
4. Report AU-ROC / AU-PR, per-pathway FPR/FNR, and per-category breakdown.

OOD features file may contain a `categories` object-array (per-image corner-case
tag, e.g. "novel_obstacle", "rain", "occlusion"); otherwise all OOD images are
grouped under "ood".

Usage:
    python tools/test_ood.py --ood-model work_dirs/ood/frcnn_perclass.npz \
        --id-features work_dirs/features/frcnn_val.npz \
        --ood-features work_dirs/features/frcnn_roadanomaly.npz \
        --agent-fpr 0.05 --driver-fpr 0.01 \
        --out work_dirs/results/frcnn_perclass
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ood_module import MahalanobisOOD, DualPathwayWarner, metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ood-model", required=True)
    p.add_argument("--id-features", required=True)
    p.add_argument("--ood-features", required=True)
    p.add_argument("--agent-fpr", type=float, default=0.05)
    p.add_argument("--driver-fpr", type=float, default=0.01)
    p.add_argument("--out", required=True, help="Output directory")
    args = p.parse_args()

    model = MahalanobisOOD.load(args.ood_model)
    id_data = np.load(args.id_features, allow_pickle=True)
    ood_data = np.load(args.ood_features, allow_pickle=True)

    id_scores = model.score(id_data["features"])
    ood_scores = model.score(ood_data["features"])

    # ---- aggregate metrics
    report = {"model": args.ood_model, "mode": model.mode}
    report.update(metrics.aggregate_metrics(id_scores, ood_scores))

    # ---- dual-pathway calibration + per-pathway warning quality
    warner = DualPathwayWarner.calibrate(id_scores, args.agent_fpr, args.driver_fpr)
    thresholds = {"agent": warner.agent_threshold, "driver": warner.driver_threshold}
    report["thresholds"] = thresholds
    for pathway, thr in thresholds.items():
        report[pathway] = metrics.fpr_fnr_at_threshold(id_scores, ood_scores, thr)

    # ---- per-category breakdown (our fine-grained contribution)
    if "categories" in ood_data:
        cats = np.asarray(ood_data["categories"])
        by_cat = {c: ood_scores[cats == c] for c in np.unique(cats)}
    else:
        by_cat = {"ood": ood_scores}
    rows = metrics.per_category_report(id_scores, by_cat, thresholds)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "per_category.csv"), index=False)
    np.savez(os.path.join(args.out, "scores.npz"),
             id_scores=id_scores, ood_scores=ood_scores)
    warner.save(os.path.join(args.out, "warner.npz"))

    print(json.dumps(report, indent=2))
    print(f"\nPer-category warning quality saved to {args.out}/per_category.csv")


if __name__ == "__main__":
    main()