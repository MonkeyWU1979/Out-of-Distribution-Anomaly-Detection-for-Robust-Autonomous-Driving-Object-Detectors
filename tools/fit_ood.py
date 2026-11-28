"""Fit the Mahalanobis OOD model on in-distribution training features.

Usage:
    python tools/fit_ood.py --features work_dirs/features/frcnn_train.npz \
        --mode per_class --shrinkage 0.01 --out work_dirs/ood/frcnn_perclass.npz
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ood_module import MahalanobisOOD


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", required=True, help="ID training features .npz")
    p.add_argument("--mode", default="per_class",
                   choices=["global", "per_class", "diagonal"])
    p.add_argument("--shrinkage", type=float, default=1e-2)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    data = np.load(args.features, allow_pickle=True)
    features = data["features"]
    labels = list(data["labels"]) if "labels" in data else None

    model = MahalanobisOOD(mode=args.mode, shrinkage=args.shrinkage)
    model.fit(features, labels)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    model.save(args.out)
    print(f"Fitted {len(model.means_)} Gaussian(s), mode={args.mode}, "
          f"dim={features.shape[1]} -> {args.out}")


if __name__ == "__main__":
    main()