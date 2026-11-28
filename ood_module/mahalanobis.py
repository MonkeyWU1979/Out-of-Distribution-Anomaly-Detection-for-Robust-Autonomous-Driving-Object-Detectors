"""Mahalanobis-distance OOD scorer.

Reference: Lee et al., "A Simple Unified Framework for Detecting
Out-of-Distribution Samples and Adversarial Attacks", NeurIPS 2018.

Adaptation for object detection (image-level scoring):
- Features are global-average-pooled neck feature vectors (one per image).
- "per_class" mode fits one Gaussian per object category. An image is
  assigned to every category present in its ground-truth annotations
  (multi-label fitting). The anomaly score is the MINIMUM Mahalanobis
  distance over all class Gaussians: an image is normal if it is close
  to ANY known class distribution.

Covariance refinements (our ablation axis):
- mode="global":    one shared full covariance (original baseline)
- mode="per_class": class-conditional full covariance (proposed refinement)
- mode="diagonal":  per-class diagonal covariance (high-dim refinement)
All modes apply Ledoit-Wolf-style shrinkage toward a scaled identity for
numerical stability in high-dimensional feature spaces.
"""

from __future__ import annotations

import numpy as np


class MahalanobisOOD:
    """Fit class-conditional Gaussians on ID features; score by distance."""

    MODES = ("global", "per_class", "diagonal")

    def __init__(self, mode: str = "per_class", shrinkage: float = 1e-2):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError("shrinkage must be in [0, 1]")
        self.mode = mode
        self.shrinkage = shrinkage
        self.means_: dict[int, np.ndarray] = {}
        self.precisions_: dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------ fit
    def fit(self, features: np.ndarray, labels: list[list[int]] | None = None):
        """Fit Gaussian(s) on in-distribution training features.

        Args:
            features: (N, D) array of pooled feature vectors.
            labels:   per-image list of GT category ids (multi-label).
                      Required for per_class / diagonal modes; ignored
                      for global mode. Images with an empty label list
                      are skipped in class-conditional modes.
        """
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2:
            raise ValueError("features must be a 2-D (N, D) array")

        if self.mode == "global":
            self.means_ = {-1: features.mean(axis=0)}
            self.precisions_ = {-1: self._precision(features)}
            return self

        if labels is None:
            raise ValueError(f"labels are required for mode={self.mode!r}")
        if len(labels) != len(features):
            raise ValueError("labels and features must have the same length")

        groups: dict[int, list[int]] = {}
        for idx, img_labels in enumerate(labels):
            for cls in set(img_labels):
                groups.setdefault(int(cls), []).append(idx)

        self.means_, self.precisions_ = {}, {}
        for cls, idxs in groups.items():
            subset = features[idxs]
            if len(subset) < 2:
                # Not enough samples to estimate a covariance; skip class.
                continue
            self.means_[cls] = subset.mean(axis=0)
            self.precisions_[cls] = self._precision(subset)

        if not self.means_:
            raise RuntimeError("No class had >= 2 samples; cannot fit.")
        return self

    def _precision(self, x: np.ndarray) -> np.ndarray:
        """Shrinkage covariance -> precision matrix."""
        cov = np.cov(x, rowvar=False)
        cov = np.atleast_2d(cov)
        if self.mode == "diagonal":
            cov = np.diag(np.diag(cov))
        d = cov.shape[0]
        mu = np.trace(cov) / d
        cov = (1.0 - self.shrinkage) * cov + self.shrinkage * mu * np.eye(d)
        return np.linalg.pinv(cov)

    # ---------------------------------------------------------------- score
    def score(self, features: np.ndarray) -> np.ndarray:
        """Return anomaly scores (higher = more anomalous).

        Score = min over class Gaussians of the Mahalanobis distance.
        """
        if not self.means_:
            raise RuntimeError("Model is not fitted. Call fit() or load().")
        features = np.asarray(features, dtype=np.float64)
        if features.ndim == 1:
            features = features[None, :]
        scores = np.full(len(features), np.inf)
        for cls, mean in self.means_.items():
            delta = features - mean
            # d^2 = delta @ precision @ delta^T, computed per row.
            dist = np.einsum("ni,ij,nj->n", delta, self.precisions_[cls], delta)
            scores = np.minimum(scores, np.sqrt(np.maximum(dist, 0.0)))
        return scores

    # ------------------------------------------------------------ save/load
    def save(self, path: str):
        classes = np.array(sorted(self.means_.keys()), dtype=np.int64)
        means = np.stack([self.means_[c] for c in classes])
        precs = np.stack([self.precisions_[c] for c in classes])
        np.savez(path, classes=classes, means=means, precisions=precs,
                 mode=np.array(self.mode), shrinkage=np.array(self.shrinkage))

    @classmethod
    def load(cls, path: str) -> "MahalanobisOOD":
        data = np.load(path, allow_pickle=False)
        obj = cls(mode=str(data["mode"]), shrinkage=float(data["shrinkage"]))
        for c, m, p in zip(data["classes"], data["means"], data["precisions"]):
            obj.means_[int(c)] = m
            obj.precisions_[int(c)] = p
        return obj