"""OOD evaluation metrics.

Beyond aggregate AU-ROC / AU-PR, we report per-pathway and per-category
FPR / FNR -- the practical warning-quality metrics that address our
research gap ("anomaly scores are not the same as actionable warnings").

Convention: OOD (anomalous) samples are the POSITIVE class.
- FPR: fraction of normal (ID) scenes wrongly flagged as anomalous.
- FNR: fraction of anomalous (OOD) scenes missed by the warning system.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def aggregate_metrics(id_scores: np.ndarray, ood_scores: np.ndarray) -> dict:
    """Threshold-free aggregate metrics."""
    y = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    s = np.concatenate([id_scores, ood_scores])
    return {
        "auroc": float(roc_auc_score(y, s)),
        "aupr": float(average_precision_score(y, s)),
        "n_id": int(len(id_scores)),
        "n_ood": int(len(ood_scores)),
    }


def fpr_fnr_at_threshold(id_scores: np.ndarray, ood_scores: np.ndarray,
                         threshold: float) -> dict:
    """FPR / FNR at a fixed decision threshold (score >= threshold -> alarm)."""
    fpr = float(np.mean(np.asarray(id_scores) >= threshold))
    fnr = float(np.mean(np.asarray(ood_scores) < threshold))
    return {"threshold": float(threshold), "fpr": fpr, "fnr": fnr}


def threshold_at_target_fpr(id_scores: np.ndarray, target_fpr: float) -> float:
    """Smallest threshold whose FPR on ID data is <= target_fpr."""
    return float(np.quantile(np.asarray(id_scores), 1.0 - target_fpr))


def per_category_report(id_scores: np.ndarray,
                        ood_scores_by_category: dict[str, np.ndarray],
                        thresholds: dict[str, float]) -> list[dict]:
    """Fine-grained warning quality: one row per (category, pathway).

    Args:
        id_scores: normal-scene scores (shared across categories).
        ood_scores_by_category: e.g. {"novel_obstacle": scores, "rain": scores}.
        thresholds: e.g. {"agent": t_agent, "driver": t_driver}.
    """
    rows = []
    for category, scores in ood_scores_by_category.items():
        for pathway, thr in thresholds.items():
            r = fpr_fnr_at_threshold(id_scores, scores, thr)
            rows.append({"category": category, "pathway": pathway, **r})
    return rows