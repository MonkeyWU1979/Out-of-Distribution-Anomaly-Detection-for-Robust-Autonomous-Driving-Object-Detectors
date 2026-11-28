"""Dual-pathway warning mechanism (core novel contribution).

One anomaly score, two recipients with asymmetric risk tolerance:

1. Autonomous-agent pathway  -- safety-first.
   Lower threshold (higher sensitivity). Missing a hazard is far more
   costly than a false alarm; false alarms only trigger conservative
   actions (deceleration, larger following distance, takeover request).

2. Human-driver pathway      -- usability-first.
   Higher threshold (higher specificity). Frequent false alarms cause
   alert fatigue and destroy trust in the system, so warnings are only
   issued at high anomaly confidence.

Thresholds are calibrated from the ID validation score distribution:
each threshold is the (1 - target_FPR) quantile of normal-scene scores,
so the false-positive rate on normal scenes is controlled by design.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class WarningLevel(str, Enum):
    NORMAL = "normal"                # both pathways silent
    AGENT_ALERT = "agent_alert"      # agent acts conservatively; driver silent
    FULL_ALERT = "full_alert"        # agent takeover request + driver warning


@dataclass
class DualPathwayWarner:
    agent_threshold: float
    driver_threshold: float

    def __post_init__(self):
        if self.agent_threshold > self.driver_threshold:
            raise ValueError(
                "agent_threshold must be <= driver_threshold "
                "(agent pathway is more sensitive by design)")

    # ----------------------------------------------------------- calibration
    @classmethod
    def calibrate(cls, id_scores: np.ndarray,
                  agent_target_fpr: float = 0.05,
                  driver_target_fpr: float = 0.01) -> "DualPathwayWarner":
        """Set thresholds from ID validation scores.

        Args:
            id_scores: anomaly scores on normal (in-distribution) scenes.
            agent_target_fpr:  acceptable FPR for the agent pathway (e.g. 5%).
            driver_target_fpr: acceptable FPR for the driver pathway (e.g. 1%).
        """
        if not 0 < driver_target_fpr <= agent_target_fpr < 1:
            raise ValueError("require 0 < driver_fpr <= agent_fpr < 1")
        id_scores = np.asarray(id_scores, dtype=np.float64)
        return cls(
            agent_threshold=float(np.quantile(id_scores, 1 - agent_target_fpr)),
            driver_threshold=float(np.quantile(id_scores, 1 - driver_target_fpr)),
        )

    # -------------------------------------------------------------- classify
    def classify(self, score: float) -> dict:
        """Map one anomaly score to per-pathway decisions."""
        agent_triggered = bool(score >= self.agent_threshold)
        driver_triggered = bool(score >= self.driver_threshold)
        if driver_triggered:
            level = WarningLevel.FULL_ALERT
        elif agent_triggered:
            level = WarningLevel.AGENT_ALERT
        else:
            level = WarningLevel.NORMAL
        return {
            "score": float(score),
            "level": level.value,
            "agent": {
                "triggered": agent_triggered,
                "action": ("Low scene confidence: decelerate / increase gap / "
                           "request human takeover") if agent_triggered else "none",
            },
            "driver": {
                "triggered": driver_triggered,
                "action": ("Unusual scene detected - please pay attention"
                           if driver_triggered else "none"),
            },
        }

    def classify_batch(self, scores: np.ndarray) -> list[dict]:
        return [self.classify(float(s)) for s in np.asarray(scores)]

    # ------------------------------------------------------------ save/load
    def save(self, path: str):
        np.savez(path, agent_threshold=self.agent_threshold,
                 driver_threshold=self.driver_threshold)

    @classmethod
    def load(cls, path: str) -> "DualPathwayWarner":
        d = np.load(path)
        return cls(float(d["agent_threshold"]), float(d["driver_threshold"]))