"""OOD module: Mahalanobis anomaly scoring + dual-pathway warning + metrics.

This package is the core original contribution of the project. It is
detector-agnostic: it consumes feature vectors extracted from any
MMDetection backbone (see tools/extract_features.py).
"""

from .mahalanobis import MahalanobisOOD
from .dual_path_warning import DualPathwayWarner, WarningLevel
from . import metrics

__all__ = ["MahalanobisOOD", "DualPathwayWarner", "WarningLevel", "metrics"]