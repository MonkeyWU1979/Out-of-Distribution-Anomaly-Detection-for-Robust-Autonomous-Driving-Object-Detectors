# OOD Module — Core Original Contribution

Detector-agnostic Mahalanobis OOD scoring + dual-pathway warning + metrics.
Consumes pooled feature vectors from any MMDetection backbone
(produced by `tools/extract_features.py`).

## Components
| File | Responsibility |
|------|----------------|
| `mahalanobis.py` | Fit class-conditional Gaussians on ID features; score = min Mahalanobis distance. Modes: `global` / `per_class` / `diagonal` (our refinement ablation), all with shrinkage for high-dim stability. |
| `dual_path_warning.py` | One score, two pathways: agent (low threshold, safety-first) vs driver (high threshold, avoids alert fatigue). Thresholds calibrated as ID-score quantiles at target FPR. |
| `metrics.py` | AU-ROC / AU-PR plus per-pathway and per-corner-case FPR / FNR — the practical warning-quality metrics addressing our research gap. |

## Design notes
- **Image-level adaptation**: original Lee et al. (2018) is class-conditional
  for classification. Here an image joins every class present in its GT
  annotations (multi-label fitting); the score is the minimum distance over
  class Gaussians — normal if close to ANY known class distribution.
- **Fair feature source**: both detectors use the last-level neck feature map
  with global average pooling (see `tools/extract_features.py`).
- **Conventions**: OOD = positive class. FPR = normal scenes wrongly flagged;
  FNR = anomalous scenes missed.