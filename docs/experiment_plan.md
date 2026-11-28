# Experiment Plan

## Experimental matrix
| # | Detector | OOD covariance mode | Augmentation | Purpose |
|---|----------|--------------------|--------------|---------|
| E1 | Faster R-CNN (R50-FPN) | global | baseline | Pure-CNN + original Mahalanobis baseline |
| E2 | Faster R-CNN (R50-FPN) | per_class | baseline | Covariance refinement effect |
| E3 | Deformable DETR (R50) | global | baseline | Hybrid backbone + original Mahalanobis |
| E4 | Deformable DETR (R50) | per_class | baseline | **Core comparison vs E2 (RQ1)** |
| E5 | both | best mode from E1–E4 | + weather/occlusion aug | Augmentation ablation (RQ2) |
| E6 (optional) | best config | + diagonal mode | best aug | High-dim refinement ablation |

## Test sets per experiment
- ID validation: nuScenes val split (normal scenes).
- OOD general: COCO-OOD held-out categories (sanity check).
- OOD driving: RoadAnomaly (novel obstacles).
- Corner-case subsets: nuScenes rain/night scenes; low-visibility (occlusion) subset.
- Optional: CARLA controlled weather-intensity series.

## Metrics
- Detection: COCO mAP@0.5 and mAP@0.5:0.95.
- OOD aggregate: AU-ROC, AU-PR.
- Warning quality (core contribution): FPR / FNR per pathway
  (agent target FPR 5%, driver target FPR 1%) and per corner-case category.
- Efficiency: per-frame inference time (FPS) of detector + OOD overhead.

## Milestones
| Phase | Deliverable | Target |
|-------|-------------|--------|
| 1 | Environment + Faster R-CNN inference + feature extraction on nuScenes mini | Week 1–2 |
| 2 | Mahalanobis module validated on COCO-OOD (E1) | Week 3 |
| 3 | RoadAnomaly evaluation + dual-pathway calibration (E2) | Week 4–5 |
| 4 | Deformable DETR integration + core comparison (E3/E4) | Week 6–8 |
| 5 | Augmentation ablation (E5) + result analysis | Week 9–11 |
| 6 | Final report + demo + defense slides | Week 12+ |