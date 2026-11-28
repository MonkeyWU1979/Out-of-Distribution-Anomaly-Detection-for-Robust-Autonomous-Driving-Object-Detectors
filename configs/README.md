# Model Configurations

All detectors run under **MMDetection 3.x** so training pipeline, data
preprocessing and evaluation are identical across models — the only
experimental variable is the architecture.

## Files
| File | Role |
|------|------|
| `_base_/nuscenes_dataset.py` | Shared dataset, pipeline, schedule, evaluator (single source of truth) |
| `faster_rcnn/faster_rcnn_r50_fpn_1x_nuscenes.py` | Control group: pure CNN baseline |
| `deformable_detr/deformable_detr_r50_1x_nuscenes.py` | Experimental group: CNN + Transformer hybrid |

## Rules for fair comparison
1. Change data pipeline / schedule / image scale ONLY in `_base_/nuscenes_dataset.py`,
   so both models always inherit the same settings.
2. If you tune a hyperparameter for one model (e.g. augmentation ablation),
   apply the identical change to the other model's run.
3. Record every deviation in `docs/meeting_notes/` or the experiment log.

## Pre-trained weights
- Faster R-CNN: downloaded automatically via `load_from` in the config.
- Deformable DETR: download the COCO checkpoint from the MMDetection model
  zoo (`configs/detr/README.md` in the mmdet repo), then set `load_from`
  to the local path.