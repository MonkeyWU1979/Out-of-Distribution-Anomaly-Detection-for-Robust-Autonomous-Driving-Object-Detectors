# Execution Scripts

Run order for one experimental group (repeat with the other config):

| Step | Script | Purpose |
|------|--------|---------|
| 1 | `train_detector.py` | Fine-tune the detector on nuScenes (or skip and use pre-trained weights first) |
| 2 | `test_detector.py` | Report COCO mAP on the validation split |
| 3 | `extract_features.py` | Extract pooled neck features + GT class labels for train / val / OOD sets |
| 4 | `fit_ood.py` | Fit Mahalanobis Gaussians on ID training features (`global` / `per_class` / `diagonal`) |
| 5 | `test_ood.py` | Score ID vs OOD, calibrate dual-pathway thresholds, output metrics + per-category CSV |

## Example (control group)
```bash
CFG=configs/faster_rcnn/faster_rcnn_r50_fpn_1x_nuscenes.py

python tools/train_detector.py --config $CFG
python tools/test_detector.py  --config $CFG --checkpoint work_dirs/frcnn_r50_nuscenes/best_coco_bbox_mAP_epoch_*.pth

python tools/extract_features.py --config $CFG --checkpoint <ckpt> \
    --ann-file <nuscenes>/annotations/train_coco.json --img-root <nuscenes> \
    --out work_dirs/features/frcnn_train.npz
python tools/extract_features.py --config $CFG --checkpoint <ckpt> \
    --ann-file <nuscenes>/annotations/val_coco.json --img-root <nuscenes> \
    --out work_dirs/features/frcnn_val.npz

python tools/fit_ood.py --features work_dirs/features/frcnn_train.npz \
    --mode per_class --out work_dirs/ood/frcnn_perclass.npz

python tools/test_ood.py --ood-model work_dirs/ood/frcnn_perclass.npz \
    --id-features work_dirs/features/frcnn_val.npz \
    --ood-features work_dirs/features/frcnn_roadanomaly.npz \
    --out work_dirs/results/frcnn_perclass
```

Outputs of `test_ood.py`: `report.json` (AU-ROC / AU-PR / per-pathway FPR-FNR),
`per_category.csv` (corner-case breakdown), `scores.npz`, `warner.npz`.