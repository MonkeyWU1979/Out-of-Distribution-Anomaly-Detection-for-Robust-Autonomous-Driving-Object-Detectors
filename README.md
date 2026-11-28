# Out-of-Distribution Anomaly Detection for Robust Autonomous Driving Object Detectors
> COMP 4610SEF Data Science Project | Hong Kong Metropolitan University | Group 15

## Overview
This project builds an end-to-end OOD (Out-of-Distribution) warning pipeline that acts as an independent safety monitor on top of 2D object detectors for autonomous driving.

Instead of designing a new detector from scratch, we add a parallel anomaly scoring module based on **Mahalanobis distance on backbone features**. The module evaluates whether the current driving scene falls outside the training distribution and triggers hierarchical warnings before the detector fails.

## Core Research Questions
- **RQ1**: Can a Mahalanobis-based OOD module, integrated with a CNN-Transformer object detector, deliver reliable warnings for both autonomous agents and human drivers across three corner-case categories (adverse weather, occlusion, novel obstacles)?
- **RQ2**: How does targeted data augmentation affect detector generalization and OOD warning reliability across corner-case categories?

## Finalized Model & Data Selection

### Models (all run under one unified framework: MMDetection 3.x)
| Group | Model | Backbone | Purpose |
|-------|-------|----------|---------|
| Control | Faster R-CNN | ResNet-50-FPN (pure CNN) | Baseline OOD performance reference |
| Experimental | Deformable DETR | ResNet-50 + Transformer (hybrid) | Test whether global context improves OOD robustness |

Using one framework guarantees identical training pipelines, preprocessing and evaluation across both models — the only variable is the architecture itself.

### Datasets
| Type | Dataset | Usage |
|------|---------|-------|
| Main training | nuScenes (start with `v1.0-mini`) | Detector training + fitting the normal feature distribution |
| OOD sanity check | COCO-OOD (held-out categories) | Fast validation of the Mahalanobis implementation |
| OOD driving benchmark | RoadAnomaly (SegmentMeIfYouCan) | Novel-obstacle evaluation in real road scenes |
| Simulation (optional) | CARLA-generated scenes | Controlled single-variable ablation (weather intensity, occlusion ratio) |

> Raw datasets are NOT stored in this repository. See `data/README.md` for download and preparation instructions.

## Project Structure
```
.
├── README.md
├── requirements.txt
├── .gitignore
├── configs/                 # MMDetection configs (unified experimental settings)
│   ├── faster_rcnn/         # Control group
│   └── deformable_detr/     # Experimental group
├── data/                    # Download / conversion scripts + path template (no raw data)
├── ood_module/              # Core original contribution: OOD scoring + dual-pathway warning
├── tools/                   # Train / extract features / fit OOD / evaluate scripts
└── docs/                    # Environment setup, experiment plan, meeting notes
```

## Quick Start
```bash
# 1. Environment (see docs/environment.md for full steps)
conda create -n fyp-ood python=3.10 -y && conda activate fyp-ood
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -U openmim && mim install mmengine "mmcv>=2.0.0" "mmdet>=3.0.0"
pip install -r requirements.txt

# 2. Prepare data (see data/README.md), then edit data/dataset_paths.yaml

# 3. Train (or use pre-trained weights first) the control detector
python tools/train_detector.py --config configs/faster_rcnn/faster_rcnn_r50_fpn_1x_nuscenes.py

# 4. Extract features from in-distribution training images
python tools/extract_features.py --config configs/faster_rcnn/faster_rcnn_r50_fpn_1x_nuscenes.py \
    --ann-file <nuscenes_train_coco.json> --img-root <nuscenes_root> --out work_dirs/features/frcnn_train.npz

# 5. Fit the Mahalanobis OOD model
python tools/fit_ood.py --features work_dirs/features/frcnn_train.npz --mode per_class \
    --out work_dirs/ood/frcnn_perclass.npz

# 6. Evaluate OOD detection + dual-pathway warnings
python tools/test_ood.py --ood-model work_dirs/ood/frcnn_perclass.npz \
    --id-features work_dirs/features/frcnn_val.npz \
    --ood-features work_dirs/features/frcnn_roadanomaly.npz \
    --out work_dirs/results/frcnn_perclass
```
Repeat steps 3–6 with the Deformable DETR config for the experimental group.

## Pipeline
```
Input image ──► Detector backbone+neck (CNN or CNN+Transformer)
                 ├─► Detection head ──► boxes / classes (mAP evaluation)
                 └─► GAP feature vector ──► Mahalanobis OOD score
                                              └─► Dual-pathway warning
                                                   ├─ Agent pathway (low threshold, high recall)
                                                   └─ Driver pathway (high threshold, low FPR)
```

## Team
| Member | Responsibilities |
|--------|------------------|
| Li Chun Lung | Data preparation, detector training & baselines |
| Ng Mun Hin Manfred | OOD module, dual-pathway warning pipeline |
| Aaditya Rawat | Experiment evaluation, metrics & result analysis |

## Collaboration Rules
1. Never commit raw datasets, checkpoints, or `.npy/.npz` feature files (see `.gitignore`).
2. Keep training schedule / augmentation identical in both config files for fair comparison.
3. `data/dataset_paths.yaml` is a template — keep personal absolute paths out of commits.
4. Work on feature branches; merge to `main` via pull request.

## License
For academic use only (COMP 4610SEF coursework).