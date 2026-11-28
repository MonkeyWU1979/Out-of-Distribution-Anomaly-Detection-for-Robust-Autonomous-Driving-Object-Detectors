# Dataset Preparation

> Do NOT commit raw datasets. This folder holds scripts and a path template only.
> Copy `dataset_paths.yaml` to `dataset_paths.local.yaml` and edit it locally.

## 1. Main training dataset: nuScenes
1. Register and download from https://www.nuscenes.org/nuscenes#download
   - **Start with `v1.0-mini`** (~4 GB) to validate the full pipeline.
   - Move to `v1.0-trainval` for formal experiments.
2. Extract so that `<nuscenes_root>/samples/` exists.
3. Convert 3D annotations to COCO-format 2D labels:
   ```bash
   python data/convert_nuscenes.py --dataroot <nuscenes_root> --version v1.0-mini \
       --split mini_train --out <nuscenes_root>/annotations/train_coco.json
   python data/convert_nuscenes.py --dataroot <nuscenes_root> --version v1.0-mini \
       --split mini_val   --out <nuscenes_root>/annotations/val_coco.json
   ```
4. Point the configs at your local root:
   `--cfg-options data_root=<nuscenes_root>/` (or edit `configs/_base_/nuscenes_dataset.py` locally without committing).

Corner-case subsets: nuScenes annotations include `visibility_token`
(occlusion level) and scene descriptions (rain / night), which we use to
build the occlusion and adverse-weather test subsets.

## 2. OOD sanity check: COCO-OOD
1. Download COCO **val2017** images and `instances_val2017.json` from
   https://cocodataset.org/#download
2. Generate the held-out-category split:
   ```bash
   python data/prepare_coco_ood.py --ann <coco_root>/annotations/instances_val2017.json \
       --out-dir <coco_root>/ood_split
   ```

## 3. Driving OOD benchmark: RoadAnomaly (SegmentMeIfYouCan)
1. Download from https://segmentmeifyoucan.com/datasets (RoadAnomaly subset,
   60 real road images with novel obstacles: animals, rocks, debris...).
2. All RoadAnomaly images are treated as OOD; no training labels are needed.
   Extract features directly with `tools/extract_features.py` using a
   minimal COCO json listing the images (no annotations).

## 4. Controlled simulation: CARLA (optional, later stage)
- Simulator: https://carla.org/ (open source).
- Use the Python API to generate scenes with controlled rain/fog intensity,
  lighting and occlusion, then export images + labels for ablation studies.
- Generation scripts will be added under `data/` in the ablation phase.