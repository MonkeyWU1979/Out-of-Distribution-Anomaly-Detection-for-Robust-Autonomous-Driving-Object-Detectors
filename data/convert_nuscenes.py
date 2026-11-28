"""Convert nuScenes 3D annotations to COCO-format 2D detection labels.

For every camera frame in the requested split, project each 3D box into
the image plane, clip to image bounds, and map the category to one of the
10 official nuScenes detection classes. Boxes behind the camera or with a
degenerate projected area are skipped.

Requires: pip install nuscenes-devkit

Usage (start with the mini split to validate the pipeline):
    python data/convert_nuscenes.py --dataroot /data/nuscenes \
        --version v1.0-mini --split mini_train \
        --out /data/nuscenes/annotations/train_coco.json
    python data/convert_nuscenes.py --dataroot /data/nuscenes \
        --version v1.0-mini --split mini_val \
        --out /data/nuscenes/annotations/val_coco.json
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm

CAMERAS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
           "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

CLASSES = ["car", "truck", "construction_vehicle", "bus", "trailer",
           "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataroot", required=True, help="nuScenes root folder")
    p.add_argument("--version", default="v1.0-mini",
                   choices=["v1.0-mini", "v1.0-trainval"])
    p.add_argument("--split", default="mini_train",
                   help="Scene split name, e.g. mini_train / mini_val / train / val")
    p.add_argument("--out", required=True, help="Output COCO json path")
    p.add_argument("--min-area", type=float, default=16.0,
                   help="Skip projected boxes smaller than this area (px^2)")
    return p.parse_args()


def main():
    args = parse_args()
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.splits import create_splits_scenes
    from nuscenes.utils.data_classes import Box
    from nuscenes.utils.geometry_utils import view_points
    from nuscenes.eval.detection.utils import category_to_detection_name

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
    split_scenes = set(create_splits_scenes()[args.split])
    cat2id = {c: i + 1 for i, c in enumerate(CLASSES)}  # COCO ids start at 1

    images, annotations = [], []
    img_id, ann_id = 0, 0

    samples = [s for s in nusc.sample
               if nusc.get("scene", s["scene_token"])["name"] in split_scenes]

    for sample in tqdm(samples, desc=f"Converting {args.split}"):
        for cam in CAMERAS:
            sd = nusc.get("sample_data", sample["data"][cam])
            if not sd["is_key_frame"]:
                continue
            width, height = sd["width"], sd["height"]
            cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
            ego = nusc.get("ego_pose", sd["ego_pose_token"])
            intrinsic = np.array(cs["camera_intrinsic"])

            img_id += 1
            images.append(dict(id=img_id, file_name=sd["filename"],
                               width=width, height=height))

            for ann_token in sample["anns"]:
                ann = nusc.get("sample_annotation", ann_token)
                det_name = category_to_detection_name(ann["category_name"])
                if det_name is None or det_name not in cat2id:
                    continue

                box = Box(ann["translation"], ann["size"],
                          Quaternion(ann["rotation"]))
                # global -> ego -> camera frame
                box.translate(-np.array(ego["translation"]))
                box.rotate(Quaternion(ego["rotation"]).inverse)
                box.translate(-np.array(cs["translation"]))
                box.rotate(Quaternion(cs["rotation"]).inverse)

                corners_cam = box.corners()
                if np.any(corners_cam[2, :] < 0.5):  # behind / too close
                    continue
                pts = view_points(corners_cam, intrinsic, normalize=True)[:2]
                x1 = max(0.0, float(pts[0].min()))
                y1 = max(0.0, float(pts[1].min()))
                x2 = min(width - 1.0, float(pts[0].max()))
                y2 = min(height - 1.0, float(pts[1].max()))
                w, h = x2 - x1, y2 - y1
                if w <= 1 or h <= 1 or w * h < args.min_area:
                    continue

                ann_id += 1
                annotations.append(dict(
                    id=ann_id, image_id=img_id, category_id=cat2id[det_name],
                    bbox=[round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
                    area=round(w * h, 2), iscrowd=0,
                    nuScenes_token=ann_token,
                    visibility_token=ann["visibility_token"],
                ))

    coco = dict(
        info=dict(description=f"nuScenes {args.version} {args.split} -> COCO 2D",
                  date_created=str(datetime.now())),
        images=images, annotations=annotations,
        categories=[dict(id=i, name=c) for c, i in cat2id.items()],
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(coco, f)
    print(f"Wrote {len(images)} images / {len(annotations)} boxes -> {args.out}")


if __name__ == "__main__":
    main()