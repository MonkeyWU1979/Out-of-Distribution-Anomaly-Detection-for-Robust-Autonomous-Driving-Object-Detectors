"""Build a COCO-OOD sanity-check benchmark from COCO val2017.

Split strategy (held-out-category protocol):
- ID categories:  person, bicycle, car, motorcycle, bus, truck
  (vehicle / VRU categories closest to the nuScenes training classes).
- ID set:  images whose annotations contain ONLY ID categories.
- OOD set: images containing at least one non-ID (held-out) category,
  e.g. animals, furniture, food -- semantically unseen objects.

Outputs two COCO-format json files usable by tools/extract_features.py.

Requires: COCO val2017 images + instances_val2017.json
(https://cocodataset.org/#download)

Usage:
    python data/prepare_coco_ood.py --ann /data/coco/annotations/instances_val2017.json \
        --out-dir /data/coco/ood_split
"""

import argparse
import json
import os
from collections import defaultdict

ID_CATEGORIES = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ann", required=True, help="instances_val2017.json path")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    with open(args.ann) as f:
        coco = json.load(f)

    id_cat_ids = {c["id"] for c in coco["categories"] if c["name"] in ID_CATEGORIES}
    cats_per_image = defaultdict(set)
    anns_per_image = defaultdict(list)
    for ann in coco["annotations"]:
        cats_per_image[ann["image_id"]].add(ann["category_id"])
        anns_per_image[ann["image_id"]].append(ann)

    id_images, ood_images = [], []
    for img in coco["images"]:
        cats = cats_per_image.get(img["id"], set())
        if not cats:
            continue  # skip unannotated images
        if cats <= id_cat_ids:
            id_images.append(img)
        else:
            ood_images.append(img)

    os.makedirs(args.out_dir, exist_ok=True)

    id_json = dict(
        info=coco.get("info", {}), images=id_images,
        annotations=[a for img in id_images for a in anns_per_image[img["id"]]],
        categories=[c for c in coco["categories"] if c["id"] in id_cat_ids],
    )
    ood_json = dict(
        info=coco.get("info", {}), images=ood_images,
        annotations=[],  # labels not needed for OOD scoring
        categories=coco["categories"],
    )

    id_path = os.path.join(args.out_dir, "coco_ood_id.json")
    ood_path = os.path.join(args.out_dir, "coco_ood_ood.json")
    with open(id_path, "w") as f:
        json.dump(id_json, f)
    with open(ood_path, "w") as f:
        json.dump(ood_json, f)
    print(f"ID images:  {len(id_images)} -> {id_path}")
    print(f"OOD images: {len(ood_images)} -> {ood_path}")


if __name__ == "__main__":
    main()