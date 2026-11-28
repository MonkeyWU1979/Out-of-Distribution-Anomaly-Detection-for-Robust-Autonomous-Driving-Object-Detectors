"""Extract pooled backbone/neck features from a detector for OOD scoring.

Feature source (identical rule for BOTH models, keeping the comparison fair):
    model.extract_feat() -> last-level feature map -> global average pooling
For Faster R-CNN this is the last FPN level; for Deformable DETR this is the
last level of the encoder input projection (ChannelMapper neck).

Outputs an .npz with:
    features: (N, D) float32 pooled feature vectors
    labels:   object-array of per-image GT category-id lists (multi-label)
    img_ids:  (N,) dataset image ids

Usage:
    python tools/extract_features.py \
        --config configs/faster_rcnn/faster_rcnn_r50_fpn_1x_nuscenes.py \
        --checkpoint work_dirs/frcnn/best.pth \
        --ann-file /data/nuscenes/annotations/train_coco.json \
        --img-root /data/nuscenes \
        --out work_dirs/features/frcnn_train.npz
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None,
                   help="Detector weights; if omitted, uses config load_from")
    p.add_argument("--ann-file", required=True, help="COCO-format annotation json")
    p.add_argument("--img-root", required=True, help="Image root directory")
    p.add_argument("--out", required=True, help="Output .npz path")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int, default=None, help="Debug: first N images")
    return p.parse_args()


def main():
    args = parse_args()
    from mmengine.config import Config
    from mmengine.dataset import pseudo_collate
    from mmdet.apis import init_detector
    from mmdet.registry import DATASETS

    cfg = Config.fromfile(args.config)

    # Build the test dataset with user-specified annotations / image root.
    ds_cfg = cfg.test_dataloader.dataset
    ds_cfg.ann_file = args.ann_file
    ds_cfg.data_root = ""
    ds_cfg.data_prefix = dict(img=args.img_root)
    ds_cfg.pipeline = [s for s in ds_cfg.pipeline
                       if s["type"] != "LoadAnnotations"]
    dataset = DATASETS.build(ds_cfg)
    if args.limit:
        dataset = torch.utils.data.Subset(dataset, range(min(args.limit, len(dataset))))
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=2, collate_fn=pseudo_collate)

    model = init_detector(cfg, args.checkpoint, device=args.device)
    model.eval()

    feats, labels, img_ids = [], [], []
    base_ds = dataset.dataset if isinstance(dataset, torch.utils.data.Subset) else dataset
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Extracting features")):
            data = model.data_preprocessor(batch, training=False)
            # Last-level neck feature map -> global average pooling.
            feat_maps = model.extract_feat(data["inputs"])
            vec = feat_maps[-1].mean(dim=(2, 3)).flatten(1)  # (B, C)
            feats.append(vec.cpu().numpy()[0])

            idx = dataset.indices[i] if isinstance(dataset, torch.utils.data.Subset) else i
            info = base_ds.get_data_info(idx)
            gt_classes = sorted({int(inst["bbox_label"])
                                 for inst in info.get("instances", [])
                                 if not inst.get("ignore_flag", 0)})
            labels.append(gt_classes)
            img_ids.append(info.get("img_id", i))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    label_arr = np.empty(len(labels), dtype=object)
    for i, l in enumerate(labels):
        label_arr[i] = l
    np.savez(args.out,
             features=np.stack(feats).astype(np.float32),
             labels=label_arr,
             img_ids=np.asarray(img_ids))
    print(f"Saved {len(feats)} feature vectors (dim={feats[0].shape[0]}) -> {args.out}")


if __name__ == "__main__":
    main()