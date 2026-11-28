"""Experimental group: Deformable DETR (ResNet-50 + Transformer) on nuScenes.

Inherits the official MMDetection Deformable DETR (refine) recipe, then
overrides dataset AND schedule with the shared nuScenes base, so data,
image scale, augmentation and evaluation are identical to the Faster
R-CNN control group. The only variable is the architecture.

Run:
    python tools/train_detector.py --config configs/deformable_detr/deformable_detr_r50_1x_nuscenes.py
"""

_base_ = [
    "mmdet::detr/deformable-detr_refine_r50_16xb2-50e_coco.py",
    "../_base_/nuscenes_dataset.py",
]

model = dict(
    data_preprocessor=dict(pad_size_divisor=32),
    bbox_head=dict(num_classes=10),
    # Keep the shared single-scale pipeline; DETR-specific multi-scale
    # augmentation is intentionally disabled for a controlled comparison.
)

# Official recipe: AdamW, lr=2e-4 at total batch size 32.
auto_scale_lr = dict(enable=True, base_batch_size=32)

# Download the COCO pre-trained Deformable DETR (refine) checkpoint from the
# MMDetection model zoo (configs/detr/README.md) and set the local path here:
# load_from = "checkpoints/deformable_detr_refine_r50_coco.pth"
load_from = None

work_dir = "./work_dirs/deformable_detr_r50_nuscenes"