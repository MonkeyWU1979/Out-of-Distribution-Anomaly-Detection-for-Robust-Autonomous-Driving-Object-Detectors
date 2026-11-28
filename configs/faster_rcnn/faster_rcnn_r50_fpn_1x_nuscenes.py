"""Control group: Faster R-CNN + ResNet-50-FPN (pure CNN) on nuScenes.

Inherits the official MMDetection 1x COCO recipe, then overrides the
dataset with the shared nuScenes base so settings match Deformable DETR.

Run:
    python tools/train_detector.py --config configs/faster_rcnn/faster_rcnn_r50_fpn_1x_nuscenes.py
"""

_base_ = [
    "mmdet::faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py",
    "../_base_/nuscenes_dataset.py",
]

model = dict(
    data_preprocessor=dict(pad_size_divisor=32),
    roi_head=dict(bbox_head=dict(num_classes=10)),
)

# Scale LR automatically for our batch size (official recipe uses 16).
auto_scale_lr = dict(enable=True, base_batch_size=16)

# COCO pre-trained weights (official MMDetection model zoo) for fine-tuning.
load_from = ("https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/"
             "faster_rcnn_r50_fpn_1x_coco/"
             "faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth")

work_dir = "./work_dirs/frcnn_r50_nuscenes"