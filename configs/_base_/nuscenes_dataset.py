"""Shared nuScenes (COCO-format) dataset & schedule settings.

IMPORTANT: both detector configs inherit this file, so data pipeline,
splits, image scale and evaluation are IDENTICAL across the control
(Faster R-CNN) and experimental (Deformable DETR) groups. The only
variable between the two experiments is the model architecture.

Expected local layout (edit data_root per machine via --cfg-options):
    data/nuscenes/
    ├── samples/            # camera images (nuScenes original layout)
    └── annotations/
        ├── train_coco.json # produced by data/convert_nuscenes.py
        └── val_coco.json
"""

dataset_type = "CocoDataset"
data_root = "data/nuscenes/"

# 10 official nuScenes detection categories.
classes = (
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
)
metainfo = dict(classes=classes)

# Identical train/test pipelines for BOTH models (fair comparison).
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", scale=(1333, 800), keep_ratio=True),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PackDetInputs"),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(1333, 800), keep_ratio=True),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="PackDetInputs",
         meta_keys=("img_id", "img_path", "ori_shape", "img_shape",
                    "scale_factor", "batch_input_shape")),
]

train_dataloader = dict(
    batch_size=2, num_workers=2, persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type, data_root=data_root, metainfo=metainfo,
        ann_file="annotations/train_coco.json", data_prefix=dict(img=""),
        filter_cfg=dict(filter_empty_gt=True, min_size=8),
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=1, num_workers=2, persistent_workers=True, drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type, data_root=data_root, metainfo=metainfo,
        ann_file="annotations/val_coco.json", data_prefix=dict(img=""),
        test_mode=True, pipeline=test_pipeline,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + "annotations/val_coco.json",
    metric="bbox", format_only=False,
)
test_evaluator = val_evaluator

# 1x schedule (12 epochs) for both groups.
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=12, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

default_hooks = dict(
    checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=3,
                    save_best="coco/bbox_mAP", rule="greater"),
    logger=dict(type="LoggerHook", interval=50),
)