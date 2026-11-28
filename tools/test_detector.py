"""Evaluate detector performance (COCO mAP) with MMDetection.

Usage:
    python tools/test_detector.py \
        --config configs/faster_rcnn/faster_rcnn_r50_fpn_1x_nuscenes.py \
        --checkpoint work_dirs/frcnn/best.pth
"""

import argparse


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--work-dir", default=None)
    args = p.parse_args()

    from mmengine.config import Config
    from mmengine.runner import Runner

    cfg = Config.fromfile(args.config)
    cfg.load_from = args.checkpoint
    if args.work_dir:
        cfg.work_dir = args.work_dir

    runner = Runner.from_cfg(cfg)
    runner.test()


if __name__ == "__main__":
    main()