"""Train a detector with MMDetection (thin, reproducible wrapper).

Usage:
    python tools/train_detector.py \
        --config configs/faster_rcnn/faster_rcnn_r50_fpn_1x_nuscenes.py \
        [--work-dir work_dirs/frcnn] [--resume]
"""

import argparse

from mmengine.config import DictAction


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--work-dir", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--cfg-options", nargs="+", action=DictAction, default=None,
                   help="Override config entries, e.g. train_dataloader.batch_size=2")
    args = p.parse_args()

    from mmengine.config import Config
    from mmengine.runner import Runner

    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    if args.work_dir:
        cfg.work_dir = args.work_dir
    cfg.resume = args.resume

    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == "__main__":
    main()