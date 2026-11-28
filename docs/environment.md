# Environment Setup Guide

Target hardware: NVIDIA GPU (tested target: RTX 4060 Laptop, 8 GB VRAM).
Windows + Anaconda is fully supported.

## Step 1 — Conda environment
```bash
conda create -n fyp-ood python=3.10 -y
conda activate fyp-ood
```

## Step 2 — PyTorch (match your CUDA driver; example CUDA 12.1)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Step 3 — MMDetection 3.x (use openmim, NOT plain pip for mmcv)
```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install "mmdet>=3.2.0"
python -c "import mmdet; print(mmdet.__version__)"
```

## Step 4 — Project dependencies
```bash
pip install -r requirements.txt
```

## Step 5 — Sanity check
```bash
# Official pre-trained inference demo (downloads weights automatically)
python -c "
from mmdet.apis import init_detector, inference_detector
m = init_detector('mmdet::faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py', None, device='cuda:0')
print('Detector ready:', type(m).__name__)
"
```
Then run `tools/extract_features.py --limit 8` on a small annotation file to
confirm feature extraction works end-to-end.

## Common issues
| Symptom | Fix |
|---------|-----|
| `mmcv` build errors / long compilation | Always `mim install mmcv`, never `pip install mmcv` from source |
| CUDA unavailable | Match the PyTorch CUDA build to `nvidia-smi` driver support |
| VRAM out-of-memory (8 GB laptop GPU) | `--cfg-options train_dataloader.batch_size=1`; LR auto-scales |
| nuScenes import errors | `pip install nuscenes-devkit pyquaternion` |
| Windows DataLoader worker crash | Set `num_workers=0` in the dataloader cfg-options |