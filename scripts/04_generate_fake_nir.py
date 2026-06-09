#!/usr/bin/env python3
"""
Step 4 – Generate fake NIR iris codes from VIS probe images using trained GAN.

VIS probes (subjects 168-209, instances 11-15) are fed through net_1 (G_VIS→NIR)
from the trained GAN checkpoint.  Output preserves the ImageFolder class structure
(one folder per subject+eye) so 05_evaluate.py can resolve genuine/impostor pairs
purely from the folder name.

Usage
-----
  cd ~/Cross-Spectral-Iris

  # grey_clahe model
  python scripts/04_generate_fake_nir.py \\
      --checkpoint ./checkpoint/model_normalized_grey_clahe_train167.pt \\
      --vis_probe  ~/data/processed/normalized_codes_split/grey_clahe/eval/VIS_probe \\
      --out_dir    ~/data/processed/fake_nir/grey_clahe_train167

  # grey_noclahe model
  python scripts/04_generate_fake_nir.py \\
      --checkpoint ./checkpoint/model_normalized_grey_noclahe_train167.pt \\
      --vis_probe  ~/data/processed/normalized_codes_split/grey_noclahe/eval/VIS_probe \\
      --out_dir    ~/data/processed/fake_nir/grey_noclahe_train167
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import datasets, transforms

# model.py lives in the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import UNet

inv_normalize = transforms.Normalize(
    mean=[-0.5 / 0.5] * 3,
    std=[1 / 0.5] * 3,
)


def tensor_to_bgr_u8(t: torch.Tensor) -> np.ndarray:
    """(3, H, W) float tensor in [0,1] → (H, W, 3) uint8 BGR for cv2.imwrite."""
    arr = t.clamp(0, 1).permute(1, 2, 0).numpy()   # HWC, RGB order
    arr = (arr * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser(description="GAN fake-NIR generation for evaluation")
    ap.add_argument('--checkpoint', required=True, help='Path to .pt GAN checkpoint')
    ap.add_argument('--vis_probe',  required=True, help='VIS probe dir  (ImageFolder layout)')
    ap.add_argument('--out_dir',    required=True, help='Output dir for fake NIR (ImageFolder layout)')
    ap.add_argument('--modality',   default='normalized', choices=['normalized', 'cropped'])
    ap.add_argument('--batch_size', default=64, type=int, help='Forward-pass batch size')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Load G_VIS→NIR (net_1) ────────────────────────────────────────────
    feat_dim = 128 if args.modality == 'normalized' else 256
    net_1 = UNet(feat_dim=feat_dim).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    net_1.load_state_dict(state['net_1'])
    net_1.eval()
    epoch = state.get('epoch', '?')
    print(f'Checkpoint: {args.checkpoint}  (epoch {epoch})')

    # ── Load VIS probe dataset ────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    dataset = datasets.ImageFolder(str(Path(args.vis_probe).expanduser()), transform=transform)
    loader  = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True)
    print(f'VIS probes: {len(dataset)} images across {len(dataset.classes)} classes')

    out_root = Path(args.out_dir).expanduser()
    n_saved  = 0

    with torch.no_grad():
        img_idx = 0
        for batch_imgs, batch_labels in loader:
            batch_imgs  = batch_imgs.to(device)
            fake_nir, _ = net_1(batch_imgs)           # G_VIS→NIR forward pass

            for i in range(fake_nir.size(0)):
                src_path, _ = dataset.samples[img_idx]
                class_name  = dataset.classes[batch_labels[i].item()]
                fname       = Path(src_path).name

                # Rename: 001_L_VIS_11.png → 001_L_fakeNIR_11.png
                out_fname   = fname.replace('_VIS_', '_fakeNIR_')

                out_img = inv_normalize(fake_nir[i].cpu())
                bgr     = tensor_to_bgr_u8(out_img)

                out_cls = out_root / class_name
                out_cls.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_cls / out_fname), bgr)

                img_idx += 1
                n_saved += 1

            if n_saved % 50 == 0 or n_saved == len(dataset):
                print(f'  {n_saved}/{len(dataset)}')

    print(f'\nDone.  {n_saved} fake NIR images → {out_root}')


if __name__ == '__main__':
    main()
