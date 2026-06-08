#!/usr/bin/env python3
"""
Step 2: Rubber-sheet normalisation → GAN-ready ImageFolder dataset.

Reads quality_results.csv (from Step 1), determines which (subject, eye)
classes have enough NIR+VIS paired instances in BOTH splits, and writes
64×512 normalised polar codes into the torchvision ImageFolder layout:

  <output_dir>/<variant>/<split_folder>/<subj_id>_<eye>/<subj_id>_<eye>_<spec>_<n>.png

  split_folder : NIR / VIS          (train, instances 1-10)
               : NIR_Valid / VIS_Valid  (test,  instances 11-15)

Variants produced
-----------------
  grey_noclahe  greyscale polar code, no CLAHE
  grey_clahe    greyscale polar code + CLAHE
  red_noclahe   VIS R-channel polar / NIR greyscale polar, no CLAHE
  red_clahe     same + CLAHE

GAN parity guarantee
--------------------
  Only instances valid in BOTH NIR and VIS are saved, so every class folder
  in NIR has exactly the same image count as the matching class folder in VIS
  (and likewise for NIR_Valid / VIS_Valid).  The same set of class folders
  appears in all four split folders.

Usage (Linux)
-------------
  python scripts/02_preprocess.py \\
      --polyu_dir  ~/iris_workspace/data/PolyU \\
      --csv        ~/iris_workspace/data/processed/quality_report/quality_results.csv \\
      --output_dir ~/iris_workspace/data/processed/normalized_codes
"""

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import models
from torchvision.transforms import Compose, Normalize, ToTensor

from network import NestedSharedAtrousResUNet, conv, fclayer


# ---------------------------------------------------------------------------
# Default model paths
# ---------------------------------------------------------------------------
_DEFAULT_MASK_MODEL = (
    _REPO_ROOT / "models" /
    "nestedsharedatrousresunet-006-0.028214-maskIoU-0.938446.pth"
)
_DEFAULT_CIRCLE_MODEL = (
    _REPO_ROOT / "models" /
    "resnet18-027-0.008222-maskIoU-0.967159.pth"
)

VARIANTS = ("grey_noclahe", "grey_clahe", "red_noclahe", "red_clahe")


# ---------------------------------------------------------------------------
# IrisSegNorm — identical to Step 1 (keeps scripts self-contained)
# ---------------------------------------------------------------------------

class IrisSegNorm:
    NET_INPUT_SIZE = (320, 240)
    ISO_RES        = (640, 480)
    POLAR_HEIGHT   = 64
    POLAR_WIDTH    = 512

    def __init__(self, mask_model_path, circle_model_path, cuda=True):
        self.device = torch.device(
            "cuda" if cuda and torch.cuda.is_available() else "cpu")
        self._tfm = Compose([ToTensor(), Normalize(mean=(0.5,), std=(0.5,))])

        with torch.inference_mode():
            self._mask_model = NestedSharedAtrousResUNet(
                1, 1, width=32, resolution=(240, 320))
            self._mask_model.load_state_dict(
                torch.load(mask_model_path, map_location=self.device,
                           weights_only=True))
            self._mask_model = self._mask_model.to(self.device).eval()

            self._circ_model = models.resnet18()
            self._circ_model.avgpool = conv(in_channels=512, out_n=6)
            self._circ_model.fc     = fclayer(out_n=6)
            self._circ_model.load_state_dict(
                torch.load(circle_model_path, map_location=self.device,
                           weights_only=True))
            self._circ_model = self._circ_model.to(self.device).eval()

    def fix_image(self, im: Image.Image) -> Image.Image:
        """Pad/crop to 4:3 then resize to 640×480."""
        w, h = im.size
        ar = float(w) / float(h)
        if 1.333 <= ar <= 1.334:
            return im.copy().resize(self.ISO_RES)
        if ar < 1.333:
            w_new = h * (4.0 / 3.0)
            pad = (w_new - w) / 2
            out = Image.new(im.mode, (int(w_new), h), 127)
            out.paste(im, (int(pad), 0))
            return out.resize(self.ISO_RES)
        h_new = w * (3.0 / 4.0)
        pad = (h_new - h) / 2
        out = Image.new(im.mode, (w, int(h_new)), 127)
        out.paste(im, (0, int(pad)))
        return out.resize(self.ISO_RES)

    @torch.inference_mode()
    def segment(self, im: Image.Image) -> np.ndarray:
        """Return binary occlusion mask (uint8 0/255), same size as im."""
        w, h = im.size
        arr = cv2.resize(np.array(im), self.NET_INPUT_SIZE,
                         cv2.INTER_LINEAR_EXACT)
        inp = self._tfm(arr).unsqueeze(0).to(self.device)
        logit = self._mask_model(inp)[0]
        m = (torch.sigmoid(logit) > 0.5).cpu().numpy()[0].astype(np.uint8) * 255
        return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST_EXACT)

    @torch.inference_mode()
    def circApprox(self, im: Image.Image):
        """Return (pupil_xyr, iris_xyr) as np.array([cx, cy, r])."""
        w, h = im.size
        arr = cv2.resize(np.array(im), self.NET_INPUT_SIZE,
                         cv2.INTER_LINEAR_EXACT)
        inp = self._tfm(arr).unsqueeze(0).repeat(1, 3, 1, 1).to(self.device)
        out = self._circ_model(inp).tolist()[0]
        diag = math.sqrt(w ** 2 + h ** 2)
        return (
            np.array([out[0] * w, out[1] * h, out[2] * 0.5 * 0.8 * diag]),
            np.array([out[3] * w, out[4] * h, out[5] * 0.5 * diag]),
        )

    @torch.inference_mode()
    def _grid_sample(self, tensor, grid, mode):
        N, C, H, W = tensor.shape
        gx = grid[..., 0]
        gy = grid[..., 1]
        gx = ((gx + 1) / 2 * W - 0.5) / (W - 1) * 2 - 1
        gy = ((gy + 1) / 2 * H - 0.5) / (H - 1) * 2 - 1
        g2 = torch.stack([gx, gy], dim=-1)
        return torch.nn.functional.grid_sample(
            tensor, g2, mode=mode, align_corners=True, padding_mode="border")

    @torch.inference_mode()
    def cartToPol_torch(self, im: Image.Image, mask: np.ndarray,
                        pupil_xyr, iris_xyr):
        """Daugman rubber-sheet → (image_polar, mask_polar) uint8 64×512."""
        if pupil_xyr is None or iris_xyr is None:
            return None, None
        pH, pW = self.POLAR_HEIGHT, self.POLAR_WIDTH
        img_t = (torch.tensor(np.array(im), dtype=torch.float32)
                 .unsqueeze(0).unsqueeze(0).to(self.device))
        msk_t = (torch.tensor(mask.astype(np.float32))
                 .unsqueeze(0).unsqueeze(0).to(self.device))
        width  = img_t.shape[3]
        height = img_t.shape[2]
        p = torch.tensor(pupil_xyr, dtype=torch.float32).unsqueeze(0).to(self.device)
        i = torch.tensor(iris_xyr,  dtype=torch.float32).unsqueeze(0).to(self.device)
        theta = (2 * math.pi * torch.linspace(0, pW - 1, pW) / pW).to(self.device)
        px_pts = p[:, 0:1] + p[:, 2:3] * torch.cos(theta).reshape(1, pW)
        py_pts = p[:, 1:2] + p[:, 2:3] * torch.sin(theta).reshape(1, pW)
        ix_pts = i[:, 0:1] + i[:, 2:3] * torch.cos(theta).reshape(1, pW)
        iy_pts = i[:, 1:2] + i[:, 2:3] * torch.sin(theta).reshape(1, pW)
        radius = (torch.linspace(1, pH, pH) / pH).reshape(pH, 1).to(self.device)
        pxC = torch.matmul(1 - radius, px_pts.reshape(-1, 1, pW))
        pyC = torch.matmul(1 - radius, py_pts.reshape(-1, 1, pW))
        ixC = torch.matmul(radius,     ix_pts.reshape(-1, 1, pW))
        iyC = torch.matmul(radius,     iy_pts.reshape(-1, 1, pW))
        x = (pxC + ixC).float()
        y = (pyC + iyC).float()
        x_norm = ((x - 1) / (width  - 1)) * 2 - 1
        y_norm = ((y - 1) / (height - 1)) * 2 - 1
        grid = torch.cat([x_norm.unsqueeze(-1), y_norm.unsqueeze(-1)], dim=-1)
        img_p = torch.clamp(
            torch.round(self._grid_sample(img_t, grid, "bilinear")), 0, 255)
        msk_p = (self._grid_sample(msk_t, grid, "nearest") > 0.5).long() * 255
        return (img_p[0, 0].cpu().numpy().astype(np.uint8),
                msk_p[0, 0].cpu().numpy().astype(np.uint8))


# ---------------------------------------------------------------------------
# Pairing plan
# ---------------------------------------------------------------------------

def build_pairing_plan(csv_path: Path, min_train: int, min_test: int) -> dict:
    """
    For each (subject_id, eye) find the instance numbers that are valid in
    BOTH NIR and VIS for each split.  Returns:
      {(subject_id, eye): {'train': [inst, ...], 'test': [inst, ...]}}
    Only classes meeting both thresholds are included.
    """
    valid = defaultdict(set)  # (subj, eye, 'train'|'test', 'NIR'|'VIS') → {instance}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["level1_pass"] == "True" and row["level2_pass"] == "True":
                inst   = int(row["instance"])
                split  = "train" if inst <= 10 else "test"
                key    = (row["subject_id"], row["eye"], split, row["spectrum"])
                valid[key].add(inst)

    all_classes = {(k[0], k[1]) for k in valid}
    plan = {}
    for subj, eye in sorted(all_classes):
        train_paired = sorted(
            valid[(subj, eye, "train", "NIR")] & valid[(subj, eye, "train", "VIS")])
        test_paired  = sorted(
            valid[(subj, eye, "test",  "NIR")] & valid[(subj, eye, "test",  "VIS")])
        if len(train_paired) >= min_train and len(test_paired) >= min_test:
            plan[(subj, eye)] = {"train": train_paired, "test": test_paired}

    return plan


# ---------------------------------------------------------------------------
# Per-pair processing
# ---------------------------------------------------------------------------

def process_pair(seg: IrisSegNorm, polyu_dir: Path,
                 subj: str, eye: str, instance: int,
                 clahe) -> tuple:
    """
    Segment and normalise one NIR+VIS instance pair.
    Returns (nir_variants, vis_variants) — each a dict variant→uint8 ndarray.

    NIR has no colour channels, so 'red' variants are identical to 'grey'.
    VIS 'grey' uses luminance; VIS 'red' uses the R channel only.
    """
    nir_path = polyu_dir / subj / eye / "NIR" / f"{subj}_{eye}_NIR_{instance}.tiff"
    vis_path = polyu_dir / subj / eye / "VIS" / f"{subj}_{eye}_VIS_{instance}.tiff"

    # ── NIR ──────────────────────────────────────────────────────────────
    nir_pil  = Image.open(nir_path).convert("L")
    nir_fix  = seg.fix_image(nir_pil)
    nir_mask = seg.segment(nir_fix)
    nir_pxyr, nir_ixyr = seg.circApprox(nir_fix)
    nir_polar, _ = seg.cartToPol_torch(nir_fix, nir_mask, nir_pxyr, nir_ixyr)
    if nir_polar is None:
        raise ValueError(f"cartToPol failed for {nir_path.name}")
    nir_polar_clahe = clahe.apply(nir_polar)

    nir_variants = {
        "grey_noclahe": nir_polar,
        "grey_clahe":   nir_polar_clahe,
        "red_noclahe":  nir_polar,        # NIR is single-channel; grey == red
        "red_clahe":    nir_polar_clahe,
    }

    # ── VIS ───────────────────────────────────────────────────────────────
    vis_rgb  = Image.open(vis_path).convert("RGB")
    vis_grey = vis_rgb.convert("L")              # luminance (for segmentation)

    vis_fix_grey = seg.fix_image(vis_grey)
    vis_fix_rgb  = seg.fix_image(vis_rgb)

    vis_mask = seg.segment(vis_fix_grey)
    vis_pxyr, vis_ixyr = seg.circApprox(vis_fix_grey)

    # Grey-channel polar
    vis_grey_polar, _ = seg.cartToPol_torch(
        vis_fix_grey, vis_mask, vis_pxyr, vis_ixyr)
    if vis_grey_polar is None:
        raise ValueError(f"cartToPol failed for {vis_path.name}")

    # Red-channel polar — normalise the R slice with the same geometry
    vis_r_pil = vis_fix_rgb.split()[0]           # PIL 'L' image of the R channel
    vis_red_polar, _ = seg.cartToPol_torch(
        vis_r_pil, vis_mask, vis_pxyr, vis_ixyr)
    if vis_red_polar is None:
        raise ValueError(f"cartToPol (red channel) failed for {vis_path.name}")

    vis_variants = {
        "grey_noclahe": vis_grey_polar,
        "grey_clahe":   clahe.apply(vis_grey_polar),
        "red_noclahe":  vis_red_polar,
        "red_clahe":    clahe.apply(vis_red_polar),
    }

    return nir_variants, vis_variants


def save_pair(output_dir: Path, subj: str, eye: str, instance: int, is_test: bool,
              nir_variants: dict, vis_variants: dict):
    """Write all 4 variants for one NIR+VIS pair to their GAN folders."""
    class_folder = f"{subj}_{eye}"
    nir_split = "NIR_Valid" if is_test else "NIR"
    vis_split = "VIS_Valid" if is_test else "VIS"

    for variant in VARIANTS:
        for spec, split_folder, polar in (
            ("NIR", nir_split, nir_variants[variant]),
            ("VIS", vis_split, vis_variants[variant]),
        ):
            out = (output_dir / variant / split_folder / class_folder /
                   f"{subj}_{eye}_{spec}_{instance}.png")
            out.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(polar).save(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="PolyU preprocessing — normalised codes in GAN ImageFolder format")
    ap.add_argument("--polyu_dir",  required=True,
                    help="Root of PolyU dataset  (e.g. ~/iris_workspace/data/PolyU)")
    ap.add_argument("--csv",        required=True,
                    help="quality_results.csv from Step 1")
    ap.add_argument("--output_dir", required=True,
                    help="Base output dir  (variant sub-dirs created inside)")
    ap.add_argument("--mask_model",
                    default=str(_DEFAULT_MASK_MODEL),
                    help="NestedSharedAtrousResUNet checkpoint")
    ap.add_argument("--circle_model",
                    default=str(_DEFAULT_CIRCLE_MODEL),
                    help="ResNet-18 circle-regressor checkpoint")
    ap.add_argument("--no_cuda", action="store_true")
    ap.add_argument("--min_train_pairs", type=int, default=3,
                    help="Min paired NIR+VIS instances per class in train [default 3]")
    ap.add_argument("--min_test_pairs",  type=int, default=1,
                    help="Min paired NIR+VIS instances per class in test  [default 1]")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    polyu_dir  = Path(args.polyu_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    csv_path   = Path(args.csv).expanduser().resolve()
    mask_path  = Path(args.mask_model).expanduser().resolve()
    circ_path  = Path(args.circle_model).expanduser().resolve()

    for p in (polyu_dir, csv_path, mask_path, circ_path):
        if not p.exists():
            raise FileNotFoundError(p)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Build pairing plan ────────────────────────────────────────────────
    print("Building pairing plan from CSV ...")
    plan = build_pairing_plan(csv_path, args.min_train_pairs, args.min_test_pairs)
    n_classes = len(plan)
    n_pairs   = sum(len(v["train"]) + len(v["test"]) for v in plan.values())
    print(f"  Classes included (subject×eye)  : {n_classes}")
    print(f"  Total NIR+VIS instance pairs    : {n_pairs}")
    print(f"  Images to write                 : {n_pairs * 2 * len(VARIANTS)}"
          f"  (pairs × 2 spectra × {len(VARIANTS)} variants)")

    # ── Load models ───────────────────────────────────────────────────────
    seg   = IrisSegNorm(str(mask_path), str(circ_path), cuda=not args.no_cuda)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    print(f"Device : {seg.device}\n")

    # ── Process all pairs ─────────────────────────────────────────────────
    errors = []

    with tqdm(total=n_pairs, desc="Pairs") as pbar:
        for (subj, eye), splits in sorted(plan.items()):
            for split_name, instances in (("train", splits["train"]),
                                          ("test",  splits["test"])):
                is_test = (split_name == "test")
                for instance in instances:
                    try:
                        nir_v, vis_v = process_pair(
                            seg, polyu_dir, subj, eye, instance, clahe)
                        save_pair(output_dir, subj, eye, instance, is_test,
                                  nir_v, vis_v)
                    except Exception as exc:
                        errors.append(f"{subj}_{eye}_{instance}: {exc}")
                    pbar.update(1)

    # ── Summary ───────────────────────────────────────────────────────────
    n_saved = n_pairs - len(errors)
    print(f"\n{'=' * 60}")
    print(f"Pairs saved  : {n_saved} / {n_pairs}")
    print(f"Errors       : {len(errors)}")
    for e in errors[:20]:
        print(f"  {e}")
    if len(errors) > 20:
        print(f"  ... and {len(errors) - 20} more")

    # Parity check: NIR and VIS image counts must match for every variant
    print("\nParity check (NIR == VIS counts per variant):")
    all_ok = True
    for variant in VARIANTS:
        for nf, vf in (("NIR", "VIS"), ("NIR_Valid", "VIS_Valid")):
            n_nir = sum(1 for _ in (output_dir / variant / nf).rglob("*.png"))
            n_vis = sum(1 for _ in (output_dir / variant / vf).rglob("*.png"))
            status = "OK" if n_nir == n_vis else "MISMATCH"
            print(f"  {variant:15s}  {nf:10s}={n_nir:5d}  {vf:10s}={n_vis:5d}  {status}")
            if n_nir != n_vis:
                all_ok = False

    if all_ok:
        print("\nAll counts match — dataset is GAN-ready.")
    else:
        print("\nMISMATCH detected — review errors above.")
    print(f"\nOutput -> {output_dir}")


if __name__ == "__main__":
    main()
