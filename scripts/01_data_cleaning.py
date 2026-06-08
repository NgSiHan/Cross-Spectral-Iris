#!/usr/bin/env python3
"""
Step 1: Automated data cleaning for PolyU bi-spectral iris database.

Runs CVRL segmentation, geometric quality checks (Level 1), rubber-sheet
normalization, and texture quality checks (Level 2) on every PolyU image.
Writes quality_results.csv and prints a per-subject retention summary.

Expected dataset layout
-----------------------
  <polyu_dir>/<subject_id>/<eye>/<spectrum>/<subject_id>_<eye>_<spectrum>_<n>.tiff
  e.g.  001/L/NIR/001_L_NIR_1.tiff     (eye = L | R,  spectrum = NIR | VIS,  n = 1..15)

Train / test split (standard PolyU protocol)
--------------------------------------------
  Instances  1–10 → train
  Instances 11–15 → test

Subject retention rule
----------------------
  A subject is retained if it has ≥ MIN_IMAGES_PER_SPECTRUM (default 5) valid
  images for EACH spectrum (NIR and VIS), summed across both eyes.

Usage example (Linux)
---------------------
  # Minimal — uses model defaults from the repo's models/ directory:
  python scripts/01_data_cleaning.py \\
      --polyu_dir  ~/iris_workspace/data/PolyU \\
      --output_dir ~/iris_workspace/data/processed/quality_report

  # Explicit model paths (if running from a non-standard location):
  python scripts/01_data_cleaning.py \\
      --polyu_dir    ~/iris_workspace/data/PolyU \\
      --output_dir   ~/iris_workspace/data/processed/quality_report \\
      --mask_model   /path/to/nestedsharedatrousresunet-006-0.028214-maskIoU-0.938446.pth \\
      --circle_model /path/to/resnet18-027-0.008222-maskIoU-0.967159.pth
"""

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Add repo's src/ to path so network.py (NestedSharedAtrousResUNet etc.)
# is importable without any external-repo dependency.
# ---------------------------------------------------------------------------
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
# Default model paths (relative to repo root)
# ---------------------------------------------------------------------------
_DEFAULT_MASK_MODEL = (
    _REPO_ROOT / "models" /
    "nestedsharedatrousresunet-006-0.028214-maskIoU-0.938446.pth"
)
_DEFAULT_CIRCLE_MODEL = (
    _REPO_ROOT / "models" /
    "resnet18-027-0.008222-maskIoU-0.967159.pth"
)


# ---------------------------------------------------------------------------
# IrisSegNorm – segmentation + rubber-sheet normalisation
# Adapted from CVRL HDBIF irisRecognition.py; BSIF encoding stripped out.
# ---------------------------------------------------------------------------

class IrisSegNorm:
    """
    Wraps the CVRL NestedSharedAtrousResUNet (mask) and ResNet-18 (circles).
    Provides fix_image, segment, circApprox, and cartToPol_torch.
    All geometry identical to the HDBIF irisRecognition implementation.
    """

    NET_INPUT_SIZE = (320, 240)   # (W, H) passed to cv2.resize
    ISO_RES        = (640, 480)   # PIL resize target (W, H)
    POLAR_HEIGHT   = 64
    POLAR_WIDTH    = 512

    def __init__(self, mask_model_path: str, circle_model_path: str,
                 cuda: bool = True):
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
            self._circ_model.fc = fclayer(out_n=6)
            self._circ_model.load_state_dict(
                torch.load(circle_model_path, map_location=self.device,
                           weights_only=True))
            self._circ_model = self._circ_model.to(self.device).eval()

    # ------------------------------------------------------------------
    def fix_image(self, im: Image.Image) -> Image.Image:
        """Pad/crop to 4:3 then resize to 640×480 (ISO convention)."""
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
        """Return binary occlusion mask, same (W, H) as im, uint8 {0, 255}."""
        w, h = im.size
        arr = cv2.resize(np.array(im), self.NET_INPUT_SIZE,
                         cv2.INTER_LINEAR_EXACT)
        inp = self._tfm(arr).unsqueeze(0).to(self.device)
        logit = self._mask_model(inp)[0]
        m = (torch.sigmoid(logit) > 0.5).cpu().numpy()[0].astype(np.uint8) * 255
        return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST_EXACT)

    @torch.inference_mode()
    def circApprox(self, im: Image.Image):
        """Return (pupil_xyr, iris_xyr), each np.array([cx, cy, r])."""
        w, h = im.size
        arr = cv2.resize(np.array(im), self.NET_INPUT_SIZE,
                         cv2.INTER_LINEAR_EXACT)
        inp = self._tfm(arr).unsqueeze(0).repeat(1, 3, 1, 1).to(self.device)
        out = self._circ_model(inp).tolist()[0]
        diag = math.sqrt(w ** 2 + h ** 2)
        return (
            np.array([out[0] * w,  out[1] * h,  out[2] * 0.5 * 0.8 * diag]),
            np.array([out[3] * w,  out[4] * h,  out[5] * 0.5 * diag]),
        )

    # ------------------------------------------------------------------
    # grid_sample helper — identical to CVRL HDBIF irisRecognition.grid_sample
    @torch.inference_mode()
    def _grid_sample(self, tensor, grid, mode):
        N, C, H, W = tensor.shape
        gx = grid[..., 0]
        gy = grid[..., 1]
        gx = ((gx + 1) / 2 * W - 0.5) / (W - 1) * 2 - 1
        gy = ((gy + 1) / 2 * H - 0.5) / (H - 1) * 2 - 1
        g2 = torch.stack([gx, gy], dim=-1)
        return torch.nn.functional.grid_sample(
            tensor, g2, mode=mode, align_corners=True,
            padding_mode="border")

    @torch.inference_mode()
    def cartToPol_torch(self, im: Image.Image, mask: np.ndarray,
                        pupil_xyr, iris_xyr):
        """
        Daugman rubber-sheet normalisation.
        Returns (image_polar, mask_polar) both (POLAR_HEIGHT, POLAR_WIDTH) uint8.
        Logic identical to CVRL HDBIF irisRecognition.cartToPol_torch.
        """
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

        # Pupil and iris boundary points — shape (1, pW)
        px_pts = p[:, 0:1] + p[:, 2:3] * torch.cos(theta).reshape(1, pW)
        py_pts = p[:, 1:2] + p[:, 2:3] * torch.sin(theta).reshape(1, pW)
        ix_pts = i[:, 0:1] + i[:, 2:3] * torch.cos(theta).reshape(1, pW)
        iy_pts = i[:, 1:2] + i[:, 2:3] * torch.sin(theta).reshape(1, pW)

        radius = (torch.linspace(1, pH, pH) / pH).reshape(pH, 1).to(self.device)

        # matmul((pH,1), (1,1,pW)) → (1, pH, pW) via broadcasting
        pxC = torch.matmul(1 - radius, px_pts.reshape(-1, 1, pW))
        pyC = torch.matmul(1 - radius, py_pts.reshape(-1, 1, pW))
        ixC = torch.matmul(radius,     ix_pts.reshape(-1, 1, pW))
        iyC = torch.matmul(radius,     iy_pts.reshape(-1, 1, pW))

        x = (pxC + ixC).float()
        y = (pyC + iyC).float()
        x_norm = ((x - 1) / (width  - 1)) * 2 - 1
        y_norm = ((y - 1) / (height - 1)) * 2 - 1

        grid = torch.cat(
            [x_norm.unsqueeze(-1), y_norm.unsqueeze(-1)], dim=-1)  # 1×pH×pW×2

        img_p = torch.clamp(
            torch.round(self._grid_sample(img_t, grid, "bilinear")), 0, 255)
        msk_p = (self._grid_sample(msk_t, grid, "nearest") > 0.5).long() * 255

        return (img_p[0, 0].cpu().numpy().astype(np.uint8),
                msk_p[0, 0].cpu().numpy().astype(np.uint8))


# ---------------------------------------------------------------------------
# Level 1 quality checks (Cartesian domain)
# ---------------------------------------------------------------------------

def level1_check(pxyr, ixyr, mask_cart, min_pr: int = 12, min_ir: int = 16):
    """
    Apply geometric checks and visible-iris-area check in Cartesian space.

    Returns (passed: bool, visible_fraction: float).
    passed=False → discard this image.
    visible_fraction = unoccluded annulus pixels / π*(ir²−pr²).
    """
    px, py, pr = pxyr
    ix, iy, ir = ixyr

    if ir <= pr:
        return False, 0.0
    if pr < min_pr or ir < min_ir:
        return False, 0.0
    ratio = pr / ir
    if ratio < 0.1 or ratio > 0.8:
        return False, 0.0
    if math.sqrt((px - ix) ** 2 + (py - iy) ** 2) / ir > 0.50:
        return False, 0.0

    H, W = mask_cart.shape
    ys, xs = np.mgrid[0:H, 0:W]
    d_iris  = np.hypot(xs - ix, ys - iy)
    d_pupil = np.hypot(xs - px, ys - py)
    annulus = (d_iris <= ir) & (d_pupil >= pr)
    theoretical_area = math.pi * (ir ** 2 - pr ** 2)
    if theoretical_area <= 0:
        return False, 0.0
    vis_frac = float(np.sum(annulus & (mask_cart > 127))) / theoretical_area
    if vis_frac < 0.10:
        return False, round(vis_frac, 4)

    return True, round(vis_frac, 4)


# ---------------------------------------------------------------------------
# Level 2 quality checks (polar domain)
# ---------------------------------------------------------------------------

def _mean_gradient(polar_img: np.ndarray) -> float:
    gx = cv2.Sobel(polar_img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(polar_img, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(np.sqrt(gx ** 2 + gy ** 2)))


def level2_check(mask_polar: np.ndarray, polar_img: np.ndarray,
                 gradient_threshold: float):
    """
    Returns (passed: bool, unmasked_frac: float, mean_grad: float).
    Checks: ≥60% unmasked polar pixels, and gradient ≥ P10 threshold.
    """
    unmasked_frac = float(np.sum(mask_polar > 127)) / mask_polar.size
    if unmasked_frac < 0.60:
        return False, round(unmasked_frac, 4), 0.0
    mg = _mean_gradient(polar_img)
    if mg < gradient_threshold:
        return False, round(unmasked_frac, 4), round(mg, 4)
    return True, round(unmasked_frac, 4), round(mg, 4)


# ---------------------------------------------------------------------------
# Dataset iterator
# ---------------------------------------------------------------------------

def iter_images(polyu_dir: Path):
    """
    Yield (subject_id, eye, spectrum, instance, filepath) for every image.
    """
    for subj_dir in sorted(polyu_dir.iterdir()):
        if not subj_dir.is_dir():
            continue
        subject_id = subj_dir.name
        for eye in ("L", "R"):
            for spectrum in ("NIR", "VIS"):
                spec_dir = subj_dir / eye / spectrum
                if not spec_dir.is_dir():
                    continue
                tiffs = sorted(
                    spec_dir.glob("*.tiff"),
                    key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
                )
                for fp in tiffs:
                    instance = int(fp.stem.rsplit("_", 1)[-1])
                    yield subject_id, eye, spectrum, instance, fp


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="PolyU data cleaning — segmentation + quality filtering")
    ap.add_argument(
        "--polyu_dir", required=True,
        help="Root of PolyU dataset  (e.g. ~/iris_workspace/data/PolyU)")
    ap.add_argument(
        "--output_dir", required=True,
        help="Directory to write quality_results.csv")
    ap.add_argument(
        "--mask_model",
        default=str(_DEFAULT_MASK_MODEL),
        help=f"NestedSharedAtrousResUNet checkpoint  [default: models/ in repo]")
    ap.add_argument(
        "--circle_model",
        default=str(_DEFAULT_CIRCLE_MODEL),
        help=f"ResNet-18 circle-regressor checkpoint  [default: models/ in repo]")
    ap.add_argument(
        "--no_cuda", action="store_true",
        help="Disable GPU (default: use CUDA if available)")
    ap.add_argument(
        "--min_images_per_spectrum", type=int, default=5,
        help="Min valid images per spectrum to keep a subject  [default: 5]")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    polyu_dir  = Path(args.polyu_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_path   = Path(args.mask_model).expanduser().resolve()
    circle_path = Path(args.circle_model).expanduser().resolve()

    for p in (mask_path, circle_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Model not found: {p}\n"
                "Make sure the .pth files are in models/ (they should be "
                "committed to the repo).")

    seg = IrisSegNorm(
        mask_model_path   = str(mask_path),
        circle_model_path = str(circle_path),
        cuda              = not args.no_cuda,
    )
    print(f"Device : {seg.device}")
    print(f"Dataset: {polyu_dir}")

    # ── Pass 1: segmentation + Level 1 checks + polar normalisation ───────
    all_images = list(iter_images(polyu_dir))
    print(f"Images found: {len(all_images)}")

    records        = []   # one dict per image
    grad_values    = []   # gradient of each L1-passing image (for P10)
    l1_polar_refs  = []   # (rec_idx, mask_polar, polar_img) for L1-passers

    for subject_id, eye, spectrum, instance, fpath in tqdm(all_images, desc="Pass 1"):
        split = "train" if instance <= 10 else "test"
        rec = dict(
            subject_id       = subject_id,
            eye              = eye,
            spectrum         = spectrum,
            filename         = str(fpath),
            instance         = instance,
            split            = split,
            level1_pass      = False,
            visible_fraction = 0.0,
            unmasked_polar   = 0.0,
            mean_gradient    = 0.0,
            level2_pass      = False,
            keep             = False,
        )

        try:
            im_raw = Image.open(fpath).convert("L")
            im     = seg.fix_image(im_raw)
            mask   = seg.segment(im)
            pxyr, ixyr = seg.circApprox(im)

            l1_pass, vis_frac = level1_check(pxyr, ixyr, mask)
            rec["visible_fraction"] = vis_frac
            rec["level1_pass"]      = l1_pass

            if l1_pass:
                polar_img, mask_polar = seg.cartToPol_torch(im, mask, pxyr, ixyr)
                if polar_img is not None:
                    mg = _mean_gradient(polar_img)
                    rec["mean_gradient"] = round(mg, 4)
                    grad_values.append(mg)
                    l1_polar_refs.append((len(records), mask_polar, polar_img))

        except Exception as exc:
            print(f"  ERROR {fpath.name}: {exc}")

        records.append(rec)

    # ── Gradient threshold: P10 of all L1-passing images ──────────────────
    if grad_values:
        grad_threshold = float(np.percentile(grad_values, 10))
        print(f"\nGradient P10 threshold : {grad_threshold:.4f}"
              f"  ({len(grad_values)} L1-passing images)")
    else:
        grad_threshold = 0.0
        print("\nWARNING: no L1-passing images; gradient threshold set to 0.")

    # ── Pass 2: Level 2 checks ────────────────────────────────────────────
    for rec_idx, mask_polar, polar_img in tqdm(l1_polar_refs, desc="Pass 2"):
        rec = records[rec_idx]
        l2_pass, unmasked, _ = level2_check(mask_polar, polar_img, grad_threshold)
        rec["level2_pass"]    = l2_pass
        rec["unmasked_polar"] = unmasked

    # ── Subject retention ─────────────────────────────────────────────────
    # Count valid images per (subject, spectrum) across both eyes.
    valid_counts: dict = defaultdict(lambda: defaultdict(int))
    for rec in records:
        if rec["level1_pass"] and rec["level2_pass"]:
            valid_counts[rec["subject_id"]][rec["spectrum"]] += 1

    kept_subjects = {
        subj for subj, cnt in valid_counts.items()
        if cnt.get("NIR", 0) >= args.min_images_per_spectrum
        and cnt.get("VIS", 0) >= args.min_images_per_spectrum
    }

    for rec in records:
        if (rec["level1_pass"] and rec["level2_pass"]
                and rec["subject_id"] in kept_subjects):
            rec["keep"] = True

    # ── Write CSV ──────────────────────────────────────────────────────────
    csv_path = output_dir / "quality_results.csv"
    fieldnames = [
        "subject_id", "eye", "spectrum", "filename", "instance", "split",
        "level1_pass", "visible_fraction", "unmasked_polar",
        "mean_gradient", "level2_pass", "keep",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec[k] for k in fieldnames})

    # ── Summary ───────────────────────────────────────────────────────────
    n_l1   = sum(r["level1_pass"] for r in records)
    n_l2   = sum(r["level1_pass"] and r["level2_pass"] for r in records)
    n_kept = sum(r["keep"] for r in records)

    print(f"\n{'=' * 55}")
    print(f"Total images processed      : {len(records)}")
    print(f"Passed Level 1 (geometric)  : {n_l1}")
    print(f"Passed Level 2 (texture)    : {n_l2}")
    print(f"Images in retained subjects : {n_kept}")
    print(f"Retained subjects           : {len(kept_subjects)} / 209")
    print(f"  (expected ~140 per Wang & Kumar 2019)")
    nir_kept = sum(r["keep"] and r["spectrum"] == "NIR" for r in records)
    vis_kept = sum(r["keep"] and r["spectrum"] == "VIS" for r in records)
    print(f"  NIR kept: {nir_kept}   VIS kept: {vis_kept}")
    print(f"\nCSV → {csv_path}")

    if len(kept_subjects) < 100:
        print("\nWARNING: fewer subjects than expected. "
              "Check that model .pth files loaded correctly.")


if __name__ == "__main__":
    main()
