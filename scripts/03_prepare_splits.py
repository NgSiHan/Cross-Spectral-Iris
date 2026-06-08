#!/usr/bin/env python3
"""
Step 3 (data): Reorganise normalised polar codes into subject-disjoint
train / eval splits using symlinks.  No re-processing needed.

Subject split (standard PolyU protocol)
----------------------------------------
  Train (GAN training)  : subjects 001–167, ALL 15 instances
  Eval  (ArcIris match) : subjects 168–209
    - NIR gallery : instances  1–10  (from NIR/)
    - VIS probe   : instances 11–15  (from VIS_Valid/)

Output structure
----------------
  <output_dir>/<variant>/
    train/
      NIR/   <subj_id>_<eye>/   ← instances 1–15 of subjects 001–167
      VIS/   <subj_id>_<eye>/   ← instances 1–15 of subjects 001–167
    eval/
      NIR_gallery/ <subj_id>_<eye>/  ← instances  1–10 of subjects 168–209
      VIS_probe/   <subj_id>_<eye>/  ← instances 11–15 of subjects 168–209

Usage
-----
  python scripts/03_prepare_splits.py \\
      --input_dir  ~/data/processed/normalized_codes \\
      --output_dir ~/data/processed/normalized_codes_split \\
      --variants grey_clahe grey_noclahe
"""

import argparse
import os
from pathlib import Path


TRAIN_MAX = 167
ALL_VARIANTS = ("grey_clahe", "grey_noclahe", "red_clahe", "red_noclahe")


def subject_id(class_name: str) -> int:
    """'001_L' → 1,  '168_R' → 168"""
    return int(class_name.split("_")[0])


def symlink_files(src_dir: Path, dst_dir: Path):
    """Symlink every .png in src_dir into dst_dir (file-level merge)."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for img in sorted(src_dir.glob("*.png")):
        link = dst_dir / img.name
        if not link.exists() and not link.is_symlink():
            link.symlink_to(img.resolve())


def symlink_dir(src_dir: Path, dst_dir: Path):
    """Symlink src_dir itself as dst_dir."""
    dst_dir.parent.mkdir(parents=True, exist_ok=True)
    if dst_dir.exists() or dst_dir.is_symlink():
        dst_dir.unlink()
    dst_dir.symlink_to(src_dir.resolve())


def main():
    ap = argparse.ArgumentParser(
        description="Create subject-disjoint train/eval splits via symlinks")
    ap.add_argument("--input_dir",  required=True,
                    help="Base dir from Step 2  (contains variant sub-dirs)")
    ap.add_argument("--output_dir", required=True,
                    help="Destination for split dataset")
    ap.add_argument("--variants", nargs="+", default=["grey_clahe", "grey_noclahe"],
                    help="Which variants to process")
    ap.add_argument("--train_max", type=int, default=TRAIN_MAX,
                    help="Subjects <= this go into training  [default 167]")
    args = ap.parse_args()

    input_dir  = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    for variant in args.variants:
        var_in  = input_dir  / variant
        var_out = output_dir / variant
        if not var_in.exists():
            print(f"  Skipping {variant} — not found at {var_in}")
            continue

        print(f"\nProcessing {variant} ...")

        # ── Training split: subjects 001–train_max, all 15 instances ─────
        # Merge NIR/ (instances 1–10) + NIR_Valid/ (instances 11–15) per class
        for src_split, dst_spec in [("NIR",       "train/NIR"),
                                     ("NIR_Valid",  "train/NIR"),
                                     ("VIS",        "train/VIS"),
                                     ("VIS_Valid",  "train/VIS")]:
            src_root = var_in / src_split
            if not src_root.exists():
                continue
            for cls_dir in sorted(src_root.iterdir()):
                if not cls_dir.is_dir():
                    continue
                if subject_id(cls_dir.name) <= args.train_max:
                    symlink_files(cls_dir, var_out / dst_spec / cls_dir.name)

        # ── Eval split: subjects (train_max+1)–209 ────────────────────────
        # Gallery : NIR instances  1–10  (from NIR/)
        # Probe   : VIS instances 11–15  (from VIS_Valid/)
        for src_split, dst_split in [("NIR",       "eval/NIR_gallery"),
                                      ("VIS_Valid", "eval/VIS_probe")]:
            src_root = var_in / src_split
            if not src_root.exists():
                continue
            for cls_dir in sorted(src_root.iterdir()):
                if not cls_dir.is_dir():
                    continue
                if subject_id(cls_dir.name) > args.train_max:
                    symlink_dir(cls_dir, var_out / dst_split / cls_dir.name)

        # ── Summary ───────────────────────────────────────────────────────
        # Use os.walk(followlinks=True) — Path.rglob() does not traverse
        # symlinked subdirectories in Python < 3.12, which would give a
        # false zero count for the eval split (directory-level symlinks).
        for split in ("train/NIR", "train/VIS",
                      "eval/NIR_gallery", "eval/VIS_probe"):
            p = var_out / split
            if p.exists():
                n_cls  = sum(1 for x in p.iterdir() if x.is_dir())
                n_imgs = sum(
                    1 for _, _, files in os.walk(p, followlinks=True)
                    for f in files if f.endswith(".png")
                )
                print(f"  {split:22s} : {n_cls:3d} classes  {n_imgs:5d} images")

    print("\nDone.")


if __name__ == "__main__":
    main()
