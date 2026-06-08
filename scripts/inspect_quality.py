#!/usr/bin/env python3
"""
Visual inspection of quality_results.csv.

Generates a figure for each failure category and saves them to --output_dir.
Run this locally on Windows (or Linux) — just point --polyu_dir at wherever
the PolyU images live; paths in the CSV are ignored and reconstructed.

Usage:
  python scripts/inspect_quality.py \
      --csv       quality_results.csv \
      --polyu_dir "C:/dev/polyu_iris_database/PolyU_Cross_Submit/PolyU_Cross_Session_1/PolyU_Cross_Iris" \
      --output_dir quality_inspection
"""

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless — saves PNG without a display
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_image(polyu_dir: Path, subject_id, eye, spectrum, instance) -> np.ndarray:
    p = polyu_dir / subject_id / eye / spectrum
    fname = f"{subject_id}_{eye}_{spectrum}_{instance}.tiff"
    return np.array(Image.open(p / fname).convert("L"))


def sample_rows(rows, n=6, seed=42):
    rng = random.Random(seed)
    return rng.sample(rows, min(n, len(rows)))


def img_grid(rows, polyu_dir, title, subtitle_fn, ncols=6, figsize_per=(2.2, 2.5)):
    n = len(rows)
    if n == 0:
        return None
    ncols = min(ncols, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * figsize_per[0], nrows * figsize_per[1] + 0.6))
    fig.suptitle(title, fontsize=11, fontweight="bold")
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]
    for ax, rec in zip(axes_flat, rows):
        try:
            img = load_image(
                polyu_dir, rec["subject_id"], rec["eye"],
                rec["spectrum"], rec["instance"])
            ax.imshow(img, cmap="gray", aspect="auto")
        except Exception as e:
            ax.text(0.5, 0.5, f"load\nerror\n{e}", ha="center", va="center",
                    fontsize=6, transform=ax.transAxes)
        ax.set_title(subtitle_fn(rec), fontsize=6, pad=2)
        ax.axis("off")
    for ax in axes_flat[n:]:
        ax.axis("off")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",        required=True)
    ap.add_argument("--polyu_dir",  required=True)
    ap.add_argument("--output_dir", default="quality_inspection")
    ap.add_argument("--n_samples",  type=int, default=12,
                    help="Images to show per panel [default 12]")
    return ap.parse_args()


def main():
    args = parse_args()
    polyu_dir  = Path(args.polyu_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(args.csv)))
    N    = args.n_samples

    # ── Classify rows ──────────────────────────────────────────────────────
    l1_fail  = [r for r in rows if r["level1_pass"] == "False"]
    l2_occl  = [r for r in rows
                if r["level1_pass"] == "True"
                and r["level2_pass"] == "False"
                and float(r["unmasked_polar"]) < 0.60]
    l2_grad  = [r for r in rows
                if r["level1_pass"] == "True"
                and r["level2_pass"] == "False"
                and float(r["unmasked_polar"]) >= 0.60]
    l2_grad_nir = [r for r in l2_grad if r["spectrum"] == "NIR"]
    l2_grad_vis = [r for r in l2_grad if r["spectrum"] == "VIS"]

    kept     = [r for r in rows if r["keep"] == "True"]
    kept_nir = [r for r in kept if r["spectrum"] == "NIR"]
    kept_vis = [r for r in kept if r["spectrum"] == "VIS"]

    dropped_subjs = {
        r["subject_id"] for r in rows
        if r["level1_pass"] == "True" and r["level2_pass"] == "True"
        and r["keep"] == "False"
    }
    dropped_images = [r for r in rows
                      if r["subject_id"] in dropped_subjs
                      and r["level1_pass"] == "True"
                      and r["level2_pass"] == "True"]

    # ── Print summary ──────────────────────────────────────────────────────
    print(f"Total images          : {len(rows)}")
    print(f"L1 failures           : {len(l1_fail)}")
    print(f"L2 occlusion failures : {len(l2_occl)}  "
          f"(NIR={sum(1 for r in l2_occl if r['spectrum']=='NIR')}  "
          f"VIS={sum(1 for r in l2_occl if r['spectrum']=='VIS')})")
    print(f"L2 gradient failures  : {len(l2_grad)}  "
          f"(NIR={len(l2_grad_nir)}  VIS={len(l2_grad_vis)})")
    print(f"Dropped subjects      : {sorted(dropped_subjs, key=int)}")
    print(f"Kept                  : {len(kept)}")

    # ── Panel 1: L1 failures (all of them, ≤N) ────────────────────────────
    fig = img_grid(
        sample_rows(l1_fail, N),
        polyu_dir,
        title=f"Level 1 failures (geometric) — {len(l1_fail)} total",
        subtitle_fn=lambda r: (
            f"{r['subject_id']} {r['eye']} {r['spectrum']} #{r['instance']}\n"
            f"visfrac={r['visible_fraction']}"),
    )
    if fig:
        fig.savefig(output_dir / "01_l1_failures.png", dpi=120)
        plt.close(fig)
        print(f"Saved 01_l1_failures.png")

    # ── Panel 2: L2 occlusion failures ────────────────────────────────────
    fig = img_grid(
        sample_rows(l2_occl, N),
        polyu_dir,
        title=f"Level 2 failures — occlusion mask <60%  ({len(l2_occl)} total)",
        subtitle_fn=lambda r: (
            f"{r['subject_id']} {r['eye']} {r['spectrum']} #{r['instance']}\n"
            f"mask={r['unmasked_polar']}"),
    )
    if fig:
        fig.savefig(output_dir / "02_l2_occlusion.png", dpi=120)
        plt.close(fig)
        print(f"Saved 02_l2_occlusion.png")

    # ── Panel 3: L2 gradient failures — NIR ───────────────────────────────
    fig = img_grid(
        sample_rows(l2_grad_nir, N),
        polyu_dir,
        title=f"Level 2 failures — gradient too low (NIR)  ({len(l2_grad_nir)} total)",
        subtitle_fn=lambda r: f"{r['subject_id']} {r['eye']} #{r['instance']}\ngrad={r['mean_gradient']}",
    )
    if fig:
        fig.savefig(output_dir / "03_l2_gradient_nir.png", dpi=120)
        plt.close(fig)
        print(f"Saved 03_l2_gradient_nir.png")

    # ── Panel 4: L2 gradient failures — VIS ───────────────────────────────
    fig = img_grid(
        sample_rows(l2_grad_vis, N),
        polyu_dir,
        title=f"Level 2 failures — gradient too low (VIS)  ({len(l2_grad_vis)} total)",
        subtitle_fn=lambda r: f"{r['subject_id']} {r['eye']} #{r['instance']}\ngrad={r['mean_gradient']}",
    )
    if fig:
        fig.savefig(output_dir / "04_l2_gradient_vis.png", dpi=120)
        plt.close(fig)
        print(f"Saved 04_l2_gradient_vis.png")

    # ── Panel 5: Dropped subjects — best images they had ──────────────────
    # One representative NIR and one VIS image per dropped subject
    rep_rows = []
    for subj in sorted(dropped_subjs, key=int):
        for spec in ("NIR", "VIS"):
            candidates = [r for r in dropped_images
                          if r["subject_id"] == subj and r["spectrum"] == spec]
            if candidates:
                rep_rows.append(candidates[0])
    fig = img_grid(
        rep_rows,
        polyu_dir,
        title=f"Dropped subjects — one NIR + one VIS per subject  ({len(dropped_subjs)} subjects)",
        subtitle_fn=lambda r: f"{r['subject_id']} {r['spectrum']}\ngrad={r['mean_gradient']}",
        ncols=6,
    )
    if fig:
        fig.savefig(output_dir / "05_dropped_subjects.png", dpi=120)
        plt.close(fig)
        print(f"Saved 05_dropped_subjects.png")

    # ── Panel 6: Reference — sample kept NIR and VIS for comparison ───────
    ref_rows = sample_rows(kept_nir, N // 2) + sample_rows(kept_vis, N // 2)
    fig = img_grid(
        ref_rows,
        polyu_dir,
        title=f"Reference — sample KEPT images (NIR + VIS)",
        subtitle_fn=lambda r: f"{r['subject_id']} {r['eye']} {r['spectrum']}\ngrad={r['mean_gradient']}",
    )
    if fig:
        fig.savefig(output_dir / "06_kept_reference.png", dpi=120)
        plt.close(fig)
        print(f"Saved 06_kept_reference.png")

    # ── Gradient histogram ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Gradient distribution by spectrum (L1-passing images)", fontweight="bold")

    for ax, spec, color in zip(axes, ("NIR", "VIS"), ("steelblue", "coral")):
        grads = [float(r["mean_gradient"])
                 for r in rows
                 if r["level1_pass"] == "True" and r["spectrum"] == spec]
        ax.hist(grads, bins=60, color=color, edgecolor="none", alpha=0.8)
        ax.set_title(spec)
        ax.set_xlabel("Mean Sobel gradient magnitude")
        ax.set_ylabel("Count")
        if grads:
            p10 = float(np.percentile(grads, 10))
            ax.axvline(p10, color="red", linestyle="--", linewidth=1.5,
                       label=f"P10={p10:.2f}")
            ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(output_dir / "07_gradient_histogram.png", dpi=120)
    plt.close(fig)
    print(f"Saved 07_gradient_histogram.png")

    print(f"\nAll figures saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
