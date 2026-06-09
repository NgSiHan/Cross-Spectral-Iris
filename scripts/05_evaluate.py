#!/usr/bin/env python3
"""
Step 5 – Cross-spectral iris evaluation.

Supported matchers
------------------
  arciris  — CVRL iresnet100 (cosine similarity of 512-dim embeddings).
             Model: ./models-ArcIris/ResNet100_154000.pt
             Fast: all-to-all via matrix multiplication.

  hdbif    — HDBIF BSIF iris codes + Hamming distance.
             Filters: ./filters_pt/finetuned_bsif_eyetracker_data/
               (download from https://notredame.box.com/s/vxwwtm68th0nvdag6dhqn7u64kf2i2r0)
             Note: polar masks were not saved during preprocessing; an
             all-ones mask is used as fallback (slight performance over-estimate).
             Slower: pairwise Python loop.

  osiris   — NOT YET IMPLEMENTED.
             OSIRIS uses log-Gabor IrisCode and is not included in
             OpenSourceIrisRecognition.  See: https://svn.irisa.fr/osiris/

Metrics reported
----------------
  EER (%), TAR @ FAR=0.1%, TAR @ FAR=0.01%, Decidability Index (DI), ROC curve.

Both all-to-all and score-fusion (max per identity) variants are reported,
matching the two-row format in Anderson et al. Table 1.

Evaluation protocol (standard PolyU, subjects 168–209)
-------------------------------------------------------
  Gallery  : NIR_gallery/  — real NIR,        instances 1–10
  Probe    : fake_nir/     — GAN output,      instances 11–15
  Baseline : VIS_probe/    — real VIS (no GAN), instances 11–15

Usage
-----
  cd ~/Cross-Spectral-Iris

  # ArcIris (default)
  python scripts/05_evaluate.py \\
      --matcher   arciris \\
      --gallery   ~/data/processed/normalized_codes_split/grey_clahe/eval/NIR_gallery \\
      --probe     ~/data/processed/fake_nir/grey_clahe_train167 \\
      --baseline  ~/data/processed/normalized_codes_split/grey_clahe/eval/VIS_probe \\
      --model     ./models-ArcIris/ResNet100_154000.pt \\
      --out_dir   results/grey_clahe_train167_arciris

  # HDBIF  (requires filter files downloaded first — see filters_pt/README.txt)
  python scripts/05_evaluate.py \\
      --matcher    hdbif \\
      --gallery    ~/data/processed/normalized_codes_split/grey_clahe/eval/NIR_gallery \\
      --probe      ~/data/processed/fake_nir/grey_clahe_train167 \\
      --baseline   ~/data/processed/normalized_codes_split/grey_clahe/eval/VIS_probe \\
      --filter_dir ./filters_pt/finetuned_bsif_eyetracker_data \\
      --out_dir    results/grey_clahe_train167_hdbif
"""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_curve
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

# ── repo root on sys.path so we can import src.matchers.* ─────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.matchers.arciris_network import iresnet100        # noqa: E402
from src.matchers.hdbif_coder import HDBIFCoder            # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
#  Matcher classes
# ══════════════════════════════════════════════════════════════════════════════

class ArcIrisExtractor:
    """
    Loads iresnet100 from src/matchers/arciris_network.py and extracts
    L2-normalised 512-dim embeddings from 64×512 uint8 greyscale polar images.

    Transform is identical to CVRL irisRecognition.extractVector():
      PIL "L"  →  ToTensor()  →  Normalize(mean=0.5, std=0.5)
    Then repeated to 3 channels to match iresnet100's input expectations.
    """

    MATRIX_SCORING = True   # can use fast matmul for all-to-all

    def __init__(self, model_path: str, device=None):
        self.device = (device if device is not None else
                       torch.device('cuda' if torch.cuda.is_available()
                                    else 'cpu'))
        model_path = Path(model_path).expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(f'ArcIris model not found: {model_path}')

        model = iresnet100(pretrained=False, progress=False)
        state_dict = torch.load(str(model_path), map_location=self.device)
        # Strip DataParallel 'module.' prefix if present
        clean_sd = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(clean_sd, strict=True)
        model.eval().to(self.device)
        self.model = model

        self.transform = Compose([
            ToTensor(),
            Normalize(mean=(0.5,), std=(0.5,)),
        ])
        print(f'ArcIris iresnet100 loaded from {model_path}  '
              f'(device: {self.device})')

    @torch.inference_mode()
    def extract(self, img_path: Path) -> np.ndarray:
        """Return 512-dim L2-normalised float32 embedding."""
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise FileNotFoundError(f'Cannot read: {img_path}')
        grey   = img_bgr[:, :, 0]                          # (64, 512) uint8
        pil_img = Image.fromarray(grey, 'L')
        tensor  = (self.transform(pil_img)
                   .unsqueeze(0)
                   .repeat(1, 3, 1, 1)
                   .to(self.device))
        emb  = self.model(tensor).squeeze(0).cpu().numpy() # (512,)
        norm = np.linalg.norm(emb)
        return (emb / (norm + 1e-12)).astype(np.float32)


class HDBIFExtractor:
    """
    Extracts HDBIF binary ICA codes from 64×512 uint8 polar images and
    computes Hamming-based similarity scores.

    Score convention: 0.5 - hamming_distance  (higher = more genuine).
    """

    MATRIX_SCORING = False  # requires pairwise loop, no matmul

    def __init__(self, filter_dir: str, filter_size: int = 17,
                 num_filters: int = 5, max_shift: int = 16,
                 score_norm: bool = False, device=None):
        self.coder = HDBIFCoder(
            filter_dir=filter_dir,
            filter_size=filter_size,
            num_filters=num_filters,
            max_shift=max_shift,
            score_norm=score_norm,
            device=(device if device is not None else
                    torch.device('cuda' if torch.cuda.is_available()
                                 else 'cpu')),
        )

    def extract(self, img_path: Path):
        """Return HDBIF binary code (list of bool ndarrays)."""
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise FileNotFoundError(f'Cannot read: {img_path}')
        grey = img_bgr[:, :, 0]                            # (64, 512) uint8
        return self.coder.extract_code(grey)

    def match(self, codes1, codes2,
              mask1: np.ndarray = None,
              mask2: np.ndarray = None) -> float:
        """Return similarity score (higher = more genuine)."""
        return self.coder.match_pair(codes1, codes2, mask1, mask2)


# ══════════════════════════════════════════════════════════════════════════════
#  Feature / code loading
# ══════════════════════════════════════════════════════════════════════════════

def load_features(root: Path, extractor, desc: str = ''):
    """
    Walk an ImageFolder-layout directory and extract one feature per image.
    Follows directory-level symlinks (eval split uses them).

    For ArcIrisExtractor  → returns (N×512 float32 ndarray, list[str] labels)
    For HDBIFExtractor    → returns (list[code], list[str] labels)
      where code is a list of bool ndarrays (output of extract_code)
    """
    features, labels = [], []
    classes = sorted(p for p in root.iterdir() if p.is_dir())
    for cls_dir in classes:
        imgs = sorted(cls_dir.resolve().glob('*.png'))
        for img_path in imgs:
            features.append(extractor.extract(img_path))
            labels.append(cls_dir.name)

    n_imgs = len(features)
    n_cls  = len(classes)
    print(f'  {desc:12s}: {n_imgs:4d} images  ({n_cls} classes)')
    if not features:
        raise RuntimeError(f'No PNG images found under {root}')

    if isinstance(extractor, ArcIrisExtractor):
        return np.stack(features), labels      # (N, 512) ndarray
    else:
        return features, labels                # list of codes


# ══════════════════════════════════════════════════════════════════════════════
#  Score matrix computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_score_matrix_arciris(probe_emb, probe_labels,
                                 gallery_emb, gallery_labels):
    """
    All-to-all cosine similarity via matrix multiplication.
    Both embedding matrices must already be L2-normalised.
    Higher score = more similar = genuine.
    """
    scores     = (probe_emb @ gallery_emb.T).astype(np.float32)   # (Np, Ng)
    is_genuine = (np.array(probe_labels)[:, None] ==
                  np.array(gallery_labels)[None, :])               # bool
    return scores, is_genuine


def compute_score_matrix_hdbif(probe_codes, probe_labels,
                                gallery_codes, gallery_labels,
                                extractor: HDBIFExtractor):
    """
    All-to-all pairwise Hamming similarity (0.5 - hamming_distance).
    Higher score = more similar = genuine.
    """
    Np, Ng = len(probe_codes), len(gallery_codes)
    scores     = np.zeros((Np, Ng), dtype=np.float32)
    is_genuine = (np.array(probe_labels)[:, None] ==
                  np.array(gallery_labels)[None, :])

    print(f'  Computing {Np}×{Ng} = {Np*Ng:,} Hamming comparisons …')
    for i, c1 in enumerate(tqdm(probe_codes, desc='  probe', leave=False)):
        for j, c2 in enumerate(gallery_codes):
            scores[i, j] = extractor.match(c1, c2)

    return scores, is_genuine


def compute_scores(probe_feat, probe_labels,
                   gallery_feat, gallery_labels,
                   extractor):
    """Dispatch to the correct score-matrix function."""
    if isinstance(extractor, ArcIrisExtractor):
        return compute_score_matrix_arciris(
            probe_feat, probe_labels, gallery_feat, gallery_labels)
    else:
        return compute_score_matrix_hdbif(
            probe_feat, probe_labels, gallery_feat, gallery_labels, extractor)


# ══════════════════════════════════════════════════════════════════════════════
#  Score fusion
# ══════════════════════════════════════════════════════════════════════════════

def apply_score_fusion(scores, is_genuine, probe_labels, gallery_labels,
                       mode='max'):
    """
    Fuse per-image gallery scores into per-identity scores.
    mode = 'max'  (Anderson et al.) or 'mean'.

    Returns
    -------
    fused_scores  : (N_probe, N_gallery_classes) float32
    fused_genuine : (N_probe, N_gallery_classes) bool
    """
    gallery_classes = sorted(set(gallery_labels))
    cls_idx = {c: [j for j, g in enumerate(gallery_labels) if g == c]
               for c in gallery_classes}
    agg = np.max if mode == 'max' else np.mean

    fused_scores  = np.stack(
        [agg(scores[:, cls_idx[c]], axis=1) for c in gallery_classes],
        axis=1)
    fused_genuine = (np.array(probe_labels)[:, None] ==
                     np.array(gallery_classes)[None, :])
    return fused_scores.astype(np.float32), fused_genuine


# ══════════════════════════════════════════════════════════════════════════════
#  Metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(scores_flat: np.ndarray, genuine_flat: np.ndarray,
                    label: str):
    """
    EER, TAR@FAR=0.1%, TAR@FAR=0.01%, Decidability Index.

    Both ArcIris (cosine similarity) and HDBIF (0.5 - Hamming) use
    higher-score = more-genuine convention, so roc_curve is called the same way.
    """
    genuine_scores  = scores_flat[genuine_flat]
    impostor_scores = scores_flat[~genuine_flat]

    # Filter out failed HDBIF comparisons (sentinel score == -999.0).
    # Threshold at -900 so legitimate ArcIris cosine scores in [-1, 1]
    # (including values near -1) are NOT dropped — only the failure sentinel.
    genuine_scores  = genuine_scores[genuine_scores > -900]
    impostor_scores = impostor_scores[impostor_scores > -900]

    if len(genuine_scores) == 0 or len(impostor_scores) == 0:
        print(f'WARNING: no valid scores for "{label}" — skipping')
        return None, None, None

    # Rebuild flat arrays after filtering
    valid_mask = scores_flat > -900
    s_flat = scores_flat[valid_mask]
    g_flat = genuine_flat[valid_mask]

    fpr, tpr, _ = roc_curve(g_flat.astype(int), s_flat)
    fnr = 1.0 - tpr

    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer     = float((fpr[eer_idx] + fnr[eer_idx]) / 2 * 100)

    def tar_at_far(target_far):
        idx = np.searchsorted(fpr, target_far, side='right') - 1
        return float(tpr[max(0, min(idx, len(tpr) - 1))] * 100)

    # Decidability Index: (µ_impostor − µ_genuine) / √(0.5*(σ²_gen + σ²_imp))
    # Note: for higher-is-genuine convention, genuine_mean > impostor_mean,
    # so we keep the sign consistent with the definition in the handoff doc
    # which uses (µ_impostor − µ_genuine) and expects a positive DI when
    # the distributions separate.  We use abs to always give positive DI.
    num = abs(genuine_scores.mean() - impostor_scores.mean())
    den = math.sqrt(0.5 * (genuine_scores.std() ** 2 +
                            impostor_scores.std() ** 2)) + 1e-12
    di  = float(num / den)

    result = {
        'label':          label,
        'n_genuine':      int(genuine_flat.sum()),
        'n_impostor':     int((~genuine_flat).sum()),
        'EER_%':          round(eer, 3),
        'TAR@FAR=0.1%':   round(tar_at_far(0.001), 3),
        'TAR@FAR=0.01%':  round(tar_at_far(0.0001), 3),
        'DI':             round(di, 4),
        'genuine_mean':   round(float(genuine_scores.mean()), 4),
        'genuine_std':    round(float(genuine_scores.std()), 4),
        'impostor_mean':  round(float(impostor_scores.mean()), 4),
        'impostor_std':   round(float(impostor_scores.std()), 4),
    }

    print(f'\n{"="*62}\n  {label}\n{"="*62}')
    for k, v in result.items():
        if k not in ('label', 'n_genuine', 'n_impostor'):
            print(f'  {k:25s}: {v}')
    print(f'  {"pairs":25s}: {result["n_genuine"]} genuine  '
          f'{result["n_impostor"]} impostor')

    return result, fpr, tpr


# ══════════════════════════════════════════════════════════════════════════════
#  Plotting + summary
# ══════════════════════════════════════════════════════════════════════════════

def plot_roc(curves, out_path: Path, title: str = ''):
    fig, ax = plt.subplots(figsize=(7, 6))
    for fpr, tpr, lbl in curves:
        ax.plot(fpr * 100, tpr * 100, label=lbl, linewidth=1.5)
    ax.set_xlabel('FAR (%)')
    ax.set_ylabel('TAR (%)')
    ax.set_title(title or 'ROC — PolyU Cross-Spectral Iris (subjects 168–209)')
    ax.set_xscale('log')
    ax.set_xlim([1e-2, 100])
    ax.set_ylim([0, 100])
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150)
    print(f'ROC saved → {out_path}')
    plt.close(fig)


def print_summary_table(all_results, matcher_name: str):
    print('\n' + '─' * 76)
    print(f'  Matcher: {matcher_name}')
    print('─' * 76)
    print(f'  {"Condition":<42} {"EER%":>6}  {"TAR@0.1%":>9}  '
          f'{"TAR@0.01%":>10}  {"DI":>6}')
    print('─' * 76)
    for r in all_results:
        if r is None:
            continue
        print(f'  {r["label"]:<42} {r["EER_%"]:>6.3f}  '
              f'{r["TAR@FAR=0.1%"]:>9.3f}  '
              f'{r["TAR@FAR=0.01%"]:>10.3f}  '
              f'{r["DI"]:>6.4f}')
    print('─' * 76)
    print(f'\n  [Reference] IP-GAN score fusion — Anderson et al. 2024  EER = 5.500%')


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description='Cross-spectral iris evaluation (ArcIris or HDBIF)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── matcher selection ────────────────────────────────────────────────────
    ap.add_argument(
        '--matcher', default='arciris',
        choices=['arciris', 'hdbif', 'osiris'],
        help='Iris matcher to use  [default: arciris]')

    # ── data paths ───────────────────────────────────────────────────────────
    ap.add_argument('--gallery',  required=True,
                    help='Real NIR gallery dir  (ImageFolder, inst 1-10)')
    ap.add_argument('--probe',    required=True,
                    help='Fake NIR probe dir    (GAN output,  inst 11-15)')
    ap.add_argument('--baseline', default=None,
                    help='Real VIS probe dir    (no-GAN baseline, optional)')
    ap.add_argument('--out_dir',  required=True,
                    help='Output directory for results JSON + ROC PNG')

    # ── ArcIris options ──────────────────────────────────────────────────────
    ap.add_argument('--model',
                    default='./models-ArcIris/ResNet100_154000.pt',
                    help='ArcIris iresnet100 checkpoint  '
                         '[default: ./models-ArcIris/ResNet100_154000.pt]')

    # ── HDBIF options ────────────────────────────────────────────────────────
    ap.add_argument('--filter_dir',
                    default='./filters_pt/finetuned_bsif_eyetracker_data',
                    help='HDBIF ICA filter directory  '
                         '[default: ./filters_pt/finetuned_bsif_eyetracker_data]')
    ap.add_argument('--filter_size',  type=int, default=17,
                    help='HDBIF filter size (default 17)')
    ap.add_argument('--num_filters',  type=int, default=5,
                    help='HDBIF number of filters (default 5)')
    ap.add_argument('--max_shift',    type=int, default=16,
                    help='HDBIF max cyclic column shift (default 16)')

    # ── scoring ──────────────────────────────────────────────────────────────
    ap.add_argument('--fusion', default='both',
                    choices=['max', 'mean', 'both'],
                    help='Score fusion mode  [default: both]')

    args = ap.parse_args()

    # ── OSIRIS guard ─────────────────────────────────────────────────────────
    if args.matcher == 'osiris':
        print('ERROR: OSIRIS matcher is not yet implemented.')
        print('  OSIRIS uses log-Gabor IrisCode and is not included in '
              'OpenSourceIrisRecognition.')
        print('  See: https://svn.irisa.fr/osiris/')
        sys.exit(1)

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Build extractor ──────────────────────────────────────────────────────
    if args.matcher == 'arciris':
        extractor = ArcIrisExtractor(
            model_path=args.model,
            device=device,
        )
    else:  # hdbif
        extractor = HDBIFExtractor(
            filter_dir=args.filter_dir,
            filter_size=args.filter_size,
            num_filters=args.num_filters,
            max_shift=args.max_shift,
            device=device,
        )

    fusion_modes = ['max', 'mean'] if args.fusion == 'both' else [args.fusion]
    all_results  = []
    roc_curves   = []

    # ── Gallery (real NIR) ───────────────────────────────────────────────────
    print(f'\n[1] Extracting gallery features (real NIR) — matcher: {args.matcher}')
    gal_feat, gal_lbl = load_features(
        Path(args.gallery).expanduser(), extractor, desc='gallery')

    # ── GAN probe (fake NIR) ─────────────────────────────────────────────────
    print('\n[2] Extracting probe features (GAN fake NIR)…')
    prb_feat, prb_lbl = load_features(
        Path(args.probe).expanduser(), extractor, desc='probe')

    scores, genuine = compute_scores(prb_feat, prb_lbl, gal_feat, gal_lbl,
                                     extractor)

    # All-to-all
    res, fpr, tpr = compute_metrics(
        scores.ravel(), genuine.ravel(), f'GAN  all-to-all [{args.matcher}]')
    if res:
        all_results.append(res)
        roc_curves.append((fpr, tpr,
                           f'GAN a2a  EER={res["EER_%"]:.2f}%'))

    # Score fusion
    for mode in fusion_modes:
        fused, fused_gen = apply_score_fusion(
            scores, genuine, prb_lbl, gal_lbl, mode)
        res_f, fpr_f, tpr_f = compute_metrics(
            fused.ravel(), fused_gen.ravel(),
            f'GAN  score fusion ({mode}) [{args.matcher}]')
        if res_f:
            all_results.append(res_f)
            roc_curves.append((fpr_f, tpr_f,
                               f'GAN fusion ({mode})  EER={res_f["EER_%"]:.2f}%'))

    # ── No-GAN baseline (real VIS vs real NIR) ───────────────────────────────
    if args.baseline:
        print('\n[3] Extracting baseline features (real VIS, no GAN)…')
        bas_feat, bas_lbl = load_features(
            Path(args.baseline).expanduser(), extractor, desc='baseline')
        b_scores, b_genuine = compute_scores(
            bas_feat, bas_lbl, gal_feat, gal_lbl, extractor)

        res_b, fpr_b, tpr_b = compute_metrics(
            b_scores.ravel(), b_genuine.ravel(),
            f'Baseline  NIR vs VIS  all-to-all [{args.matcher}]')
        if res_b:
            all_results.append(res_b)
            roc_curves.append((fpr_b, tpr_b,
                               f'Baseline a2a  EER={res_b["EER_%"]:.2f}%'))

        for mode in fusion_modes:
            fused_b, fused_b_gen = apply_score_fusion(
                b_scores, b_genuine, bas_lbl, gal_lbl, mode)
            res_bf, fpr_bf, tpr_bf = compute_metrics(
                fused_b.ravel(), fused_b_gen.ravel(),
                f'Baseline  score fusion ({mode}) [{args.matcher}]')
            if res_bf:
                all_results.append(res_bf)
                roc_curves.append((fpr_bf, tpr_bf,
                                   f'Baseline fusion ({mode})  '
                                   f'EER={res_bf["EER_%"]:.2f}%'))

    # ── Save outputs ─────────────────────────────────────────────────────────
    print_summary_table(all_results, matcher_name=args.matcher)
    plot_roc(roc_curves, out_dir / 'roc_curve.png',
             title=f'ROC — PolyU Cross-Spectral Iris (subjects 168–209)  '
                   f'[{args.matcher}]')

    results_path = out_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump([r for r in all_results if r], f, indent=2)
    print(f'\nResults → {results_path}')


if __name__ == '__main__':
    main()
