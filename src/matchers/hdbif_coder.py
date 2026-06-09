"""
Lightweight HDBIF iris-code extractor and matcher.

Adapted from:
  OpenSourceIrisRecognition/methods/HDBIF/Python/modules/irisRecognition.py

Only the code-extraction and Hamming-distance matching logic is kept here.
The segmentation / circle-fitting models are NOT loaded — those were used
during preprocessing (02_preprocess.py) and are not needed for evaluation
on already-normalised 64×512 polar codes.

Filter files
------------
HDBIF requires pre-trained ICA texture filter files (.pt) that are NOT
bundled in this repo because of their size.  Download them from:

    https://notredame.box.com/s/vxwwtm68th0nvdag6dhqn7u64kf2i2r0

and place them in:

    <repo_root>/filters_pt/finetuned_bsif_eyetracker_data/

The expected filename format is:
    ICAtextureFilters_{filter_size}x{filter_size}_{num_filters}bit.pt

Default settings (matching HDBIF cfg.yaml):
    filter_size  = 17
    num_filters  = 5
    max_shift    = 16

Score convention
----------------
``match_pair`` returns a *similarity* score (higher = more similar = genuine):

    score = 0.5 - hamming_distance

so genuine pairs cluster near +0.4 and impostor pairs cluster near 0.0.
This is consistent with the cosine-similarity convention used by ArcIris,
making the two matchers directly comparable in the same evaluation script.

Masks
-----
HDBIF matching ideally uses a polar occlusion mask to exclude eyelid/eyelash
pixels.  Our preprocessing pipeline does not save polar masks (only the polar
image).  When ``mask`` is None, an all-ones mask is used (all pixels treated
as valid iris).  This is a slight over-estimation of performance but is the
correct fallback when masks are unavailable.
"""

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# Average valid bits per filter size (from HDBIF irisRecognition.py).
# Used when score_norm=True.
_AVG_BITS_BY_FILTER_SIZE = {
    5: 25056, 7: 24463, 9: 23764, 11: 23010,
    13: 22225, 15: 21420, 17: 20603, 19: 19777,
    21: 18945, 27: 16419, 33: 13864, 39: 11289,
}


class HDBIFCoder:
    """
    Extracts binary ICA iris codes from 64×512 uint8 polar images and
    computes normalised Hamming distances between code pairs.

    Parameters
    ----------
    filter_dir   : path to the directory containing ICAtextureFilters_*.pt
    filter_size  : convolution kernel size (default 17, matches HDBIF cfg.yaml)
    num_filters  : number of ICA filters (default 5, matches HDBIF cfg.yaml)
    max_shift    : maximum cyclic column shift for rotational tolerance (default 16)
    score_norm   : apply Doddington-style score normalisation (default False)
    device       : torch device; None → auto-select CUDA if available
    """

    def __init__(self,
                 filter_dir: str,
                 filter_size: int = 17,
                 num_filters: int = 5,
                 max_shift: int = 16,
                 score_norm: bool = False,
                 device=None):

        self.device     = (device if device is not None else
                           torch.device('cuda' if torch.cuda.is_available()
                                        else 'cpu'))
        self.filter_sizes         = [filter_size]
        self.num_filters_per_size = [num_filters]
        self.total_num_filters    = num_filters
        self.max_shift  = max_shift
        self.score_norm = score_norm

        avg_bits = _AVG_BITS_BY_FILTER_SIZE.get(filter_size, 20603)
        self.avg_num_bits = float(avg_bits)

        filter_dir_path = Path(filter_dir).expanduser().resolve()
        if not filter_dir_path.exists():
            raise FileNotFoundError(
                f"HDBIF filter directory not found: {filter_dir_path}\n"
                "Download the filters from:\n"
                "  https://notredame.box.com/s/vxwwtm68th0nvdag6dhqn7u64kf2i2r0\n"
                "and place them in that directory."
            )

        self.torch_filters = self._load_filters(
            str(filter_dir_path) + '/',
            self.filter_sizes,
            self.num_filters_per_size,
        )
        print(f'HDBIFCoder: loaded {num_filters} filter(s) of size '
              f'{filter_size}×{filter_size}  (device: {self.device})')

    # ------------------------------------------------------------------
    # Copied verbatim from HDBIF irisRecognition.load_filters
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _load_filters(self, recog_bsif_dir, filter_sizes, num_filters_per_size):
        torch_filters = []
        for filter_size, num_filters in zip(filter_sizes, num_filters_per_size):
            mat_file_path = (recog_bsif_dir +
                             f'ICAtextureFilters_{filter_size}x{filter_size}'
                             f'_{num_filters}bit.pt')
            if not Path(mat_file_path).exists():
                raise FileNotFoundError(
                    f"Missing HDBIF filter file: {mat_file_path}\n"
                    "Download from https://notredame.box.com/s/vxwwtm68th0nvdag6dhqn7u64kf2i2r0"
                )
            filter_mat = (torch.jit.load(mat_file_path, torch.device('cpu'))
                          .ICAtextureFilters.detach().numpy())
            torch_filter = torch.FloatTensor(filter_mat).to(self.device)
            torch_filter = (torch.moveaxis(torch_filter.unsqueeze(0), 3, 0)
                            .detach().requires_grad_(False))
            torch_filters.append(torch_filter.clone().detach())
        return torch_filters

    # ------------------------------------------------------------------
    # Copied verbatim from HDBIF irisRecognition.extractCode
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def extract_code(self, polar: np.ndarray):
        """
        Extract binary ICA codes from a 64×512 uint8 polar image.

        Parameters
        ----------
        polar : (64, 512) uint8 numpy array

        Returns
        -------
        list of (num_filters, code_height, 512) bool ndarrays
            code_height = 64 - filter_size + 1  (48 for filter_size=17)
        """
        if polar is None:
            return None
        codeBinaries = []
        for filter_size, torch_filter in zip(self.filter_sizes,
                                             self.torch_filters):
            r = int(np.floor(filter_size / 2))
            polar_t = (torch.tensor(polar).float()
                       .unsqueeze(0).unsqueeze(0).to(self.device))
            padded_polar = nn.functional.pad(
                polar_t, (r, r, 0, 0), mode='circular')
            codeContinuous = nn.functional.conv2d(padded_polar, torch_filter)
            codeBinary = torch.where(
                codeContinuous.squeeze(0) > 0, True, False)
            codeBinaries.append(codeBinary.cpu().numpy())
        return codeBinaries

    # ------------------------------------------------------------------
    # Adapted from HDBIF irisRecognition.matchCodesEfficient
    # Returns a *similarity* score (higher → more similar → genuine)
    # ------------------------------------------------------------------
    def match_pair(self,
                   codes1, codes2,
                   mask1: np.ndarray = None,
                   mask2: np.ndarray = None) -> float:
        """
        Compute a similarity score between two iris code lists.

        Parameters
        ----------
        codes1, codes2 : outputs of extract_code()
        mask1, mask2   : (64, 512) uint8 polar occlusion masks (255=valid).
                         Pass None to use an all-ones mask (all pixels valid).

        Returns
        -------
        float
            ``0.5 - best_hamming_distance``, in approximately [-0.5, 0.5].
            Higher values indicate a better (more genuine) match.
            Returns -999.0 if too few bits can be compared.
        """
        # Fall back to all-ones mask when not available
        code_h = codes1[0].shape[1]  # e.g. 48 for filter_size=17
        if mask1 is None:
            mask1 = np.full((64, 512), 255, dtype=np.uint8)
        if mask2 is None:
            mask2 = np.full((64, 512), 255, dtype=np.uint8)

        # Precompute cropped binary masks (matching code_h)
        precomputed_masks = []
        for code1 in codes1:
            num_filters, code_size, _ = code1.shape
            r = int((mask1.shape[0] - code_size) / 2)
            m1_bin = mask1[r:-r, :] > 127.5
            m2_bin = mask2[r:-r, :] > 127.5
            precomputed_masks.append((num_filters, m1_bin, m2_bin))

        def _score_at_shift(xshift):
            sumXor = 0
            sumBits = 0
            tot_filters = 0
            for (c1, c2), (nf, m1b, m2b) in zip(
                    zip(codes1, codes2), precomputed_masks):
                andM = np.logical_and(m1b, np.roll(m2b, xshift, axis=1))
                s_and = np.sum(andM)
                if s_and == 0:
                    return float('inf')
                xorC = np.logical_xor(c1, np.roll(c2, xshift, axis=2))
                xorM = np.logical_and(
                    xorC, np.tile(np.expand_dims(andM, 0), (nf, 1, 1)))
                sumXor   += np.sum(xorM)
                sumBits  += s_and * nf
                tot_filters += nf
            if sumBits == 0:
                return float('inf')
            s = sumXor / sumBits
            if self.score_norm:
                s = 0.5 - (0.5 - s) * math.sqrt(
                    sumBits / (self.avg_num_bits * tot_filters))
            return s

        # Even shifts first (matchCodesEfficient approach)
        best_score = float('inf')
        best_shift = 0
        for xshift in range(-self.max_shift, self.max_shift + 1, 2):
            s = _score_at_shift(xshift)
            if s < best_score:
                best_score = s
                best_shift = xshift

        if best_score == float('inf'):
            return -999.0

        # Also check the two odd shifts neighbouring the best even shift
        for xshift in (best_shift - 1, best_shift + 1):
            s = _score_at_shift(xshift)
            if s < best_score:
                best_score = s

        # Convert to similarity score (higher = more genuine)
        return float(0.5 - best_score)
