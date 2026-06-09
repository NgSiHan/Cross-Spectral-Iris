HDBIF ICA texture filters — finetuned_bsif_eyetracker_data variant
===================================================================

Source: https://notredame.box.com/s/vxwwtm68th0nvdag6dhqn7u64kf2i2r0

These filters were fine-tuned on eye-tracker / iris-scanner captures and are
the best match for PolyU-style iris scanner data.  The other two variants in
the original download (finetuned_bsif_random_iris_patches,
finetuned_bsif_user_annotations) are NOT copied here — they are less relevant
for near-IR / visible iris scanner evaluation.

Contents: 96 filter files, one per (filter_size × num_filters) combination
  Filter sizes : 5, 7, 9, 11, 13, 15, 17, 19, 21, 27, 33, 39
  Num filters  : 5, 6, 7, 8, 9, 10, 11, 12

HDBIF default (cfg.yaml): filter_size=17, num_filters=5
  → ICAtextureFilters_17x17_5bit.pt   (used by --filter_size 17 --num_filters 5)

To use a different combination, pass the corresponding flags to 05_evaluate.py:
  --filter_size 17 --num_filters 5    (default, from HDBIF cfg.yaml)
  --filter_size 13 --num_filters 8    (example alternative)
