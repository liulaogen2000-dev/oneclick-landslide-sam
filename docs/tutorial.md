# Tutorial

1. Create the Python 3.11 environment and install the pinned CPU requirements using the README commands. Keep the repository and the data package as sibling directories.
2. Run `scripts/verify_package.py`. Resolve a checksum failure before reading results.
3. Run `scripts/reproduce.py all` into a new output directory. Read every row of `comparison.csv`; its manuscript location identifies the table row or prose claim. A failed row stops the run.
4. For a new image, install the optional GPU requirements and obtain SAM 2.1 Hiera-Large from its official release. Use the original dataset's GSD and matching 13- or 14-feature selector.
5. Put integer image coordinates in a JSON list of `[x,y]` pairs. Run the scene or tile example in the README. Inputs at deployment are the RGB image, clicks, GSD where required, frozen SAM and a trained selector. No reference object or bounding box is read.
6. Inspect the object masks and confidence JSON. For box correction, call `landslide_sam.correction.boxed_indices` with predicted confidences or predicted areas. Use one budget pool per region/event on the scene datasets and CAS, and one pool for the full Jiuzhaigou or Bijie test set. The mixed annotation study ranks each person's 88 objects together. Reference masks never determine that ranking.
7. If masks and references are available for evaluation, compute full queried-object IoU and the whole-image union separately. Retain all reference foreground for the whole-image denominator. Precision/Recall and k90 are in `evaluation.py` and `correction.py`; do not replace full-object IoU with ROI IoU.
8. To refit a selector, run the GBT command or a GPU variant command. They use original training IDs and selected weights. Evaluate the saved model only after fitting; the test set is not a validation set.

Frozen-result reproduction requires no GPU or original raster redistribution. Running SAM on newly downloaded images does require the source datasets and the GPU stack. Peak GPU/RAM requirements and GPU validation in the current release environment are **MISSING** unless explicitly recorded in the release verification report.
