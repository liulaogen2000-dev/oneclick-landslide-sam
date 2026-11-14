# How good is one click?

Code and frozen intermediate results for **How good is one click? Realistic evaluation and learned mask selection for landslide delineation with the Segment Anything Model**. The selector learns from previous inventories and chooses a mask for a located landslide from one positive click. SAM 2.1 Hiera-Large stays frozen. Reference masks enter training targets and evaluation, not candidate generation or deployment.

Code repository: [oneclick-landslide-sam](https://github.com/liulaogen2000-dev/oneclick-landslide-sam). Companion data DOI: [10.5281/zenodo.24214459](https://doi.org/10.5281/zenodo.24214459). Paper journal and paper DOI: **MISSING**. The manuscript, embedded figures, source imagery and source label rasters are not included.

## Install

Use Python 3.11. The recorded experimental version is 3.11.13. From this repository:

```bash
python -m venv ../verification/venv
../verification/venv/bin/python -m pip install -r requirements.txt
```

The CPU requirements are pinned to the recorded environment. Do not replace unavailable versions silently: record the installation failure. NumPy and SciPy perform the statistics; scikit-learn and joblib load the frozen GBT models; Pillow reads RGB images. Optional GPU work also requires PyTorch, torchvision, sam2, Hydra and rasterio:

```bash
../verification/venv/bin/python -m pip install -r requirements-gpu.txt
```

The paper's experiments used an NVIDIA GeForce RTX 4090 with 24 GB of GPU memory, PyTorch 2.8.0, CUDA 12.8 and four CPU threads. CPU reproduction of the tables and reported numerical results from frozen intermediate results has been verified; see the accompanying `VERIFICATION.md`. The SAM inference examples and neural selector variants require a CUDA GPU. These GPU paths use the paper's experimental code and were not rerun in a new environment for this release.

## Reproduce the main results

Download `oneclick_landslide_sam_data.zip` from [Zenodo DOI 10.5281/zenodo.24214459](https://doi.org/10.5281/zenodo.24214459), and extract its contents into `../zenodo/` beside this repository. The authors' derived data and anonymous annotation records are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); source datasets and SAM-derived weights retain their own terms. All paths below are relative. Check the payload first:

```bash
../verification/venv/bin/python scripts/verify_package.py --data ../zenodo
../verification/venv/bin/python scripts/reproduce.py all --data ../zenodo --output ../verification/results
```

Each output directory must be new. `comparison.csv` lists the manuscript location, recomputed value, printed value, difference, unit, frozen source and PASS/FAIL. A mismatch stops execution; the code does not change a manuscript value or a threshold to pass.

| Manuscript result | Command target | Input |
| --- | --- | --- |
| Table 1 query-object counts | `table2` | Frozen evaluation counts, compared with Table 1 |
| Table 2 and Supplementary Table S3.10 | `table2` | TP, predicted/reference area counts and sums of queried-object IoU |
| Table 3 and its beta intervals in Sect. S3.8 / Table S3.9 | `table3` | Per-object areas and the frozen reference cutoff |
| Abstract gains, external IoU, window/box inflation and time savings | `abstract` | Evaluation counts, window-control counts, paired box results and anonymous timings |

For one target, substitute `table2`, `table3` or `abstract` for `all` and use a fresh output directory. The abstract target also computes Table 2 because it needs the same ratios. IoU differences must be within 0.1 percentage points, counts must match exactly, and other quantities must match the manuscript at its printed precision. The 37–48% and 17–28% time ranges use ratios of means on each annotator's same 88 objects, not ratios of medians. At k=20%, the actual count is floor(88×0.2)=17.

To cross-check existing downloaded source rasters against the frozen references, arrange the five datasets under the relative paths listed in the package's `DATASET_SOURCES.csv` and run:

```bash
../verification/venv/bin/python scripts/verify_public_data.py --data ../zenodo --datasets ../../landslide --output ../verification/public_data_checks.csv
../verification/venv/bin/python scripts/verify_models.py --data ../zenodo --output ../verification/model_checks.csv
```

The first command reads source image dimensions and label pixels, checks all foreground counts and the exact areas of frozen query components, and writes no rasters. It does not choose a new query list. The second checks every GBT's frozen parameters and candidate choices.

The final manuscript uses paired percentile bootstrap with **10,000 resamples, seed 0**, for S3.10, the beta intervals and directly quoted numeric intervals. Its Fig. 3c budget curves and k90 explicitly use **1,000**. The original `final_v1` tables retain the earlier 1,000-resample intervals. This release keeps model fitting unchanged and recomputes the published intervals at the declared count. Five *training* seeds and five *bootstrap* seeds are different experiments.

## Two candidate modes

| Dataset/mode | Window | Candidates per click | Features |
| --- | --- | --- | --- |
| GVLM v1.0 | Four click-centred windows, sides 512/1024/2048/4096 pixels at 0.59 m | 12 | 14 |
| GVLM v1.1 | The same ground sides, converted using native x/y pixel sizes | 12 | 14 |
| CAS | Whole RGB tile, externally resized as in the frozen CAS path | 3 | 13 |
| Jiuzhaigou (`jz`), Bijie | Whole tile; original FP32 SAM input path | 3 | 13 |

Use the scene modes for large scenes and the corresponding trained 14-feature model. Use the dataset-specific tile modes with a 13-feature model. Do not pad features or add scales to tile models. Jiuzhaigou area features remain in native pixels, as in the frozen code; Bijie uses its recorded 0.8 m scale. SAM scores and stability, shape, click position, ring contrast, scale and cross-scale agreement follow the original arithmetic. `_frozen_features.py` preserves the legacy uint8 compactness arithmetic; changing it would change the trained model's inputs.

GBT receives its original raw feature matrix, including NaNs supported by HistGradientBoostingRegressor. Median filling and standardisation are fitted on training folds for the **hybrid network**. The code is authoritative where historical prose describes preprocessing more broadly.

## Minimal image-and-click example

For a CPU check using the two packaged frozen clicks and public RGB images:

```bash
../verification/venv/bin/python examples/cached_demo.py --data ../zenodo --datasets ../../landslide --dataset GVLM_v1_0 --output ../verification/cached_scene
../verification/venv/bin/python examples/cached_demo.py --data ../zenodo --datasets ../../landslide --dataset CAS --output ../verification/cached_tile
```

These commands read a public image and a packaged click, reuse its frozen candidate masks/SAM scores, recompute all 14 or 13 features, and output a predicted mask plus confidence. They check exact feature parity and frozen candidate choice. They do **not** run SAM or accept new clicks; the live example below does.

Download the official [SAM 2.1 Hiera-Large checkpoint](https://github.com/facebookresearch/sam2) separately. It is not in the repository or this data package. Create `clicks.json` containing integer global pixel centres, for example `[[400, 300], [700, 520]]`, in `[x,y]` order. Replace the illustrative image and click coordinates with a downloaded image and its real clicks.

Scene mode:

```bash
../verification/venv/bin/python examples/delineate.py --image data/scene.png --clicks clicks.json --dataset GVLM_v1_1 --gsd 1.0 1.0 --checkpoint checkpoints/sam2.1_hiera_large.pt --model ../zenodo/models/GVLM_v1_1/gbt_fold_00.joblib --output outputs/scene
```

`--gsd` must be the image's native pixel sizes, not the illustrative values above. GVLM v1.0 uses the frozen 0.59 m scale automatically.

Tile mode:

```bash
../verification/venv/bin/python examples/delineate.py --image data/tile.png --clicks clicks.json --dataset bijie --checkpoint checkpoints/sam2.1_hiera_large.pt --model ../zenodo/models/bijie/gbt_fold_00.joblib --output outputs/tile
```

For CAS, pass the tile's actual `--gsd X Y`. The result contains one binary PNG per click, `union.png` and `confidence.json`. Confidence is the selected candidate's raw predicted IoU, with no clipping or calibration. `models/MODELS.csv` identifies every model and its training source; CAS models are event-held-out fits, not an undocumented fit to all CAS events.

## Training and review of selector variants

`scripts/train_selector.py` refits one GBT from the packaged training IDs and the original training-only chosen area weighting. Its seed and parameters come from `configs/frozen_model.json`.

```bash
../verification/venv/bin/python scripts/train_selector.py --data ../zenodo --dataset bijie --fold-index 0 --output outputs/bijie_refit.joblib
```

`experiments/train_variants.py` trains the independent SAM IoU head or hybrid MLP on the same selected outer fold. **Both require a CUDA GPU**. SAM image/prompt encoders and the other decoder parameters are not trained. The head has 132,612 parameters. Hybrid inputs are 270 dimensions for scenes and 269 for tiles. Both use AdamW, batch size 256, 30 epochs and the final epoch, with the original CUDA-graph loss implementation.

```bash
../verification/venv/bin/python experiments/train_variants.py --data ../zenodo --dataset bijie --model head --fold-index 0 --output outputs/head.pt
../verification/venv/bin/python experiments/train_variants.py --data ../zenodo --dataset bijie --model hybrid --fold-index 0 --output outputs/hybrid.pt
```

These commands replay final outer fits with the original selected weighting. `experiments/selection_protocol.py` gives the training-only nested weighting procedure and tie order; full nested retraining additionally needs candidate masks and public labels to form validation unions. Those large full candidate caches are not in this compact frozen-results package. No test labels select weights, epochs or features.

The head comparison in manuscript Sect. 3.2 matches **per-object IoU, P1, training seed 0**; it does not establish that the tree is never worse in whole-image IoU. The hybrid replacement rule instead uses the five-training-seed mean paired difference in **joint P1+P2 whole-image IoU**, and requires a positive 95% lower bound on at least three datasets. No dataset met that condition with either interval edition.

## Obtain the five source datasets

Download imagery and labels from their official sources. They are not redistributed here. Keep the exact dataset versions and the supplied fold IDs. See [docs/datasets.md](docs/datasets.md) for sources, rights and unresolved metadata; a code repository's license does not license its source imagery.

## Layout and citation

`landslide_sam/` contains candidate generation, features, GBT selection, correction and evaluation; `experiments/` contains neural variants and training-only selection; `scripts/` contains reproduction and integrity commands; `examples/` contains the image-and-click interface; `configs/` stores frozen model settings and manuscript targets; `docs/` contains the tutorial and source provenance. Models, feature tensors, tokens and frozen counts belong in the data package and are ignored by Git. Every tracked file must be smaller than 50 MB.

Cite the manuscript title and authors in [CITATION.cff](CITATION.cff). The paper journal and paper DOI remain **MISSING** until publication. The [code repository](https://github.com/liulaogen2000-dev/oneclick-landslide-sam) is MIT licensed to Zhaoyang Gao, Xinyue Mo, Huan Li and Mingshan Feng. The authors' derived data and anonymous annotation records are available at [DOI 10.5281/zenodo.24214459](https://doi.org/10.5281/zenodo.24214459) under CC BY 4.0. SAM and external datasets retain their own terms.
