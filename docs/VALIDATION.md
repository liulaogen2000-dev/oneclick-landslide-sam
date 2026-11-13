# Local validation of release candidate

CPU reproduction passed 326 manuscript comparisons, including Tables 2/3, S3.10/S3.9 intervals and the requested abstract figures. The maximum absolute IoU difference from a printed value was 0.049956033942 percentage points, below 0.1. Counts matched exactly; beta and area ratios matched printed precision.

The data checks covered 25 GBT models with unchanged candidate choices and 1804 evaluated source-image/label units with exact foreground/query-component areas. Cached scene and tile examples matched every feature exactly (14 and 13) and produced predicted masks and confidences.

The fresh Python 3.11.13 environment installed only requirements.txt. The accompanying release's VERIFICATION.md and verification/ contain full commands, timings and comparison rows. One Jiuzhaigou input-column reader error was corrected to preserve the original native-pixel calculation; no missing area was filled and no numerical standard was changed.

GPU inference and neural training were NOT_RUN because no CUDA GPU was available. Peak SAM GPU/host memory: MISSING. Source imagery, labels and full manuscripts are not part of this repository. Unresolved data rights, official Jiuzhaigou archive URL and publication DOI are documented as MISSING.
