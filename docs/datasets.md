# Source datasets and rights

Download the original RGB images and labels from the publishers. This repository and its companion package do not redistribute either. A label-derived scalar, such as reference area or IoU, is a frozen evaluation statistic, not a redistributed label raster.

The authors' derived data and anonymous annotation records are available under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) at [DOI 10.5281/zenodo.24214459](https://doi.org/10.5281/zenodo.24214459), for use with the [MIT-licensed code repository](https://github.com/liulaogen2000-dev/oneclick-landslide-sam). These terms do not replace the source-dataset terms below.

| Dataset | Official source and download instructions | Terms established from the source |
| --- | --- | --- |
| GVLM v1.0 | [Official GVLM repository](https://github.com/zxk688/GVLM): follow its V1.0 Google Drive download; use post-event RGB and the reference map | Research/non-commercial use is stated in the repository README. A separate SPDX data license is **MISSING**. |
| GVLM v1.1 | [Official release README](https://github.com/zxk688/GVLM/blob/main/README_v1_1.md): follow its Google Drive link, extract the HR post-event TIFFs and label TIFFs | The parent repository's non-commercial statement is recorded; a v1.1-specific license grant is **MISSING**. Do not infer a permissive imagery license. |
| CAS | [Official Zenodo record 10294997](https://zenodo.org/records/10294997): download the study's UAV/SAT subsets and read its dataset README; use `img` and binary `mask`, not the inverted display labels | **CC BY-NC 4.0**, from the record's `metadata.license.id=cc-by-nc-4.0` in the [official API](https://zenodo.org/api/records/10294997). The processing-code repository's MIT license is separate. |
| Jiuzhaigou | [Official dataset source](https://github.com/zhangliangdong30/MTGF-SAM/tree/main/datasets/JiuzhaigouDataset); Zhang et al. (2026), [DOI 10.1109/JSTARS.2026.3734278](https://doi.org/10.1109/JSTARS.2026.3734278) | Licence not yet finalized by the providers (see LICENSE_NOTES.md in the source repository). This package contains no Jiuzhaigou imagery, DEM or label rasters, only derived per-object statistics and features. |
| Bijie | [Wuhan University GPCV dataset page](https://gpcv.whu.edu.cn/data/Bijie_pages.html): follow the official ZIP download; use the landslide image/mask pairs | An explicit data license on the official page is **MISSING**. Do not assign a third-party mirror's license to the official archive. |

The paper evaluates only six GVLM v1.1 events; São Sebastião remains excluded. CAS has a frozen sample of 1,500 tiles grouped into five events, not a new random sample on every run. Jiuzhaigou uses the recorded spatial split; Bijie uses the recorded image split. `splits/*.json`, `simulated_clicks.csv` and the source hashes preserve those choices. The main test-object counts are given in the manuscript's Table 1 and verified by the reproduction script.

For frozen-result reproduction, download `oneclick_landslide_sam_data.zip` from the data DOI, extract its contents into `../zenodo/` beside the code and run the README commands. Source rasters are needed for new SAM inference or independent regeneration of candidate masks, not for recomputing TP/PP/GG summaries already frozen here. No dataset is downloaded by the reproduction commands, and no silent sample, split or GSD substitution occurs.

The Jiuzhaigou feature and area-frequency calculations retain native pixel areas from the frozen implementation, even though the final manuscript lists a physical resolution. Its official source and the providers' licence status are recorded above; they do not change the frozen numerical protocol.
