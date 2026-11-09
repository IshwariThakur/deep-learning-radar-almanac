# Land Use and Land Cover (LULC) from Sentinel-1

Deep learning project (Data Analytics 3, SRH University Heidelberg, winter 2026/27) – Ishwari Thakur

A semantic segmentation model that maps land use and land cover from **Sentinel-1 radar only** (VV and VH backscatter),
and a web app that produces **12 monthly LULC maps** for any location, the share of each class per month,
the month-to-month change and the total change from the first to the last month (in percentage points).

---

## 1. Requirements of the project and where they are met

| Requirement | Where |
|---|---|
| Collect data, train a DL model for LULC segmentation with Sentinel-1 as the only input | `01_*`, `01b_*`, `01c_*` (data), `02_*`, `02b_*`, `02d_*`, `02c_*` (training and selection) |
| Independent evaluation on unseen random areas | `01d_*`, `02e_*` |
| Prototype – input: latitude/longitude and start month | `prototype/app.py` (web app), `03_*`, `03b_*` |
| Prototype – processing: 12 monthly LULC maps | `prototype/lulc_core.py` → `fetch_months`, `predict_map` |
| Prototype – calculation: class share per month, month-to-month change and total change in percentage points | `lulc_core.class_shares`, `lulc_core.changes` |
| Prototype – visualisation: LULC map of the location over 12 months | web app (12-map grid, chart, tables), notebook 03 |
| Dataset (download link) | Zenodo, <https://doi.org/10.5281/zenodo.24091923> (section 4.5) |
| Notebooks with execution results and training performance plots | all `*.ipynb` at the top level (executed) |
| Prototype source code, README, requirements file | `prototype/`, this file, `requirements.txt`, `install.sh` |
| Trained model weights | Zenodo, same record (`lulc_s1_weights.zip`, unzips into `models/round3/`; sections 4.5 and 6) |
| Screen recording of a full run for an urban area | `recording/radar_almanac_demo.mp4` (Mannheim, Germany; plus Spreewald and the Amazon) |

---

## 2. Repository structure

The notebooks are at the top level, next to `data/` and `models/`, because they use paths relative to their own
folder (`data/...`, `models/...`). This is exactly the layout in which they were run.

```
README.md
requirements.txt                          pinned packages (Earth Engine, rasterio, TensorFlow, Keras, Gradio, ...)
install.sh                                setup: requirements.txt + keras-hub (section 3)
01_prepare_dataset.ipynb                  round-1 dataset: 33 areas, export, patches (train / validation / test)
01b_extra_training_areas.ipynb            34 additional training areas (deserts, wetlands, cities, farmland, forest)
01c_round1_labels_current_scheme.ipynb    class-scheme fix of the round-1 patches (section 4.4)
01d_holdout_random_areas.ipynb            30 randomly drawn, unseen areas for the independent evaluation
02_train_unet.ipynb                       round 1: U-Net from scratch vs. U-Net with ResNet50 encoder
02b_round2_final.ipynb                    round 2: four variants on the extended data
02d_round3_final.ipynb                    round 3 (final): six variants, selection on the validation set
02c_tta_and_ensemble.ipynb                flips/rotations and ensemble, chosen on validation; final setting
02e_holdout_evaluation.ipynb              frozen final model on the 30 random areas, 95 % ranges, experiment record
03_case_study_mannheim.ipynb              case study Mannheim 2021: 12 maps, changes, Dynamic World and WorldCover,
                                          persistent predicted transitions
03b_world_check.ipynb                     prototype on 17 requests worldwide (incl. data gaps, sea, polar, 180°)
04_web_app.ipynb                          starts the web app on the JupyterHub (temporary gradio.live link)
prototype/
  lulc_core.py                            all prototype functions (download, prediction, statistics, checks)
  app.py                                  web app (Gradio): input checks, one complete request, page layout
  ui.py                                   look of the web app: theme, location map, result plates (no calculations)
  world_check.py                          runs the app code for the 17 requests
models/round3/                            final model, normalisation, metadata, training histories
holdout/evaluation_v4_holdout/            results of 02e (tables, figures, frozen-model fingerprint)
data/                                     not included – see section 4.5 (download link)
recording/radar_almanac_demo.mp4          screen recording (5:47): Mannheim (urban, with Dynamic World), Spreewald
                                          (place search), Amazon forest edge (typed coordinates, with Dynamic World)
```

---

## 3. Setup

### 3.1 Accounts

The project downloads all satellite data from **Google Earth Engine**. You need

1. a Google account,
2. a Google Cloud project registered for **non-commercial** Earth Engine use
   (<https://earthengine.google.com/noncommercial/>), and its **project ID**.

The project ID is set in the notebooks (`PROJECT_ID = "..."`) and in the app (`--project` or the
environment variable `EE_PROJECT`). Replace it with your own project ID.

On first use, Earth Engine prints a link: open it, log in, and paste the code back.
The token is stored on the machine, so the login is only asked again when the token is missing.
No other accounts or API keys are needed (the pretrained weights for training are public downloads).

### 3.2 Python environment

Training and evaluation: Python 3.11, TensorFlow 2.20.0, Keras 3.11.2, keras-hub 0.24.0 on the university
JupyterHub (NVIDIA RTX A6000); the exact versions are saved in `models/round3/environment.json`.
Setup (one sequence for the notebooks and the prototype; tested with Python 3.11 in a new virtual environment, CPU,
and on the JupyterHub, GPU):

```bash
python3.11 -m venv .venv && source .venv/bin/activate
bash install.sh
```

`install.sh` installs the pinned `requirements.txt` (TensorFlow 2.20.0, Keras 3.11.2, Gradio 6.31.0, ...), then
keras-hub 0.24.0 (SegFormer) **without dependencies**, then the four small packages keras-hub needs. keras-hub's
package metadata asks for `tensorflow-text`, which would replace TensorFlow 2.20 with another version; SegFormer does
not use it. pip therefore prints "keras-hub 0.24.0 requires tensorflow-text, which is not installed" – this is expected.
On a shared machine (like the JupyterHub) pip may also print conflicts of other, unrelated packages installed there.

The first cells of the notebooks install what is missing on the JupyterHub (the prototype packages and keras-hub),
because the JupyterHub resets user-installed packages.

---

## 4. Data

### 4.1 Sources

| Data | Earth Engine collection | Use |
|---|---|---|
| Sentinel-1 GRD, IW mode, VV + VH, 10 m (calibrated, thermal noise removed, terrain corrected, in dB) | `COPERNICUS/S1_GRD` | **only model input** |
| ESA WorldCover 2021 v200, 10 m | `ESA/WorldCover/v200` | training labels |
| Dynamic World V1 (Sentinel-2 based) | `GOOGLE/DYNAMICWORLD/V1` | **evaluation only** (independent reference in the case study), never used for training or as input |

### 4.2 How the dataset was built

- **Areas:** 15.4 km × 15.4 km squares. Round 1: 33 areas on six continents (21 train, 6 validation, 6 test; a 22nd planned training area,
  Ontario_forest, was dropped because one track covered only 65 % of it).
  Round 2: 34 additional **training** areas; validation and test are unchanged.
  The split is **by area**: all patches of one area belong to one split. Areas of different splits are at least
  50 km apart, no areas overlap, and no area is within 50 km of the case-study city Mannheim.
- **Time:** January, April, July and October 2021 (same year as WorldCover 2021).
  Every month is a separate sample with the same (annual) label. This encourages the model to give the same class in
  every season; real changes within the year (flooding, harvest, construction) are not represented in the labels.
- **One satellite track per area** (orbit direction + relative orbit) for all months: same viewing geometry and the
  same pixel grid. **Monthly median** of all scenes of that track (reduces speckle).
- **Grid:** all files of an area are exported on the original 10 m radar grid (UTM); labels are resampled with
  nearest neighbour. Files are only used when coordinate system, grid, size and band count match the area's `info.json`.
- **Coverage rule:** areas with less than 90 % radar coverage in any month are not used.
- **Patches:** non-overlapping 256 × 256 pixels; patches with missing radar values or less than 90 % labelled pixels are skipped.

| | areas | patches |
|---|---|---|
| train (round 1 + round 2) | 21 + 34 = 55 | 2,920 + 4,618 = 7,538 |
| validation | 6 | 864 |
| test | 6 | 864 |

### 4.3 Classes

WorldCover's 11 source classes are mapped to 7 output classes:

| Output class | WorldCover codes |
|---|---|
| 0 Trees (incl. mangroves) | 10 Tree cover, 95 Mangroves |
| 1 Grassland / shrubland | 30 Grassland, 20 Shrubland, 100 Moss and lichen |
| 2 Cropland | 40 |
| 3 Built-up | 50 |
| 4 Bare / sparse vegetation | 60 |
| 5 Permanent water | 80 |
| 6 Herbaceous wetland | 90 |
| 255 ignored | 70 Snow and ice (out of scope, about 0.01 % of training pixels), no data |

Label 255 is excluded from the loss and from all metrics.

### 4.4 Note on the round-1 patches (notebook 01c)

The round-1 patches on the university JupyterHub had been cut by an earlier version of notebook 01, in which snow
was merged into bare land. Notebook 01c re-cut their labels from the original WorldCover files with the scheme
above and checked that snow pixels are the only difference (26,084 pixels, 0.014 % of the training labels).
A fresh run of notebook 01 produces the current scheme directly and does not need 01c.

### 4.5 Download

The dataset and the trained weights are too large for the repository. Both are published on Zenodo:
**<https://doi.org/10.5281/zenodo.24091923>** (licence CC BY 4.0). The record holds two files:

| Download | Content | Put it here |
|---|---|---|
| **Dataset:** `lulc_s1_dataset.zip` (2.9 GB) | `data/patches_v2/` (round 1: train / validation / test, labels in the class scheme of section 4.3, written by 01c), `data/patches_v3_extra/` (round 2: 34 extra training areas), `holdout/patches_v4_holdout/` (the 30 random areas), each with `*_x.npy`, `*_y.npy`, `*_meta.csv`, `classes.json`; the area lists and checks (`data/config/`, `holdout/config/`) | unzip in the project folder (creates `data/...` and `holdout/...`) |
| **Model weights:** `lulc_s1_weights.zip` (2.0 GB) | the whole `models/round3/` folder: `lulc_model.keras` (standard mode), `segformer_b3/`, `segformer_b2/`, `deeplabv3plus_resnet50/`, `unet_resnet50/` (each `best.keras`, ensemble mode), the two comparison models (`unet_scratch/`, `unet_resnet50_v2_only/`), `normalisation.json`, `inference_choice.json`, `model_info.json`, training histories | unzip in the project folder (creates `models/round3/...`) |

The class definitions are in section 4.3 and in `classes.json` (WorldCover code → class, ignore value 255).

Format: `*_x.npy` float16, shape (N, 256, 256, 2) = VV, VH in dB; `*_y.npy` uint8, shape (N, 256, 256) = class 0–6 or 255;
`*_meta.csv` = area, continent, month, row, column of every patch.

---

## 5. How to run

All notebooks are run from the top-level folder (open them there in Jupyter).

| Step | Notebook | Earth Engine login | Hardware | Measured runtime (ours) |
|---|---|---|---|---|
| 1 | `01_prepare_dataset` | yes | CPU | not measured (mostly Earth Engine downloads) |
| 2 | `01b_extra_training_areas` | yes | CPU | not measured (mostly Earth Engine downloads) |
| 3 | `01c_round1_labels_current_scheme` | no | CPU | a few minutes |
| 4 | `02d_round3_final` | no | GPU | training 118 min for all six variants (per variant 9–28 min, printed in the notebook), RTX A6000 |
| 5 | `02c_tta_and_ensemble` | no | GPU | validation comparison 2.4 min, plus the test evaluation |
| 6 | `01d_holdout_random_areas` | yes | CPU | not measured (mostly downloads; can run during step 4) |
| 7 | `02e_holdout_evaluation` | no | GPU | prediction 7.7 min; only after step 5, with the fingerprinted model |
| 8 | `03_case_study_mannheim`, `03b_world_check` or the web app | yes | GPU or CPU | see the prediction times printed by 03b |

`01_prepare_dataset` and `02_train_unet` ran on Kaggle in round 1. The first cell of `01_prepare_dataset` writes
its own (older) `requirements.txt`; skip that cell when running the notebook inside this project folder.

`02_train_unet` (round 1, run on Kaggle) and `02b_round2_final` (round 2) are kept to document the improvement
rounds; their numbers are historical. They are not needed to reproduce the final model.

### Web app

```bash
python prototype/app.py --project YOUR-EARTH-ENGINE-PROJECT --model-dir models/round3
```

Open <http://127.0.0.1:7860> (only reachable from the same computer). Choose the place by clicking the map, by
searching a place name above the map, by typing latitude and longitude or with one of the example places (the name of
the place at the centre is shown under the map), set the size of the area (2–16 km), the start year and
month and the prediction mode, then click **Create 12 monthly maps**. `--model-dir` defaults to `models/round3`.

The page (titled *The Radar Almanac*) shows, after a request:
- a summary of the request (place, period, radar track, scenes per month, counted area, model) and notes on missing
  months or low coverage,
- **Plate 1:** the twelve predicted maps with corner coordinates, scale bar and legend (hatched = no radar data),
- **Plate 2:** the total change first → last month (pp), the share of every class per month and the month-to-month
  changes (pp) as a coloured table,
- **Plate 3:** the month-by-month agreement with Dynamic World (only if selected),
- the exact tables and one download (.zip) with all files of the request.

While a request runs, the page shows which month is being predicted. Old results are removed as soon as a new
request starts, and an error (for example latitude 78°) is shown instead of the results, so results of two
requests are never mixed. The location map uses Leaflet with EOxCloudless 2021 (Sentinel-2) and OpenStreetMap as
background, loaded by the browser; it is only for choosing the place and is never given to the model. Without
internet access to these sites the coordinates can still be typed.
`ui.py` builds the page only from the files and tables that `run_with_record()` produces, so the page shows exactly
the numbers in the CSV files.

On a JupyterHub, `127.0.0.1` is the hub's computer, not your own, so that address does not open in your browser.
Run `04_web_app.ipynb` instead: its last cell starts the app (`app.start(share=True, ...)`) and prints a temporary
`https://....gradio.live` link. The app runs as long as that cell runs; the Stop button (■) stops it.

`share=True` creates a temporary **public** link: anyone who has the link can use the app (and your Earth Engine
quota) while the cell runs. Use it only for testing or a short demonstration and stop the cell afterwards.

**Saved results:** every run gets its own folder in `data/prototype_results/app/` (named after place, area size,
start month, mode, a short fingerprint of the model description files and the date and time of the run, so earlier
results are never overwritten) with the 12 maps (GeoTIFF, full 10 m grid; the page shows reduced previews),
`common_valid_area.tif` (the pixels the shares and changes are calculated on), the three tables (CSV),
`dynamic_world_comparison.csv` (if selected), `request.json` (all inputs, track, missing months, prediction time and
the SHA-256 checksums of the weight files that were used) and one `.zip` with all of these files.

**Two prediction modes:** *Standard* uses the selected model (SegFormer-B3, one prediction per tile). *Ensemble*
uses the setting chosen on the validation set in notebook 02c: SegFormer-B3, SegFormer-B2, DeepLabV3+ and the
ResNet50 U-Net, each also predicting 8 flipped/rotated versions of every tile, averaged (32 predictions per tile;
validation mIoU 0.364 → 0.379, holdout mIoU 0.350 → 0.375). A higher score on these sets does not guarantee a better
map at every place. It is slower (on the RTX A6000: 12 maps of 10 km in 108 s instead of 31 s) and meant for a GPU.
Start the app with `--mode ensemble` to load it in advance. Flips and rotations change the
apparent SAR viewing geometry; they were used as training augmentation and their benefit was measured on the
validation set, not assumed.

**Any place on Earth:** nothing is pre-downloaded. For every request the Sentinel-1 images of that place and period are
fetched from Earth Engine, so the app works anywhere Sentinel-1 recorded in IW mode with VV + VH. Where this is not
the case, the app gives a clear message instead of maps:

| Situation | What the app does |
|---|---|
| Latitude north of 75° or south of 60° | Message: Sentinel-1 records mostly in EW mode there (the model was trained on IW) |
| Open sea, or no Sentinel-1 recording in the 12 months | Message: no Sentinel-1 images found. Example: Lusaka, the Algerian Sahara and Ulaanbaatar have no IW recording in 2023 (Sentinel-1A alone after Sentinel-1B failed in December 2021), but have data in 2021 |
| Area crossing the 180° meridian | Message: move the point slightly or choose a smaller area |
| Area larger than 16 km | Message: Earth Engine's direct download is limited to about 32 MB per file |
| Dynamic World comparison fails for a month | The month is marked in the comparison table; the Sentinel-1 results are still shown |
| Some months without an image on the chosen track | Those months are shown as "no radar image"; shares and changes use the other months |
| Part of the area not covered in a month | Shown as "low coverage"; only pixels covered in every month are counted |
| 12 months not finished yet / before 2015 | Message asking for another start month |

Check with 17 requests (11 land places on all continents, 3 of them again in 2023 when Sentinel-1 did not record
them, and 3 impossible places), through exactly the same code as
the app. Every place is reported as complete (12 maps), partial (some months without an image), unavailable,
rejected, error or crash, and the tables are checked automatically (12 consecutive months, shares adding up to 100 %,
changes equal to the differences of the shares):

```bash
python prototype/world_check.py --project YOUR-EARTH-ENGINE-PROJECT --model-dir models/round3 --mode standard   # or: --mode ensemble
```

---

## 6. Model

**SegFormer-B3** (Mix Transformer encoder pretrained on ADE20K, 44.6 M parameters, fully fine-tuned on Sentinel-1),
selected on validation mIoU in round 3 (notebook 02d). This is the app's *standard* mode.

**Ensemble setting** (notebook 02c, chosen on the validation set): the probabilities of SegFormer-B3,
SegFormer-B2, DeepLabV3+ and the ResNet50 U-Net, each for 8 flipped/rotated versions of every tile, are averaged.
This is the app's *ensemble* mode.

| | validation mIoU | test OA | test mIoU | 30 random areas OA | 30 random areas mIoU (95 % range) |
|---|---|---|---|---|---|
| SegFormer-B3 | 0.364 | 69.2 % | 0.428 | 60.6 % | 0.350 (0.292–0.420) |
| Ensemble × 8 flips/rotations | 0.379 | 69.9 % | 0.432 | 64.8 % | 0.375 (0.312–0.448) |

**Transfer learning:** the encoders start from weights learned on photographs (ImageNet for ResNet50, ADE20K for the
Mix Transformer). These weights are only the starting point; all layers are then trained on the Sentinel-1 patches.
The **only input** of every model, in training and in the prototype, is Sentinel-1: VV, VH and VV − VH. No optical
data is used. A U-Net trained from scratch is included for comparison (test mIoU 0.301).

Files in `models/round3/`: `lulc_model.keras` (selected model), `normalisation.json` (clipping limits, mean and
standard deviation, computed in float64 on the training data only), `model_info.json` (classes, colours, scores),
`inference_choice.json` (ensemble setting from 02c), `environment.json` (package versions), and one folder per
variant with `best.keras` and `history.json`.

Input preprocessing: VV, VH and VV − VH, clipped to the 0.5–99.5 % range of the training data and standardised.

---

## 7. Prototype: what is calculated

- **Share of a class in a month** = pixels of that class / counted pixels × 100.
  Only pixels with radar data in **every** available month are counted, so missing coverage cannot look like change.
- **Month-to-month change** = share this month − share previous month, in **percentage points (pp)**.
- **Total change** = share in the last month − share in the first month, in pp.
  Example: built-up 40.0 % → 43.0 % is +3.0 pp (not +7.5 %).
- Large images are predicted in overlapping 256 × 256 tiles with centre-weighted blending (no visible seams).
- **Track choice:** for every Sentinel-1 track (orbit direction + relative orbit) over the area, the number of scenes
  per month and the share of the area that has both VV and VH in all of its months are measured. Tracks that cover at
  least 90 % of the area are preferred; among them the one with the most months with images, then the largest
  coverage, then the most scenes. If no track covers 90 %, the app warns that the statistics describe only the
  counted part of the area. One track for all 12 months gives the same viewing geometry and the same pixel grid.
- Months without an image on the chosen track are shown as "no data"; months with less than 90 % coverage are flagged,
  and so is a counted area (radar in every available month) below 90 % of the requested area.
- The Dynamic World comparison uses the same counted pixels as the shares.
- Earth Engine login happens once at setup / app start; during a request the app never waits for a login but shows a
  clear message.
- Every model output is checked (shape, finite values, probabilities adding up to 1) before a map is made.
- Every map is checked against its radar file (same grid; no class where radar is missing).
- Downloads are cached and re-validated against the grid before they are reused.

---

## 8. Results

### Improvement rounds (validation = 6 areas, test = 6 areas; selection always on validation)

| Round | Model | training areas | parameters | val OA | val mIoU | test OA | test mIoU |
|---|---|---|---|---|---|---|---|
| 1 | U-Net from scratch | 21 | 7.8 M | 0.576 | 0.317 | – | – |
| 1 | U-Net + ResNet50 (ImageNet) | 21 | 32.6 M | 0.583 | 0.324 | 0.642 | 0.374 |
| 2 | SegFormer-B2 (ADE20K) | 55 | 24.7 M | 0.620 | 0.361 | 0.691 | 0.426 |
| 3 | U-Net from scratch | 55 | 7.8 M | 0.563 | 0.308 | 0.577 | 0.301 |
| 3 | U-Net + ResNet50 (ImageNet) | 55 | 32.6 M | 0.607 | 0.350 | 0.658 | 0.377 |
| 3 | DeepLabV3+ + ResNet50 (ImageNet) | 55 | 17.8 M | 0.618 | 0.353 | 0.664 | 0.394 |
| 3 | SegFormer-B2 (ADE20K) | 55 | 24.7 M | 0.620 | 0.361 | 0.682 | 0.422 |
| 3 | **SegFormer-B3 (ADE20K), selected** | 55 | 44.6 M | **0.625** | **0.364** | **0.692** | **0.428** |
| 3 | control: U-Net + ResNet50, round-1 data | 21 | 32.6 M | 0.585 | 0.318 | 0.637 | 0.363 |
| 3 | ensemble × 8 flips/rotations (02c) | 55 | – | 0.642 | 0.379 | 0.699 | 0.432 |

Round-1 and round-2 rows are historical results of earlier notebooks (02, 02b); round 2 used a normalisation with a
float32 rounding error that was fixed in round 3. The best of 15 settings was chosen on validation in 02c, which makes
its validation score slightly optimistic; the independent holdout below is the fair check.

- Round 3 fixed the normalisation (float64 statistics) and the control run; the scores stayed practically the same
  as in round 2 (SegFormer-B2 again 0.361 on validation).
- **Effect of the additional 34 training areas** (control vs. full run, same model and settings, one training run
  each): validation mIoU 0.318 → 0.350 (+0.032); cropland 0.24 → 0.30, permanent water 0.51 → 0.62. With a single run
  per setting, part of this difference can be random variation between training runs.
- Test IoU of SegFormer-B3: water 0.79, built-up 0.61, cropland 0.58, trees 0.48, grassland/shrubland 0.41,
  bare/sparse vegetation 0.08, herbaceous wetland 0.04 (only 62,424 wetland test pixels).
- SegFormer is the most stable over the seasons (test mIoU 0.417–0.437 in the four months); the CNNs drop in July.

### Independent evaluation on 30 randomly sampled, unseen areas (notebooks 01d, 02e)

Areas drawn at random (fixed seed, five per continent, ≥ 50 km from every project area), never used for training or
selection. The model files and the 02c choice were fingerprinted (SHA-256) before the data was loaded. The notebook was
run a second time with the same fingerprinted model only to save more detail (per-area matrices of both settings,
second bootstrap, experiment record); it reproduces the first run, which is kept as `*_first_run.*`. 95 % ranges come from a bootstrap over areas (2,000 repetitions), because neighbouring pixels are not
independent.

| | OA | mIoU (95 % range) |
|---|---|---|
| Ensemble × 8 flips/rotations | 0.648 (0.557–0.736) | 0.375 (0.312–0.448) |
| SegFormer-B3 alone | 0.606 (0.511–0.704) | 0.350 (0.292–0.420) |

A second bootstrap that resamples five areas within each continent (matching how the areas were drawn) is reported
next to it in notebook 02e. Scope: the areas were drawn inside rough boxes around six continents, outside 50 km of all project areas,
with the stopping rule "first five valid areas per continent"; the result describes this sample and does not prove
accuracy everywhere on Earth.

- Per class (ensemble): trees 0.72, cropland 0.52, grassland/shrubland 0.44, built-up 0.36, bare 0.34,
  water 0.16, wetland 0.10. Random land is mostly natural: built-up is only 0.9 % of the pixels.
- Lower than on the hand-picked test areas, as expected for unseen and unusual places; the ensemble's gain from 02c
  holds on the new areas (+0.025 mIoU, +4.2 pp OA).
- Largest errors: a sand desert in Egypt (99.9 % bare in WorldCover) and a steppe in Inner Mongolia (100 % grassland)
  are predicted as water. Very smooth, dry surfaces (and possibly frozen or snow-covered ground in winter) reflect the
  radar away from the satellite, like calm water, so they look dark in both VV and VH.
- By continent: Europe 0.44 mIoU, North America 0.35, Asia 0.32, South America 0.31, Oceania 0.28, Africa 0.18.
- Sentinel-1 data was usable for 30 of the 34 eligible land candidates that were checked in this procedure (88 %);
  checking stopped after five accepted areas per continent.

### Case study: Mannheim, January–December 2021 (notebook 03)

Mannheim (49.4875 N, 8.4660 E, 10 × 10 km) is an urban area that is not part of the training, validation or test data.
The prototype was run with the final setting (ensemble); all numbers are on the pixels counted in the shares
(100 % of the image). WorldCover is the product the training labels come from (evaluated here at a place not used for
training); Dynamic World is an external reference that was never used for training.

| | Result |
|---|---|
| Class shares over the year | built-up 52.1–58.5 %, trees 22.6–29.3 %, water 6.8–7.7 %, grassland 6.2–10.1 %, cropland 2.9–5.5 % |
| Total change January → December | built-up −1.75 pp, trees +2.21 pp, all other classes within ±0.3 pp |
| Largest month-to-month change | June → July: trees +3.98 pp, built-up −2.48 pp; July → August: trees −4.40 pp, built-up +3.78 pp |
| Agreement with ESA WorldCover 2021 | 74.3–76.8 % per month; IoU water 0.80, built-up 0.72, cropland 0.53, trees 0.49, grassland 0.32 |
| Agreement with Dynamic World (cloud-free pixels; 8 months usable) | 65.2–76.4 %; IoU water 0.75, built-up 0.71, cropland 0.37, trees 0.25 |
| Pixels with the same class in all 12 months | 73.5 % |
| Persistent predicted transition (one change, old class in ≥ 3 months before, new class in ≥ 3 months after) | 0.55 % of the area; Dynamic World supports 0.8 % of the checkable ones and shows no change for 90.2 % |
| Other predicted transitions (no persistent pattern, e.g. back and forth) | 25.9 % |
| Built-up in January → vegetation in July | 7.3 % of the area; Dynamic World shows built-up in both months for 59.0 % of these pixels |
| Prediction time, 12 maps (RTX A6000) | ensemble 108 s, standard 31 s |

Interpretation: the class shares are stable within a few percentage points, and the largest swings happen in summer
(trees up, built-up down, and back in August). Dynamic World, made from optical images, shows almost no change for the
same pixels. The month-to-month and total changes for Mannheim in 2021 are therefore mostly seasonal radar effects
(leaves on street and garden trees) and model errors, not real land cover change. Single-month changes of a few
percentage points should not be read as change without a persistence check.

---

## 9. Limitations

- Labels come from WorldCover, which itself is a model output and contains errors; the model can learn them.
- Radar separates some classes poorly (grassland, cropland and bare land have similar backscatter); herbaceous
  wetland and bare land are rare in the validation and test areas, so their scores are uncertain.
- Seasonal effects (leaves, soil moisture, frost) change the backscatter. In Mannheim 2021, 7.3 % of the area switched
  from built-up (January) to vegetation (July) in the prediction, while Dynamic World called 59.0 % of the comparable
  pixels built-up in both months: much of the apparent change is a radar effect, not real land cover change.
- Monthly differences in the prototype are differences between predictions; they are not verified land cover changes.
- Snow and ice are not among the 7 classes: snow-covered ground is given one of the 7 classes and is not detected or
  rejected by the prototype. Training covers four months of one year (2021).
- Prototype limits: latitude −60 to 75 (Sentinel-1 IW mode, which the model was trained on), no areas crossing the
  180° meridian, area size 2–16 km (Earth Engine download limit).
- The app runs anywhere Sentinel-1 data exists, but the quality of the maps depends on how similar a place is to the
  55 training areas: on 30 randomly drawn, unseen areas the mIoU is 0.375 (95 % range 0.31–0.45), lower than on the
  test areas.
- The test areas were already looked at during development (round 1); the model is selected on the validation set only.
- Smooth, dry surfaces (sand, flat dry or frozen steppe) can be predicted as water, because both reflect little radar
  back to the satellite (holdout areas Africa_003 and Asia_005). The model was not changed after this was found, so the
  holdout result stays independent.

---

## 10. Data credits and references

- Contains modified Copernicus Sentinel data (2021), processed in Google Earth Engine.
- ESA WorldCover 2021 v200, © ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021)
  processed by ESA WorldCover consortium. Licence CC BY 4.0.
- Brown, C. F. et al. (2022). Dynamic World, near real-time global 10 m land use land cover mapping. *Scientific Data* 9, 251. Licence CC BY 4.0.
- Ronneberger, O. et al. (2015). U-Net: Convolutional networks for biomedical image segmentation. MICCAI.
- He, K. et al. (2016). Deep residual learning for image recognition. CVPR.
- Chen, L.-C. et al. (2018). Encoder-decoder with atrous separable convolution for semantic image segmentation (DeepLabV3+). ECCV.
- Xie, E. et al. (2021). SegFormer: Simple and efficient design for semantic segmentation with transformers. NeurIPS.
- Šćepanović, S. et al. (2021). Wide-area land cover mapping with Sentinel-1 imagery using deep learning semantic segmentation models.
- Location map of the web app (background only, not used by the model): EOxCloudless 2021, https://cloudless.eox.at by
  EOX IT Services GmbH (Contains modified Copernicus Sentinel data 2021), licence CC BY-NC-SA 4.0 (non-commercial use);
  © OpenStreetMap contributors (ODbL); map library Leaflet 1.9.4 (BSD-2-Clause). Place search and place names:
  Nominatim (OpenStreetMap), asked by the browser at most once per second and only when the place changes.

## 11. Use of AI

AI assistants (Claude by Anthropic, and ChatGPT) were used during the project. All code was run and checked on the
real data before it was used. How the tools were used, and the project's decisions and problems, are described in the
report (section 12 and the project log in the appendix).
