"""
Core of the LULC prototype.

Used by the web app and the notebooks 03_case_study_mannheim and 03b_world_check, so all run the same code:
download 12 monthly Sentinel-1 images, predict a land cover map for each month,
calculate class shares and changes in percentage points.
"""
import json
import math
from pathlib import Path

import numpy as np
import requests
import rasterio
import rasterio.errors

NODATA = -9999
IGNORE = 255
PIXEL_SIZE = 10
PATCH_SIZE = 256
# Earth Engine's direct download (getDownloadURL) is limited to about 32 MB per request
MAX_DOWNLOAD_BYTES = 30_000_000
MIN_TRACK_COVERAGE = 0.9   # a track should cover at least 90 % of the area in all its months


# ---------------------------------------------------------------- model

def load_model(model_dir):
    import keras

    model_dir = Path(model_dir)
    norm = json.loads((model_dir / "normalisation.json").read_text())
    info = json.loads((model_dir / "model_info.json").read_text())
    if info.get("needs_keras_hub"):
        try:
            import keras_hub  # noqa: F401  (registers the SegFormer layers so the model can be loaded)
        except ImportError as error:
            version = "0.33.0" if tuple(int(p) for p in keras.__version__.split(".")[:2]) >= (3, 15) else "0.24.0"
            raise ImportError(
                f"This model ({info.get('model')}) needs keras-hub. Install it without dependencies, otherwise pip "
                f"replaces TensorFlow:\n  pip install --no-deps keras-hub=={version}\n  pip install regex rich kagglehub"
            ) from error
    model = keras.models.load_model(model_dir / "lulc_model.keras", compile=False)
    return model, norm, info


def load_ensemble(model_dir):
    """
    The setting chosen on the validation set in notebook 02c (models/round3/inference_choice.json):
    several models, each also predicting flipped / rotated tiles. Same normalisation as the single model.
    """
    import keras

    model_dir = Path(model_dir)
    choice_path = model_dir / "inference_choice.json"
    if not choice_path.exists():
        raise FileNotFoundError(f"{choice_path} is missing (made by notebook 02c)")
    choice = json.loads(choice_path.read_text())
    selected, norm, info = load_model(model_dir)     # also checks that keras-hub is installed
    n_classes = len(info["classes"])

    # the members are the checkpoints that were evaluated in 02c and 02e
    members = []
    for name in choice["models"]:
        own_norm = model_dir / name / "normalisation.json"
        if own_norm.exists() and json.loads(own_norm.read_text()) != norm:
            raise ValueError(f"{name} was trained with other normalisation values than the selected model")
        member = keras.models.load_model(model_dir / name / "best.keras", compile=False)
        if member.output_shape[-1] != n_classes:
            raise ValueError(f"{name} predicts {member.output_shape[-1]} classes instead of {n_classes}")
        members.append(member)

    # lulc_model.keras (standard mode) must be the same network as the selected checkpoint
    if info["model"] in choice["models"]:
        test_tiles = np.random.default_rng(0).normal(size=(1, PATCH_SIZE, PATCH_SIZE, 3)).astype(np.float32)
        same = members[choice["models"].index(info["model"])]
        difference = np.abs(np.asarray(selected.predict_on_batch(test_tiles), dtype=np.float32)
                            - np.asarray(same.predict_on_batch(test_tiles), dtype=np.float32)).max()
        if difference > 1e-3:
            raise ValueError(f"lulc_model.keras differs from {info['model']}/best.keras (max. difference {difference:.4f})")
    del selected

    transforms = [tuple(t) for t in choice["transforms"]]
    info = {**info, "model": f"ensemble of {len(members)} models ({', '.join(choice['models'])}), "
                             f"{len(transforms)} flips/rotations each",
            "test_miou": choice["test_miou"], "test_oa": choice["test_oa"]}
    return Ensemble(members, transforms), norm, info


class Ensemble:
    """Averages the probabilities of several models over flipped / rotated versions of each tile."""

    def __init__(self, members, transforms):
        self.members = members
        self.transforms = transforms     # (k, flip): rotate by k * 90 degrees, flip left-right first

    def predict_on_batch(self, tiles):
        total = 0
        for k, flip in self.transforms:
            view = tiles[:, :, ::-1] if flip else tiles
            view = np.ascontiguousarray(np.rot90(view, k, axes=(1, 2)))
            for model in self.members:
                p = np.rot90(np.asarray(model.predict_on_batch(view), dtype=np.float32), -k, axes=(1, 2))
                total = total + (p[:, :, ::-1] if flip else p)
        return total / (len(self.transforms) * len(self.members))


def normalise(x, norm):
    # x: (..., 2) VV and VH in dB, same steps and order as in training
    x = np.clip(x, np.float32(norm["clip_low"]), np.float32(norm["clip_high"]))
    if norm["add_ratio"]:
        x = np.concatenate([x, x[..., 0:1] - x[..., 1:2]], axis=-1)
    return (x - np.float32(norm["mean"])) / np.float32(norm["std"])


def check_probabilities(predicted, n_tiles, n_classes):
    # catches a broken model or a wrong model file before a map is shown
    if predicted.shape != (n_tiles, PATCH_SIZE, PATCH_SIZE, n_classes):
        raise ValueError(f"model output has shape {predicted.shape}, expected {(n_tiles, PATCH_SIZE, PATCH_SIZE, n_classes)}")
    if not np.isfinite(predicted).all():
        raise ValueError("model output contains NaN or infinite values")
    if not np.allclose(predicted.sum(axis=-1), 1.0, atol=1e-2):
        raise ValueError("model output is not a probability for every pixel (sums differ from 1)")


def predict_map(model, norm, radar, n_classes, stride=192, batch_size=8):
    """
    radar: (2, H, W) array as read with rasterio, NODATA where the radar is missing.
    Returns a (H, W) class map with IGNORE where there is no radar.

    The image is cut into overlapping 256 x 256 tiles. Every tile is weighted more in its centre
    than at its border, so the seams between tiles disappear.
    """
    valid = np.all(radar != NODATA, axis=0) & np.all(np.isfinite(radar), axis=0)
    height, width = valid.shape

    x = np.moveaxis(radar, 0, -1).astype(np.float32)
    x[~valid] = np.float32(norm["mean"][:2])          # neutral value, these pixels are masked at the end
    x = normalise(x, norm)

    # pad right and bottom so the tiles cover the whole image
    padded_h = PATCH_SIZE + max(0, math.ceil((height - PATCH_SIZE) / stride)) * stride
    padded_w = PATCH_SIZE + max(0, math.ceil((width - PATCH_SIZE) / stride)) * stride
    pad = ((0, padded_h - height), (0, padded_w - width), (0, 0))
    mode = "reflect" if height > padded_h - height and width > padded_w - width else "edge"
    x = np.pad(x, pad, mode=mode)

    window = np.outer(np.hanning(PATCH_SIZE), np.hanning(PATCH_SIZE)) + 0.01
    window = window[..., np.newaxis].astype(np.float32)

    probabilities = np.zeros((padded_h, padded_w, n_classes), dtype=np.float32)
    weights = np.zeros((padded_h, padded_w, 1), dtype=np.float32)

    corners = [(r, c) for r in range(0, padded_h - PATCH_SIZE + 1, stride)
               for c in range(0, padded_w - PATCH_SIZE + 1, stride)]

    for start in range(0, len(corners), batch_size):
        batch = corners[start:start + batch_size]
        tiles = np.stack([x[r:r + PATCH_SIZE, c:c + PATCH_SIZE] for r, c in batch])
        predicted = np.asarray(model.predict_on_batch(tiles), dtype=np.float32)
        check_probabilities(predicted, len(batch), n_classes)
        for (r, c), p in zip(batch, predicted):
            probabilities[r:r + PATCH_SIZE, c:c + PATCH_SIZE] += p * window
            weights[r:r + PATCH_SIZE, c:c + PATCH_SIZE] += window

    probabilities = probabilities[:height, :width] / weights[:height, :width]
    class_map = np.argmax(probabilities, axis=-1).astype(np.uint8)
    class_map[~valid] = IGNORE
    return class_map


# ---------------------------------------------------------------- statistics

def class_shares(maps, n_classes):
    """
    maps: list of (H, W) class maps on the same grid, one per month (None = no image that month).
    Only pixels that are valid in EVERY available month are counted, so missing radar
    coverage in one month cannot look like land cover change.
    Returns shares in % with shape (months, classes) (NaN rows for missing months) and the common mask.
    """
    available = [m for m in maps if m is not None]
    if not available:
        raise ValueError("No month has radar data.")
    common = np.all([m != IGNORE for m in available], axis=0)
    if common.sum() == 0:
        raise ValueError("No pixel has radar data in all months.")

    shares = np.full((len(maps), n_classes), np.nan)
    for i, m in enumerate(maps):
        if m is not None:
            shares[i] = [100 * np.mean(m[common] == k) for k in range(n_classes)]
    return shares, common


def changes(shares):
    # differences of percentages = percentage points (pp); NaN if one of the two months is missing
    month_to_month = np.diff(shares, axis=0)         # (months - 1, classes)
    total = shares[-1] - shares[0]                   # last month - first month
    return month_to_month, total


def month_list(start_year, start_month, count=12):
    months = []
    for i in range(count):
        index = start_month - 1 + i
        months.append((start_year + index // 12, index % 12 + 1))
    return months


# ---------------------------------------------------------------- Sentinel-1 from Earth Engine

def init_earth_engine(project_id, interactive=True):
    """
    interactive=True (setup, notebooks): asks for a login if there is no saved token.
    interactive=False (inside a web request): never waits for input, raises a clear error instead.
    """
    import ee
    try:
        ee.Initialize(project=project_id)
    except Exception as error:
        if not interactive:
            raise RuntimeError("Earth Engine is not logged in or the project is not available. Log in once before "
                               "starting the app (see README, section 3) and start the app again. "
                               f"Details: {error}") from error
        ee.Authenticate(auth_mode="notebook")
        ee.Initialize(project=project_id)


def check_place(lat, lon, buffer_metres):
    """Raises ValueError with a clear message for places the prototype cannot handle."""
    if not -60 <= lat <= 75:
        raise ValueError("Latitude must be between -60 and 75. Further north and south, Sentinel-1 mostly records "
                         "in another mode (EW instead of IW), which the model was not trained on.")
    if not -180 <= lon <= 180:
        raise ValueError("Longitude must be between -180 and 180.")
    # the square area must not cross the 180° meridian (it would wrap around the whole globe)
    half_width_degrees = buffer_metres / (111_320 * math.cos(math.radians(lat)))
    if abs(lon) + half_width_degrees >= 180:
        raise ValueError("The area crosses the 180° meridian, which is not supported. "
                         "Please move the point slightly or choose a smaller area.")


def study_area(lat, lon, buffer_metres):
    import ee
    return ee.Geometry.Point([lon, lat]).buffer(buffer_metres).bounds()


def sentinel1(region, start, end):
    import ee
    return (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.eq("resolution_meters", 10))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    )


def utm_crs(lat, lon):
    zone = min(60, max(1, math.floor((lon + 180) / 6) + 1))
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def best_track(counts, coverage):
    """
    counts: {month key: {track: number of scenes}}, coverage: {track: share of the area covered in all its months}.
    Preference: 1. covers at least 90 % of the area, 2. most months with images, 3. largest covered share,
    4. highest minimum number of scenes per month, 5. most scenes. Ties are broken by the track name.
    If no track covers 90 %, the app shows the small counted area as a warning.
    """
    def score(track):
        per_month = [counts[key].get(track, 0) for key in counts]
        share = coverage.get(track, 0.0)
        return (share >= MIN_TRACK_COVERAGE, sum(n > 0 for n in per_month), round(share, 3),
                min(per_month), sum(per_month))

    return max(sorted(coverage), key=score)


def choose_track(region, months, lat, lon):
    """
    Same idea as in the dataset notebook: one track (direction + relative orbit) for all months,
    so every month has the same viewing angle and the same pixel grid.
    For every track, the number of scenes per month and the share of the area that the track covers in all of its
    months are measured; best_track() chooses between them.
    Returns the track, the number of scenes per month (0 = month has no image on this track) and its coverage.
    """
    import ee

    def add_track(image):
        direction = ee.String(image.get("orbitProperties_pass"))
        orbit = ee.Number(image.get("relativeOrbitNumber_start")).format("%d")
        return image.set("track", direction.cat("_").cat(orbit))

    queries = {}
    for year, month in months:
        start = ee.Date.fromYMD(year, month, 1)
        queries[f"{year}-{month:02d}"] = (
            sentinel1(region, start, start.advance(1, "month")).map(add_track).aggregate_histogram("track")
        )
    counts = ee.Dictionary(queries).getInfo()

    tracks = set()
    for month_counts in counts.values():
        tracks.update(month_counts)
    if not tracks:
        raise ValueError("No Sentinel-1 images (IW mode, VV + VH) found for this place and period. "
                         "This happens over the open sea and in regions Sentinel-1 did not record in these months.")

    # share of the area covered by each track in every month in which that track has images (20 m grid)
    crs = utm_crs(lat, lon)
    shares = {}
    for track in sorted(tracks):
        direction, orbit = track.rsplit("_", 1)
        covered = []
        for year, month in months:
            if counts[f"{year}-{month:02d}"].get(track, 0) == 0:
                continue
            start = ee.Date.fromYMD(year, month, 1)
            images = (sentinel1(region, start, start.advance(1, "month"))
                      .filter(ee.Filter.eq("orbitProperties_pass", direction))
                      .filter(ee.Filter.eq("relativeOrbitNumber_start", int(orbit)))
                      .select(["VV", "VH"]))
            # a pixel counts as covered only if both VV and VH have at least one image (as needed for the prediction)
            both = images.count().reduce(ee.Reducer.min())
            covered.append(both.unmask(0, sameFootprint=False).gt(0))
        in_all_months = ee.Image.cat(*covered).reduce(ee.Reducer.min()).rename("covered")
        shares[track] = in_all_months.reduceRegion(reducer=ee.Reducer.mean(), geometry=region, crs=crs, scale=20,
                                                   maxPixels=10_000_000, tileScale=4).get("covered")
    coverage = {track: float(value or 0.0) for track, value in ee.Dictionary(shares).getInfo().items()}

    best = best_track(counts, coverage)
    direction, orbit = best.rsplit("_", 1)
    scenes = {key: int(counts[key].get(best, 0)) for key in sorted(counts)}
    return direction, int(orbit), scenes, coverage[best]


def snapped_grid(region, collection):
    import ee

    radar_projection = ee.Image(collection.first()).select("VV").projection().getInfo()
    crs = radar_projection["crs"]
    origin_x = radar_projection["transform"][2]
    origin_y = radar_projection["transform"][5]

    bounds = region.bounds(maxError=1, proj=ee.Projection(crs)).coordinates().getInfo()[0]
    xs = [point[0] for point in bounds]
    ys = [point[1] for point in bounds]

    left_col = math.floor((min(xs) - origin_x) / PIXEL_SIZE)
    right_col = math.ceil((max(xs) - origin_x) / PIXEL_SIZE)
    top_row = math.floor((origin_y - max(ys)) / PIXEL_SIZE)
    bottom_row = math.ceil((origin_y - min(ys)) / PIXEL_SIZE)

    width = right_col - left_col
    height = bottom_row - top_row
    left = origin_x + left_col * PIXEL_SIZE
    top = origin_y - top_row * PIXEL_SIZE
    return crs, [PIXEL_SIZE, 0, left, 0, -PIXEL_SIZE, top], width, height


def download_tif(image, path, crs, grid_transform, width, height):
    url = image.getDownloadURL({
        "crs": crs,
        "crs_transform": grid_transform,
        "dimensions": [width, height],
        "format": "GEO_TIFF",
    })
    response = requests.get(url, timeout=300)
    response.raise_for_status()
    path.write_bytes(response.content)
    with rasterio.open(path, "r+") as dataset:
        dataset.nodata = NODATA


def matches_grid(path, grid, band_count):
    """True if a cached GeoTIFF has exactly the expected coordinate system, pixel grid, size and bands."""
    path = Path(path)
    if not path.exists():
        return False
    try:
        with rasterio.open(path) as src:
            return (src.crs is not None and src.crs.to_string() == grid["crs"]
                    and (src.width, src.height) == (grid["width"], grid["height"])
                    and src.count == band_count
                    # rtol=0: a relative tolerance would accept shifts of tens of metres on UTM coordinates
                    and np.allclose(tuple(src.transform)[:6], grid["transform"], rtol=0, atol=1e-6))
    except rasterio.errors.RasterioIOError:
        return False


def fetch_months(lat, lon, start_year, start_month, cache_folder, buffer_metres=5000, log=print):
    """
    Downloads 12 monthly median composites (VV, VH) of one track on one shared grid.
    Files are cached, so running the same place again is fast.
    Returns a list of dicts: year, month, scenes, path (None if no image in that month), and the grid info.
    """
    import ee

    check_place(lat, lon, buffer_metres)
    months = month_list(start_year, start_month)
    region = study_area(lat, lon, buffer_metres)

    direction, orbit, scenes, track_coverage = choose_track(region, months, lat, lon)
    log(f"Track: {direction} {orbit}, scenes per month: {list(scenes.values())}, "
        f"covers {100 * track_coverage:.1f} % of the area in all its months")

    first = ee.Date.fromYMD(*months[0], 1)
    last = ee.Date.fromYMD(*months[-1], 1).advance(1, "month")
    collection = (
        sentinel1(region, first, last)
        .filter(ee.Filter.eq("orbitProperties_pass", direction))
        .filter(ee.Filter.eq("relativeOrbitNumber_start", orbit))
        .select(["VV", "VH"])
    )
    crs, grid_transform, width, height = snapped_grid(region, collection)
    if width * height * 2 * 4 > MAX_DOWNLOAD_BYTES:          # 2 bands, 4 bytes each
        raise ValueError(f"The area is too large for one Earth Engine download ({width} x {height} pixels). "
                         "Please choose a smaller area.")

    # the grid origin is part of the folder name: another period can give another grid for the same place
    origin = f"{grid_transform[2]:.0f}_{grid_transform[5]:.0f}"
    folder = Path(cache_folder) / f"{lat:.4f}_{lon:.4f}_{buffer_metres}_{direction}_{orbit}_{origin}"
    folder.mkdir(parents=True, exist_ok=True)
    grid = {"crs": crs, "transform": grid_transform, "width": width, "height": height,
            "direction": direction, "orbit": orbit}
    (folder / "grid.json").write_text(json.dumps(grid, indent=2))
    grid["folder"] = str(folder)
    grid["track_coverage"] = track_coverage

    results = []
    for year, month in months:
        key = f"{year}-{month:02d}"
        path = folder / f"s1_{year}_{month:02d}.tif"
        if scenes[key] == 0:
            log(f"{key}: no image on this track")
            results.append({"year": year, "month": month, "scenes": 0, "path": None})
            continue
        if not matches_grid(path, grid, 2):
            start = ee.Date.fromYMD(year, month, 1)
            monthly = collection.filterDate(start, start.advance(1, "month")).median()
            download_tif(monthly.toFloat().unmask(NODATA, sameFootprint=False),
                         path, crs, grid_transform, width, height)
            if not matches_grid(path, grid, 2):
                raise ValueError(f"{key}: downloaded file does not match the grid")
            log(f"{key}: downloaded ({scenes[key]} scenes)")
        else:
            log(f"{key}: from cache")
        results.append({"year": year, "month": month, "scenes": scenes[key], "path": path})
    return results, grid


def read_radar(path):
    with rasterio.open(path) as src:
        return src.read().astype(np.float32)


# ---------------------------------------------------------------- saving and checking maps

def check_map(class_map, radar_path, grid, n_classes):
    """
    Independent check of one predicted map against its radar file:
    same grid, only known class values, and IGNORE exactly where VV or VH is missing.
    Returns (classified without radar, valid radar pixels left unclassified).
    """
    from affine import Affine

    with rasterio.open(radar_path) as src:
        assert src.count == 2, "expected VV and VH"
        assert src.shape == (grid["height"], grid["width"]), "image size differs from grid"
        assert src.crs.to_string() == grid["crs"], "coordinate system differs from grid"
        assert src.transform.almost_equals(Affine(*grid["transform"])), "pixel grid differs"
        radar = src.read()

    assert class_map.shape == radar.shape[1:], "map size differs from radar"
    assert np.isin(class_map, [*range(n_classes), IGNORE]).all(), "unknown class value in map"

    radar_ok = np.all(radar != NODATA, axis=0) & np.all(np.isfinite(radar), axis=0)
    classified_without_radar = int(np.sum(~radar_ok & (class_map != IGNORE)))
    valid_skipped = int(np.sum(radar_ok & (class_map == IGNORE)))
    return classified_without_radar, valid_skipped


def save_map(class_map, path, grid):
    # GeoTIFF on the same grid as the radar, so it can be opened in QGIS or compared with other maps
    from affine import Affine

    profile = {
        "driver": "GTiff", "height": grid["height"], "width": grid["width"], "count": 1,
        "dtype": "uint8", "crs": grid["crs"], "transform": Affine(*grid["transform"]),
        "nodata": IGNORE, "compress": "deflate",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(class_map, 1)


# ---------------------------------------------------------------- Dynamic World (only for evaluation)

# Dynamic World classes -> our 7 classes; 8 snow_and_ice -> IGNORE (out of scope, as in training)
DW_TO_CLASS = {0: 5, 1: 0, 2: 1, 3: 6, 4: 2, 5: 1, 6: 3, 7: 4}


def fetch_dynamic_world(lat, lon, year, month, grid, buffer_metres=5000):
    """
    Monthly Dynamic World label (most frequent label in the month) on the prototype grid.
    Dynamic World is derived from Sentinel-2. It is NEVER given to the model, it is only
    used as an independent reference for the case study.
    Returns a (H, W) map in our classes with IGNORE for clouds / no data, or None if there is no image.
    """
    import ee

    region = study_area(lat, lon, buffer_metres)
    start = ee.Date.fromYMD(year, month, 1)
    images = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
              .filterBounds(region)
              .filterDate(start, start.advance(1, "month")))
    if images.size().getInfo() == 0:
        return None

    path = Path(grid["folder"]) / f"dw_{year}_{month:02d}.tif"
    if not matches_grid(path, grid, 1):
        label = images.select("label").mode()
        download_tif(label.toInt16().unmask(NODATA, sameFootprint=False),
                     path, grid["crs"], grid["transform"], grid["width"], grid["height"])
        if not matches_grid(path, grid, 1):
            raise ValueError(f"Dynamic World {year}-{month:02d}: downloaded file does not match the grid")

    with rasterio.open(path) as src:
        codes = src.read(1)
    reference = np.full(codes.shape, IGNORE, dtype=np.uint8)
    for code, new_class in DW_TO_CLASS.items():
        reference[codes == code] = new_class
    return reference


# ESA WorldCover codes -> our 7 classes (same mapping as for the training labels); snow and no data -> IGNORE
WORLDCOVER_TO_CLASS = {10: 0, 95: 0, 30: 1, 20: 1, 100: 1, 40: 2, 50: 3, 60: 4, 80: 5, 90: 6}


def fetch_worldcover(grid):
    """
    ESA WorldCover 2021 on the prototype grid, in our classes (IGNORE for snow and no data).
    Only used for evaluation in the case study; the place is not part of the training data.
    """
    import ee

    path = Path(grid["folder"]) / "worldcover_2021.tif"
    if not matches_grid(path, grid, 1):
        label = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
        download_tif(label.toInt16().unmask(NODATA, sameFootprint=False),
                     path, grid["crs"], grid["transform"], grid["width"], grid["height"])
        if not matches_grid(path, grid, 1):
            raise ValueError("WorldCover: downloaded file does not match the grid")

    with rasterio.open(path) as src:
        codes = src.read(1)
    reference = np.full(codes.shape, IGNORE, dtype=np.uint8)
    for code, new_class in WORLDCOVER_TO_CLASS.items():
        reference[codes == code] = new_class
    return reference


def compare_maps(prediction, reference, n_classes):
    # agreement on pixels that are valid in both maps
    valid = (prediction != IGNORE) & (reference != IGNORE)
    if valid.sum() == 0:
        return None
    p, r = prediction[valid], reference[valid]
    iou = []
    for k in range(n_classes):
        union = np.sum((p == k) | (r == k))
        iou.append(np.sum((p == k) & (r == k)) / union if union else np.nan)
    return {
        "compared_fraction": float(valid.mean()),
        "agreement": float(np.mean(p == r)),
        "iou": np.array(iou),
        "pred_share": np.array([100 * np.mean(p == k) for k in range(n_classes)]),
        "ref_share": np.array([100 * np.mean(r == k) for k in range(n_classes)]),
    }
