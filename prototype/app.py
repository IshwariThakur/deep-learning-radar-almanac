"""
Web app of the LULC prototype.

Input:  latitude, longitude, start month (and the size of the area)
Output: 12 monthly land cover maps from Sentinel-1, the share of every class per month,
        the month-to-month change and the total change (first -> last month) in percentage points.

Start on your own computer (from the project folder):
    python prototype/app.py
then open http://127.0.0.1:7860 in the browser (only reachable from this computer).

Inside a Jupyter notebook (see notebooks/04_web_app.ipynb):
    import app
    demo = app.start(share=True, prevent_thread_lock=True)
share=True creates a temporary PUBLIC link: anyone who has the link can use the app (and your Earth Engine
quota) while it runs. Use it only for a short demonstration and stop it afterwards.

The calculations are in run_with_record() (and lulc_core.py); ui.py only builds the page from their results.
"""
import json
import time
import hashlib
import argparse
import datetime
import os
import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
if "ipykernel" not in sys.modules:      # web server / command line: draw figures without a screen;
    matplotlib.use("Agg")               # in a notebook the notebook's own backend is kept, so figures are shown
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lulc_core import (load_model, load_ensemble, predict_map, class_shares, changes, read_radar, init_earth_engine, check_place,
                       fetch_months, check_map, save_map, fetch_dynamic_world, compare_maps, IGNORE)
import ui

MODEL_DIR = os.environ.get("LULC_MODEL_DIR", "models/round3")
PROJECT_ID = os.environ.get("EE_PROJECT", "sentinel1-lulc-project-512618")
CACHE_FOLDER = Path("data/prototype_cache")
RESULT_FOLDER = Path("data/prototype_results/app")
MIN_COVERAGE = 0.9
MAX_SIZE_KM = 16          # larger areas exceed Earth Engine's download limit for two radar bands
MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]

MODES = {"standard": "Standard · SegFormer-B3 · fast",
         "ensemble": "Ensemble · 4 models × 8 flips/rotations · higher validation score, slower (GPU)"}
loaded = {}


def mode_key(mode):
    # accepts the key, the text shown in the app, or "best" (the earlier name of the ensemble mode)
    if mode == "best":
        return "ensemble"
    for key, text in MODES.items():
        if mode in (key, text):
            return key
    raise gr.Error(f"Unknown prediction mode: {mode}")


def get_model(mode="standard"):
    key = (str(Path(MODEL_DIR).resolve()), mode)          # another model folder never reuses a loaded model
    if key not in loaded:
        loaded[key] = load_ensemble(MODEL_DIR) if mode == "ensemble" else load_model(MODEL_DIR)
    return loaded[key]


weight_checksums = {}


def weights_checksum(mode):
    """SHA-256 of the weight files used by a mode (computed once per mode), saved in request.json, so a result can
    be traced back to the exact model files."""
    if mode not in weight_checksums:
        model_dir = Path(MODEL_DIR)
        if mode == "ensemble":
            choice = json.loads((model_dir / "inference_choice.json").read_text())
            paths = [model_dir / name / "best.keras" for name in choice["models"]]
        else:
            paths = [model_dir / "lulc_model.keras"]
        checksums = {}
        for path in paths:
            if not path.exists():
                checksums[str(path.relative_to(model_dir))] = "missing"
                continue
            digest = hashlib.sha256()
            with open(path, "rb") as file:
                for block in iter(lambda: file.read(1 << 20), b""):
                    digest.update(block)
            checksums[str(path.relative_to(model_dir))] = digest.hexdigest()
        weight_checksums[mode] = checksums
    return weight_checksums[mode]


def model_tag():
    # short fingerprint of the model folder's description files, part of the result folder name
    digest = hashlib.sha256()
    for name in ["model_info.json", "normalisation.json", "inference_choice.json"]:
        path = Path(MODEL_DIR) / name
        digest.update(path.read_bytes() if path.exists() else b"-")
    return digest.hexdigest()[:8]


# ---------------------------------------------------------------- checks of the user input

def check_input(lat, lon, start_year, start_month, size_km):
    try:
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError):
        raise gr.Error("Please enter latitude and longitude as numbers.") from None
    if not (np.isfinite(lat) and np.isfinite(lon)):
        raise gr.Error("Please enter latitude and longitude as numbers.")
    try:
        size_km = float(size_km)
    except (TypeError, ValueError):
        raise gr.Error(f"Please choose an area size between 2 and {MAX_SIZE_KM} km.") from None
    if not 2 <= size_km <= MAX_SIZE_KM:
        raise gr.Error(f"Area size must be between 2 and {MAX_SIZE_KM} km.")
    try:
        check_place(lat, lon, int(size_km * 500))
    except ValueError as error:
        raise gr.Error(str(error)) from None

    try:
        start_year, start_month = int(start_year), MONTH_NAMES.index(start_month) + 1
    except (TypeError, ValueError):
        raise gr.Error("Please select a valid start year and start month.") from None
    if start_year < 2015:
        raise gr.Error("Sentinel-1 data in Earth Engine starts in October 2014, please choose 2015 or later.")

    # the last of the 12 months must be over, otherwise its median image is incomplete
    last_index = start_month - 1 + 11
    last_year, last_month = start_year + last_index // 12, last_index % 12 + 1
    today = datetime.date.today()
    if (last_year, last_month) >= (today.year, today.month):
        raise gr.Error(f"The 12 months would end in {MONTH_NAMES[last_month - 1]} {last_year}, "
                       "which is not finished yet. Please choose an earlier start month.")
    return lat, lon, start_year, start_month, size_km


# ---------------------------------------------------------------- figures and tables

def map_figure(months, maps, classes, colors, lat, lon):
    cmap = ListedColormap(colors)
    fig, axes = plt.subplots(3, 4, figsize=(16, 13))
    for ax, entry, class_map in zip(axes.ravel(), months, maps):
        title = f"{MONTH_NAMES[entry['month'] - 1][:3]} {entry['year']}"
        if class_map is None:
            ax.text(0.5, 0.5, "no radar image", ha="center", va="center", fontsize=13)
            ax.set_title(title)
        else:
            ax.imshow(np.ma.masked_equal(class_map, IGNORE), cmap=cmap, vmin=-0.5,
                      vmax=len(classes) - 0.5, interpolation="nearest")
            flag = "  (low coverage)" if entry["coverage"] < MIN_COVERAGE else ""
            ax.set_title(f"{title} · {entry['scenes']} scenes{flag}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.legend(handles=[Patch(color=c, label=n) for c, n in zip(colors, classes)],
               loc="lower center", ncol=4, fontsize=11)
    fig.suptitle(f"Land use / land cover from Sentinel-1 · lat {lat:.4f}, lon {lon:.4f}", fontsize=15)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.subplots_adjust(hspace=0.25)          # room for the month titles between the rows
    return fig


def share_figure(labels, shares, classes, colors):
    fig, ax = plt.subplots(figsize=(14, 5))
    for k, name in enumerate(classes):
        ax.plot(labels, shares[:, k], marker="o", color=colors[k], label=name)
    ax.set_ylabel("share of the counted area (%)")
    ax.set_title("Class share per month")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.setp(ax.get_xticklabels(), rotation=45)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- one complete run

def run(lat, lon, start_year, start_month, size_km, compare_dw, mode="standard", progress=gr.Progress()):
    # the function behind the app's button: returns what the app shows
    return run_with_record(lat, lon, start_year, start_month, size_km, compare_dw, mode, progress)[0]


def run_with_record(lat, lon, start_year, start_month, size_km, compare_dw, mode="standard", progress=None):
    """One complete request. Returns (outputs for the app, record of the request); the record is also saved as
    request.json in the request's result folder."""
    progress = progress or (lambda *args, **kwargs: None)
    lat, lon, start_year, start_month, size_km = check_input(lat, lon, start_year, start_month, size_km)
    mode = mode_key(mode)
    try:
        model, norm, info = get_model(mode)
    except Exception as error:
        raise gr.Error(f"Could not load the model ({mode}): {error}")
    classes, colors = info["classes"], info["colors"]
    n_classes = len(classes)

    progress(0, desc="Searching Sentinel-1 images")
    log = []
    try:
        init_earth_engine(PROJECT_ID, interactive=False)
        months, grid = fetch_months(lat, lon, start_year, start_month, CACHE_FOLDER,
                                    buffer_metres=int(size_km * 500), log=log.append)
    except Exception as error:
        raise gr.Error(f"Could not get Sentinel-1 data: {error}")

    maps = []
    prediction_start = time.time()
    for i, entry in enumerate(months):
        progress((i + 1) / 13, desc=f"Predicting {MONTH_NAMES[entry['month'] - 1]} {entry['year']}")
        if entry["path"] is None:
            maps.append(None)
            entry["coverage"] = 0.0
            continue
        try:
            class_map = predict_map(model, norm, read_radar(entry["path"]), n_classes)
        except ValueError as error:
            raise gr.Error(f"Prediction failed: {error}")
        without_radar, unclassified = check_map(class_map, entry["path"], grid, n_classes)
        if without_radar or unclassified:
            raise gr.Error(f"Internal check failed: {without_radar} pixels classified without radar data, "
                           f"{unclassified} pixels with radar data left without a class.")
        entry["coverage"] = float(np.mean(class_map != IGNORE))
        maps.append(class_map)
    prediction_seconds = time.time() - prediction_start

    try:
        shares, common = class_shares(maps, n_classes)
    except ValueError as error:
        raise gr.Error(str(error))
    month_to_month, total = changes(shares)
    labels = [f"{e['year']}-{e['month']:02d}" for e in months]

    share_table = pd.DataFrame(shares, columns=classes).round(2)
    share_table.insert(0, "month", labels)
    change_table = pd.DataFrame(month_to_month, columns=classes).round(2)
    change_table.insert(0, "change", [f"{a} → {b}" for a, b in zip(labels, labels[1:])])
    total_table = pd.DataFrame([total], columns=classes).round(2)
    total_table.insert(0, "change", [f"{labels[0]} → {labels[-1]}"])

    # every run gets its own folder (place, size, start month, mode, model, date and time of the run), so results of
    # different runs are never mixed and earlier results stay as they were
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{lat:.4f}_{lon:.4f}_{size_km:g}km_{labels[0]}_{mode}_{model_tag()}_{run_id}"
    folder = RESULT_FOLDER / name
    number = 2
    while folder.exists():
        folder = RESULT_FOLDER / f"{name}_{number}"
        number += 1
    folder.mkdir(parents=True)
    files = []
    for label, class_map in zip(labels, maps):
        if class_map is not None:
            save_map(class_map, folder / f"lulc_{label}.tif", grid)
            files.append(str(folder / f"lulc_{label}.tif"))
    # the pixels the shares and changes are calculated on (1 = counted, 0 = not counted)
    save_map(common.astype(np.uint8), folder / "common_valid_area.tif", grid)
    files.append(str(folder / "common_valid_area.tif"))
    for name, table in [("shares", share_table), ("month_to_month_pp", change_table), ("total_change_pp", total_table)]:
        table.to_csv(folder / f"{name}.csv", index=False)
        files.append(str(folder / f"{name}.csv"))

    missing = [labels[i] for i, m in enumerate(maps) if m is None]
    low = [labels[i] for i, e in enumerate(months) if maps[i] is not None and e["coverage"] < MIN_COVERAGE]
    area_km2 = common.sum() * 100 / 1e6
    summary = [
        f"**Track:** {grid['direction'].lower()} orbit {grid['orbit']} (same viewing geometry and pixel grid in all months; "
        f"chosen by number of months, coverage of the area and number of scenes)",
        f"**Scenes per month:** {', '.join(str(e['scenes']) for e in months)}",
        f"**Counted area:** {area_km2:.1f} km² ({100 * common.mean():.1f}% of the image). Only pixels with radar data "
        "in every available month are counted, so missing coverage cannot look like change. All shares and changes "
        "refer to this area (saved as common_valid_area.tif).",
        f"**Model:** {info['model']} (test mIoU {info['test_miou']:.3f} on the six test areas)",
        f"**Prediction time:** {prediction_seconds:.0f} s for {len(labels) - len(missing)} maps",
    ]
    if missing:
        summary.append(f"⚠️ **No radar image on this track:** {', '.join(missing)}")
    if low:
        summary.append(f"⚠️ **Low radar coverage (< {100 * MIN_COVERAGE:.0f}%):** {', '.join(low)}")
    if common.mean() < MIN_COVERAGE:
        summary.append(f"⚠️ **Small counted area:** only {100 * common.mean():.1f}% of the requested area has radar data "
                       "in every available month. All shares and changes describe only this part of the area.")
    summary.append("The tables show **predicted** land cover and **predicted** changes in percentage points. "
                   "Month-to-month changes of a few points can be caused by seasonal radar effects (leaves, soil "
                   "moisture, frost) and by model errors, not only by real land cover change.")

    # optional comparison with Dynamic World; a failed month does not stop the Sentinel-1 results
    dw_table = None
    if compare_dw:
        progress(1, desc="Comparing with Dynamic World")
        rows, failed = [], []
        for entry, class_map in zip(months, maps):
            label = f"{entry['year']}-{entry['month']:02d}"
            row = {"month": label, "status": "ok"}
            try:
                reference = fetch_dynamic_world(lat, lon, entry["year"], entry["month"], grid, int(size_km * 500))
            except Exception as error:
                reference = None
                row["status"] = f"download failed: {str(error)[:120]}"
                failed.append(label)
            result = None
            if class_map is None:
                row["status"] = "no Sentinel-1 map"
            elif reference is None and row["status"] == "ok":
                row["status"] = "no Dynamic World image"
            elif reference is not None:
                # same footprint as the shares: only the counted pixels (radar in every available month)
                counted = np.where(common, class_map, IGNORE).astype(np.uint8)
                result = compare_maps(counted, reference, n_classes)
                if result is None:
                    row["status"] = "no pixel valid in both maps"
            if result is not None:
                row["compared area %"] = round(100 * result["compared_fraction"], 1)
                row["agreement %"] = round(100 * result["agreement"], 1)
                for k, name in enumerate(classes):
                    row[f"{name}: model %"] = round(result["pred_share"][k], 1)
                    row[f"{name}: Dynamic World %"] = round(result["ref_share"][k], 1)
            rows.append(row)
        dw_table = pd.DataFrame(rows)
        dw_table.to_csv(folder / "dynamic_world_comparison.csv", index=False)
        files.append(str(folder / "dynamic_world_comparison.csv"))
        if failed:
            summary.append(f"⚠️ **Dynamic World could not be downloaded for:** {', '.join(failed)} "
                           "(the Sentinel-1 results are not affected)")

    record = {
        "lat": lat, "lon": lon, "size_km": size_km, "start": labels[0], "end": labels[-1], "mode": mode,
        "model": info["model"], "model_dir": str(MODEL_DIR),
        "track": f"{grid['direction']} {grid['orbit']}", "scenes": [e["scenes"] for e in months],
        "track_coverage": round(float(grid.get("track_coverage", float("nan"))), 4),
        "cache_folder": grid["folder"], "months": labels,
        "grid": {key: grid[key] for key in ["crs", "transform", "width", "height", "direction", "orbit"]},
        "result_folder": str(folder),
        "months_with_map": len(labels) - len(missing), "missing_months": missing, "low_coverage_months": low,
        "counted_area_km2": round(area_km2, 2), "counted_fraction": round(float(common.mean()), 4),
        "prediction_seconds": round(prediction_seconds, 1), "dynamic_world": bool(compare_dw),
        "model_tag": model_tag(), "model_weights_sha256": weights_checksum(mode), "run_id": run_id,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    (folder / "request.json").write_text(json.dumps(record, indent=2))
    files.append(str(folder / "request.json"))

    outputs = ("\n\n".join(summary),
               map_figure(months, maps, classes, colors, lat, lon),
               share_figure(labels, shares, classes, colors),
               share_table, change_table, total_table, dw_table, files)
    return outputs, record


# ---------------------------------------------------------------- user interface

TABLE_NOTE = pd.DataFrame({"info": ["Dynamic World comparison was not requested for this run."]})


def clear_results(mode, compare_dw):
    # runs at once when the button is pressed: old results disappear, so they cannot be mixed up with the new request
    return (ui.status_working(mode, compare_dw), "", gr.DownloadButton(visible=False), ui.empty_plate(), "", "",
            gr.Accordion(visible=False), None, None, None, None)


def failed(message):
    return (ui.status_error(message), "", gr.DownloadButton(visible=False), ui.empty_plate(), "", "",
            gr.Accordion(visible=False), None, None, None, None)


def make_maps(lat, lon, start_year, start_month, size_km, compare_dw, mode):
    """The function behind the button. The request runs in a second thread, so the page can show which month is
    being predicted; then the page is built from the results of run_with_record."""
    started = time.time()
    state = {"fraction": 0.0, "text": "Searching Sentinel-1 images"}
    result = {}

    def progress(fraction, desc=""):
        state["fraction"], state["text"] = fraction, desc

    def work():
        try:
            result["value"] = run_with_record(lat, lon, start_year, start_month, size_km, compare_dw, mode, progress)
        except Exception as error:
            result["error"] = error

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    while worker.is_alive():
        yield (ui.status_working(mode, compare_dw, state, time.time() - started),) + (gr.skip(),) * 10
        worker.join(0.5)

    error = result.get("error")
    if isinstance(error, gr.Error):
        yield failed(getattr(error, "message", str(error)))
        return
    if error is not None:
        yield failed(f"Unexpected error: {type(error).__name__}: {error}")
        return

    outputs, record = result["value"]
    summary, map_fig, share_fig, share_table, change_table, total_table, dw_table, files = outputs
    plt.close(map_fig)
    plt.close(share_fig)
    info = get_model(record["mode"])[2]
    zip_path, n_files = ui.make_zip(record, files)
    size_mb = zip_path.stat().st_size / 1e6
    download = gr.DownloadButton(f"Download all results · {n_files} files · {size_mb:.1f} MB (.zip)",
                                 value=str(zip_path), visible=True)
    yield (ui.status_done(record, time.time() - started), ui.readout(record, info, dw_table), download,
           ui.map_plate(record, info, share_table), ui.change_plate(share_table, change_table, total_table, info),
           ui.dw_plate(dw_table), gr.Accordion(visible=True), share_table, change_table, total_table,
           dw_table if dw_table is not None else TABLE_NOTE)


def to_picker(lat, lon, size_km):
    try:
        return ui.picker_value(lat, lon, size_km)
    except (TypeError, ValueError):
        return gr.skip()


def from_picker(value):
    try:
        lat, lon = (round(float(part), 4) for part in str(value).split(",")[:2])
    except ValueError:
        return gr.skip(), gr.skip()
    return lat, lon


def build_app():
    current_year = datetime.date.today().year
    with gr.Blocks(title="The Radar Almanac · land cover from Sentinel-1", analytics_enabled=False) as app:
        gr.HTML(ui.masthead())

        with gr.Row(equal_height=True, elem_id="lc-request"):
            with gr.Column(scale=5, elem_classes="lc-form"):
                gr.HTML(ui.step("01", "Place", "click the map, search, type or pick an example"))
                with gr.Row():
                    lat = gr.Number(label="Latitude (south is negative)", value=49.4875)
                    lon = gr.Number(label="Longitude (west is negative)", value=8.4660)
                with gr.Row(elem_classes="lc-chips"):
                    examples = [(gr.Button(place[0], size="sm"), place) for place in ui.EXAMPLES]
                size_km = gr.Slider(label="Size of the square area (km)", minimum=2, maximum=MAX_SIZE_KM, step=1,
                                    value=10)

                gr.HTML(ui.step("02", "Period", "twelve months from the start"))
                with gr.Row():
                    start_year = gr.Dropdown(label="Start year", value="2021",
                                             choices=[str(y) for y in range(2015, current_year + 1)])
                    start_month = gr.Dropdown(label="Start month", choices=MONTH_NAMES, value="January")

                gr.HTML(ui.step("03", "Model"))
                mode = gr.Radio(show_label=False, choices=list(MODES.values()), value=MODES["standard"])
                compare_dw = gr.Checkbox(label="Also compare with Dynamic World (Sentinel-2 based, never used for "
                                               "training; slower)", value=False)
                button = gr.Button("Create 12 monthly maps", variant="primary", elem_id="lc-go")

            with gr.Column(scale=7, elem_classes="lc-mapcol"):
                picker = gr.HTML(ui.picker_value(49.4875, 8.4660, 10), html_template=ui.PICKER_HTML,
                                 js_on_load=ui.PICKER_JS, head=ui.LEAFLET_HEAD, elem_id="lc-picker")

        status = gr.HTML(ui.status_idle(), elem_id="lc-status")
        readout = gr.HTML("")
        download = gr.DownloadButton("Download all results", visible=False, elem_id="lc-download", variant="secondary")
        maps = gr.HTML(ui.empty_plate())
        change = gr.HTML("")
        dw = gr.HTML("")
        with gr.Accordion("Exact numbers · tables", open=True, visible=False, elem_id="lc-tables") as tables:
            with gr.Tab("Share per month (%)"):
                shares = gr.Dataframe()
            with gr.Tab("Month-to-month change (pp)"):
                month_changes = gr.Dataframe()
            with gr.Tab("Total change (pp)"):
                total = gr.Dataframe()
            with gr.Tab("Dynamic World"):
                dw_table = gr.Dataframe()
        gr.HTML(ui.footer())

        # the map and the coordinate fields follow each other: a click or a search on the map fills the fields;
        # typed coordinates move the map when the field is left or Enter is pressed (not at every key, so that
        # half-typed numbers like "12." are not changed while typing)
        picker.input(from_picker, picker, [lat, lon], queue=False, show_progress="hidden")
        for field in [lat, lon]:
            field.blur(to_picker, [lat, lon, size_km], picker, queue=False, show_progress="hidden")
            field.submit(to_picker, [lat, lon, size_km], picker, queue=False, show_progress="hidden")
        size_km.change(to_picker, [lat, lon, size_km], picker, queue=False, show_progress="hidden")
        for chip, (_, place_lat, place_lon, year, month) in examples:
            chip.click(lambda size, a=place_lat, b=place_lon, y=year, m=month:
                       (a, b, str(y), MONTH_NAMES[m - 1], ui.picker_value(a, b, size)),
                       size_km, [lat, lon, start_year, start_month, picker], queue=False, show_progress="hidden")

        # first clear the old results, then make the new ones (one request at a time: the loaded models,
        # the figures and the result files are shared)
        results = [status, readout, download, maps, change, dw, tables, shares, month_changes, total, dw_table]
        button.click(clear_results, [mode, compare_dw], results, queue=False, show_progress="hidden").then(
            make_maps, [lat, lon, start_year, start_month, size_km, compare_dw, mode], results,
            concurrency_limit=1, show_progress="hidden")
    return app


def start(share=False, port=7860, **launch_options):
    """Builds the app and starts it with the page style (theme, fonts, colours). Returns the running app."""
    app = build_app()
    app.launch(share=share, server_port=port, theme=ui.THEME, css=ui.CSS, head=ui.HEAD, js=ui.FORCE_LIGHT,
               **launch_options)
    return app


def main():
    global MODEL_DIR, PROJECT_ID
    parser = argparse.ArgumentParser(description="Sentinel-1 LULC prototype")
    parser.add_argument("--model-dir", default=MODEL_DIR, help="folder with lulc_model.keras, normalisation.json, model_info.json")
    parser.add_argument("--project", default=PROJECT_ID, help="Google Cloud project registered for Earth Engine")
    parser.add_argument("--share", action="store_true", help="create a temporary public link")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--mode", choices=list(MODES) + ["best"], default="standard",
                        help="model to load before the app starts")
    args = parser.parse_args()
    MODEL_DIR, PROJECT_ID = args.model_dir, args.project

    # log in to Earth Engine (may ask for a login once) and load the model, before the app starts
    init_earth_engine(PROJECT_ID, interactive=True)
    get_model(mode_key(args.mode))
    start(share=args.share, port=args.port)


if __name__ == "__main__":
    main()
