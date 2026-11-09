"""
Checks that the prototype works for places all over the world, through exactly the same code as the web app.

For every place, the app's run() is called and the outcome is sorted into one of these groups:
    complete      12 monthly maps
    partial       1-11 monthly maps (some months have no Sentinel-1 image on the chosen track)
    unavailable   clear message that there are no Sentinel-1 images (open sea, no recordings)
    rejected      clear message that the place is not supported (polar latitude, 180° meridian)
    error         any other message (for example a failed download or a model problem)
    crash         an unexpected Python error
For every place with maps, the returned tables are checked: 12 consecutive months, class shares adding up to 100 %,
and month-to-month / total changes equal to the differences of the shares.

    python prototype/world_check.py --model-dir models/round3 --mode standard

The script ends with exit code 1 if any request does not behave as expected (main() returns False).

Small areas (4 km) are used, so the check is quick.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import app
from lulc_core import month_list

# (name, lat, lon, start year, start month, expected outcome)
# "maps" = complete or partial; partial results are reported separately with the missing months
PLACES = [
    ("Tokyo",            35.6800,  139.7600, 2023, 1, "maps"),
    ("Lusaka",          -15.4200,   28.2800, 2021, 3, "maps"),
    ("Reykjavik",        64.1500,  -21.9400, 2022, 6, "maps"),
    ("Ushuaia",         -54.8000,  -68.3000, 2023, 1, "maps"),
    ("Sahara_Algeria",   25.0000,    5.0000, 2021, 1, "maps"),
    ("Amazon_Rondonia",  -8.0000,  -63.0000, 2022, 9, "maps"),
    ("Ulaanbaatar",      47.9200,  106.9200, 2021, 1, "maps"),
    ("Anchorage",        61.2200, -149.9000, 2023, 5, "maps"),
    ("Mumbai",           19.0700,   72.8800, 2023, 1, "maps"),
    ("Suva_Fiji",       -18.1400,  178.4400, 2023, 1, "maps"),
    ("Mannheim_2024",    49.4875,    8.4660, 2024, 6, "maps"),
    # no Sentinel-1 IW recordings in these 12 months (found in the first run of this check): after Sentinel-1B
    # failed in December 2021, Sentinel-1A alone did not record these regions; the app must say so clearly
    ("Lusaka_2023",     -15.4200,   28.2800, 2023, 3, "unavailable"),
    ("Sahara_2023",      25.0000,    5.0000, 2023, 1, "unavailable"),
    ("Ulaanbaatar_2023", 47.9200,  106.9200, 2023, 1, "unavailable"),
    ("Pacific_ocean",     0.0000, -140.0000, 2023, 1, "unavailable"),
    ("Dateline_Fiji",   -16.8000,  179.9900, 2023, 1, "rejected"),
    ("Svalbard",         78.2200,   15.6500, 2023, 1, "rejected"),
]

# the messages that mean "this place / period cannot be mapped" (and nothing else)
UNAVAILABLE_MESSAGES = ["No Sentinel-1 images"]
REJECTED_MESSAGES = ["Latitude must be between", "Longitude must be between", "crosses the 180° meridian"]

TOLERANCE = 0.05     # the tables are rounded to 2 decimals


def no_progress(*args, **kwargs):
    pass


def check_tables(year, month, share_table, change_table, total_table):
    """Returns a list of problems (empty if the tables are consistent)."""
    problems = []
    expected = [f"{y}-{m:02d}" for y, m in month_list(year, month)]
    if list(share_table["month"]) != expected:
        problems.append("month labels are not the 12 consecutive months")

    shares = share_table.drop(columns="month").to_numpy(dtype=float)
    with_map = ~np.isnan(shares).any(axis=1)
    if np.isnan(shares[~with_map]).sum() != shares[~with_map].size:
        problems.append("a month without map has some values")
    sums = shares[with_map].sum(axis=1)
    if len(sums) and np.abs(sums - 100).max() > TOLERANCE:
        problems.append(f"class shares add up to {sums.min():.2f}-{sums.max():.2f} %")

    month_changes = change_table.drop(columns="change").to_numpy(dtype=float)
    expected_changes = np.diff(shares, axis=0)
    if month_changes.shape != (11, shares.shape[1]):
        problems.append("there are not 11 month-to-month changes")
    elif not np.array_equal(np.isnan(month_changes), np.isnan(expected_changes)) or \
            np.nanmax(np.abs(month_changes - expected_changes), initial=0) > TOLERANCE:
        problems.append("month-to-month changes are not the differences of the shares")

    total = total_table.drop(columns="change").to_numpy(dtype=float)[0]
    expected_total = shares[-1] - shares[0]
    if not np.array_equal(np.isnan(total), np.isnan(expected_total)) or \
            np.nanmax(np.abs(total - expected_total), initial=0) > TOLERANCE:
        problems.append("total change is not last month minus first month")
    return problems


def check(lat, lon, year, month, size_km, mode):
    start = time.time()
    try:
        outputs, record = app.run_with_record(lat, lon, str(year), app.MONTH_NAMES[month - 1], size_km, False, mode)
        summary, *_, share_table, change_table, total_table, dw_table, files = outputs
    except gr.Error as error:
        message = getattr(error, "message", str(error))
        if any(text in message for text in REJECTED_MESSAGES):
            outcome = "rejected"
        elif any(text in message for text in UNAVAILABLE_MESSAGES):
            outcome = "unavailable"
        else:
            outcome = "error"
        return {"outcome": outcome, "details": message[:200], "seconds": round(time.time() - start, 1)}
    except Exception as error:
        return {"outcome": "crash", "details": f"{type(error).__name__}: {error}"[:200],
                "seconds": round(time.time() - start, 1)}
    finally:
        plt.close("all")

    problems = check_tables(year, month, share_table, change_table, total_table)
    shares = share_table.drop(columns="month")
    main_class = shares.mean().idxmax()
    n_maps = record["months_with_map"]
    details = (f"{n_maps}/12 months; main class {main_class} (mean share {shares.mean().max():.1f} %); "
               f"common radar area {record['counted_area_km2']:.1f} km² ({100 * record['counted_fraction']:.1f} % of the image)")
    if record["missing_months"]:
        details += f"; no image in {', '.join(record['missing_months'])}"
    if record["low_coverage_months"]:
        details += f"; low coverage in {', '.join(record['low_coverage_months'])}"
    if problems:
        details += "; TABLE PROBLEMS: " + "; ".join(problems)
    return {"outcome": "complete" if n_maps == 12 else "partial", "details": details,
            "table_problems": len(problems), "seconds": round(time.time() - start, 1),
            "prediction_seconds": record["prediction_seconds"]}


def main():
    parser = argparse.ArgumentParser(description="World-wide check of the prototype")
    parser.add_argument("--model-dir", default=app.MODEL_DIR)
    parser.add_argument("--project", default=app.PROJECT_ID)
    parser.add_argument("--size-km", type=float, default=4)
    parser.add_argument("--mode", choices=list(app.MODES), default="standard")
    args = parser.parse_args()
    app.MODEL_DIR, app.PROJECT_ID = args.model_dir, args.project

    app.init_earth_engine(app.PROJECT_ID, interactive=True)
    app.get_model(args.mode)
    print(f"mode: {args.mode} | model folder: {args.model_dir} | area: {args.size_km:g} km\n")

    rows = []
    for name, lat, lon, year, month, expected in PLACES:
        print(f"{name:<17} lat {lat:9.4f} lon {lon:10.4f} from {year}-{month:02d} ...", flush=True)
        row = {"place": name, "lat": lat, "lon": lon, "start": f"{year}-{month:02d}", "expected": expected,
               "table_problems": 0, **check(lat, lon, year, month, args.size_km, args.mode)}
        if expected == "maps":
            row["as_expected"] = row["outcome"] in ("complete", "partial") and row["table_problems"] == 0
        else:
            row["as_expected"] = row["outcome"] == expected
        print(f"   -> {row['outcome']}: {row['details']}  ({row['seconds']:.0f} s)", flush=True)
        rows.append(row)

    table = pd.DataFrame(rows)
    out = Path(f"data/prototype_results/world_check_{args.mode}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)

    counts = table["outcome"].value_counts()
    print("\noutcomes: " + ", ".join(f"{k} {int(counts.get(k, 0))}" for k in
                                     ["complete", "partial", "unavailable", "rejected", "error", "crash"]))
    print(f"table problems: {int(table['table_problems'].sum())}")
    print(f"{int(table['as_expected'].sum())} of {len(table)} places as expected")
    with_maps = table[table["outcome"].isin(["complete", "partial"])]
    if len(with_maps):
        print(f"prediction time ({args.mode}): {with_maps['prediction_seconds'].mean():.1f} s per place on average "
              f"(12 maps of {args.size_km:g} km)")
    for _, row in table[~table["as_expected"]].iterrows():
        print(f"NOT AS EXPECTED: {row['place']} -> {row['outcome']}: {row['details']}")
    print("saved:", out)
    return bool(table["as_expected"].all())


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
