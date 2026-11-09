"""
Look of the web app: colours, fonts, the location map and the HTML pieces that show the results.

The calculations are in app.py and lulc_core.py. This file only turns their results (the saved GeoTIFFs and the
tables) into a page, so the numbers on the page are exactly the numbers in the CSV files.
The style follows printed map sheets: paper colour, thin lines, coordinates in the corners, a scale bar.
"""
import base64
import html
import io
import math
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform
from PIL import Image
import gradio as gr

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
IGNORE = 255

PAPER = "#f2eee3"
SHEET = "#faf8f2"
INK = "#1b2420"
MUTED = "#5f655f"
RULE = "#d6cfbd"
RED = "#b5381c"       # survey red, the only accent colour
BLUE = "#2d5a78"      # for decreases in the change tables

MODEL_NAMES = {"segformer_b3": "SegFormer-B3", "segformer_b2": "SegFormer-B2",
               "deeplabv3plus_resnet50": "DeepLabV3+ (ResNet-50)", "unet_resnet50": "U-Net (ResNet-50)"}

# places with a known good result in the world check (prototype/world_check.py)
EXAMPLES = [
    ("Mannheim", 49.4875, 8.4660, 2021, 1),
    ("Tokyo", 35.6800, 139.7600, 2023, 1),
    ("Mumbai", 19.0700, 72.8800, 2023, 1),
    ("Lusaka", -15.4200, 28.2800, 2021, 3),
    ("Reykjavík", 64.1500, -21.9400, 2022, 6),
    ("Rondônia, Amazon", -8.0000, -63.0000, 2022, 9),
]

# ---------------------------------------------------------------- theme, fonts, page style

THEME = gr.themes.Base(
    font=[gr.themes.GoogleFont("IBM Plex Sans"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
    radius_size=gr.themes.sizes.radius_none,
).set(
    body_background_fill=PAPER,
    body_text_color=INK,
    body_text_color_subdued=MUTED,
    background_fill_primary=SHEET,
    background_fill_secondary=PAPER,
    block_background_fill=SHEET,
    block_border_color=RULE,
    block_border_width="1px",
    block_shadow="none",
    block_label_background_fill=SHEET,
    block_label_text_color=MUTED,
    block_title_text_color=MUTED,
    panel_background_fill=SHEET,
    border_color_primary=RULE,
    border_color_accent=RED,
    color_accent=RED,
    color_accent_soft="#f3dcd3",
    input_background_fill="#fffdf8",
    input_border_color=RULE,
    input_border_color_focus=INK,
    shadow_drop="none",
    button_primary_background_fill=INK,
    button_primary_background_fill_hover=RED,
    button_primary_border_color=INK,
    button_primary_text_color=PAPER,
    button_secondary_background_fill="#fffdf8",
    button_secondary_border_color=RULE,
    slider_color=RED,
    loader_color=RED,
    checkbox_background_color_selected=RED,
    checkbox_border_color_selected=RED,
    checkbox_label_background_fill_selected="#fffdf8",
    checkbox_label_border_color_selected=INK,
    checkbox_label_text_color_selected=INK,
    table_even_background_fill=SHEET,
    table_odd_background_fill="#f5f2e9",
)

HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;1,6..72,400&display=swap">
"""

# Leaflet for the location map (loaded by the browser; the hashes make sure the file is the original one)
LEAFLET_HEAD = """
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
      integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin="">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
        integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
"""

# the page is designed for a light background: open it with ?__theme=light (Gradio otherwise follows the
# dark mode of the computer)
FORCE_LIGHT = """
() => {
  const url = new URL(window.location.href);
  if (url.searchParams.get("__theme") !== "light") {
    url.searchParams.set("__theme", "light");
    window.location.replace(url.href);
  }
}
"""

CSS = f"""
:root {{ --paper: {PAPER}; --sheet: {SHEET}; --ink: {INK}; --muted: {MUTED}; --rule: {RULE};
         --red: {RED}; --blue: {BLUE}; --serif: "Newsreader", Georgia, serif;
         --mono: "IBM Plex Mono", ui-monospace, monospace; }}
body, gradio-app {{
  background-color: var(--paper) !important;
  background-image: linear-gradient(rgba(27,36,32,.035) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(27,36,32,.035) 1px, transparent 1px) !important;
  background-size: 28px 28px !important;
}}
.html-container {{ padding: 0 !important; min-width: 0 !important; max-width: 100%; }}
.html-container > .prose, #lc-status, .lc-plate {{ min-width: 0; max-width: 100%; }}
.gradio-container {{ width: 100% !important; max-width: 1240px !important; margin: 0 auto !important; padding: 0 20px !important;
                     background: transparent !important; }}
@media (max-width: 600px) {{ .gradio-container {{ padding: 0 4px !important; }} }}
.gradio-container, .gradio-container > .main, .gradio-container > .main > .wrap, .gradio-container main.contain {{ min-width: 0 !important; }}
body > gradio-app footer, .gradio-container > footer {{ display: none !important; }}

/* masthead */
.lc-mast {{ border-top: 3px solid var(--ink); border-bottom: 1px solid var(--ink); margin: 28px 0 22px;
            padding: 10px 0 20px; position: relative; }}
.lc-mast::before {{ content: ""; position: absolute; left: 0; right: 0; top: 3px; border-top: 1px solid var(--ink); }}
.lc-kicker {{ display: flex; justify-content: space-between; flex-wrap: wrap; gap: 6px 24px; font-family: var(--mono);
              font-size: 11px; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); margin-top: 6px; }}
.lc-mast h1 {{ font-family: var(--serif); font-weight: 400; font-size: clamp(38px, 6vw, 68px); line-height: .98;
               letter-spacing: -.02em; margin: 22px 0 14px; color: var(--ink); }}
.lc-mast h1 em {{ font-style: italic; color: var(--red); }}
.lc-sub-title {{ font-family: var(--serif); font-style: italic; font-size: clamp(20px, 2.6vw, 28px); line-height: 1.2;
                 margin: -4px 0 16px; color: var(--muted); }}
.lc-deck {{ font-family: var(--serif); font-size: 19px; line-height: 1.45; max-width: 760px; margin: 0 0 18px;
            color: var(--ink); }}
.lc-facts {{ display: flex; flex-wrap: wrap; gap: 0; border-top: 1px solid var(--rule); padding-top: 10px; margin: 0; }}
.lc-facts div {{ padding: 0 18px 0 0; margin-right: 18px; border-right: 1px solid var(--rule); }}
.lc-facts div:last-child {{ border-right: none; }}
.lc-facts dt {{ font-family: var(--mono); font-size: 10px; letter-spacing: .12em; text-transform: uppercase;
                color: var(--muted); }}
.lc-facts dd {{ margin: 2px 0 0; font-size: 14px; }}

/* request form */
#lc-request {{ gap: 22px; align-items: stretch; }}
.lc-form {{ background: var(--sheet); border: 1px solid var(--ink); padding: 18px 18px 20px !important; }}
.lc-step {{ display: flex; align-items: baseline; gap: 10px; border-bottom: 1px solid var(--rule);
            padding-bottom: 6px; margin: 4px 0 2px; }}
.lc-step b {{ font-family: var(--mono); font-size: 12px; color: var(--red); font-weight: 500; }}
.lc-step span {{ font-family: var(--serif); font-size: 22px; }}
.lc-step i {{ font-style: normal; color: var(--muted); font-size: 13px; margin-left: auto; text-align: right; }}
.lc-chips {{ gap: 6px !important; flex-wrap: wrap !important; }}
.lc-chips button {{ flex: 0 0 auto !important; min-width: 0 !important; font-family: var(--mono) !important;
                    font-size: 12px !important; padding: 3px 10px !important; border: 1px solid var(--rule) !important;
                    border-radius: 999px !important; background: transparent !important; color: var(--ink) !important; }}
.lc-chips button:hover {{ border-color: var(--red) !important; color: var(--red) !important; }}
#lc-go {{ font-family: var(--mono) !important; letter-spacing: .08em; text-transform: uppercase; font-size: 14px !important;
          padding: 14px !important; margin-top: 6px; }}
.lc-form label span, .lc-form .label-wrap span {{ font-size: 12px; }}

.lc-form .block, .lc-form .form {{ border: none !important; background: transparent !important; box-shadow: none !important; }}
.lc-form .form {{ gap: 0 !important; }}

/* location map */
.lc-mapcol {{ display: flex !important; }}
#lc-picker {{ flex: 1 1 auto; height: 100%; display: flex; }}
#lc-picker .html-container, #lc-picker .prose {{ flex: 1; display: flex; padding: 0 !important; margin: 0 !important; }}
.lc-pick {{ flex: 1; width: 100%; display: flex; flex-direction: column; border: 1px solid var(--ink); background: var(--sheet); }}
.lc-pick-map {{ flex: 1; min-height: 470px; background: #dfe3dc; }}
.lc-pick-foot {{ display: flex; flex-direction: column; gap: 3px; padding: 8px 12px 9px; font-size: 12px;
                 color: var(--muted); border-top: 1px solid var(--ink); }}
.lc-pick-foot b {{ color: var(--ink); font-family: var(--mono); font-weight: 500; }}
.lc-pick-name {{ font-family: var(--serif); font-size: 17px; color: var(--ink); line-height: 1.3; min-height: 22px; }}
.lc-pick-note {{ font-size: 11px; }}
.lc-pick-label {{ font-family: var(--mono); font-size: 10px; letter-spacing: .12em; text-transform: uppercase; }}
.lc-search-msg {{ display: none; padding: 7px 12px; font-size: 12px; color: var(--red); background: #f6e9e2;
                  border-bottom: 1px solid var(--ink); }}
.lc-search {{ display: flex; border-bottom: 1px solid var(--ink); margin: 0; }}
.lc-search input {{ flex: 1; min-width: 0; border: none !important; background: #fffdf8; padding: 10px 12px; font-size: 14px;
                    font-family: "IBM Plex Sans", system-ui, sans-serif; color: var(--ink); outline: none; box-shadow: none; }}
.lc-search input:focus {{ background: #fff; }}
.lc-search button {{ border: none; border-left: 1px solid var(--ink); background: var(--ink); color: var(--paper);
                     font-family: var(--mono); font-size: 12px; letter-spacing: .1em; text-transform: uppercase;
                     padding: 0 18px; cursor: pointer; }}
.lc-search button:hover {{ background: var(--red); }}
.lc-pick .leaflet-container {{ font-family: "IBM Plex Sans", system-ui, sans-serif; cursor: crosshair; }}
.lc-pick .leaflet-control-attribution {{ font-size: 10px; }}

/* status line */
#lc-status {{ margin-top: 26px; }}
.lc-status {{ display: flex; align-items: center; gap: 14px; border-top: 1px solid var(--ink); border-bottom: 1px solid var(--ink);
              padding: 12px 2px; font-size: 14px; }}
.lc-status .lc-dot {{ width: 9px; height: 9px; border-radius: 50%; background: var(--muted); flex: none; }}
.lc-status.working .lc-dot {{ background: var(--red); animation: lc-pulse 1.1s ease-in-out infinite; }}
.lc-status.done .lc-dot {{ background: #2e6b3a; }}
.lc-status.error {{ border-color: var(--red); color: var(--red); }}
.lc-status.error .lc-dot {{ background: var(--red); }}
.lc-status code {{ font-family: var(--mono); font-size: 12px; color: var(--muted); background: none; word-break: break-all; }}
@keyframes lc-pulse {{ 50% {{ opacity: .25; }} }}
.lc-status small {{ color: var(--muted); font-size: 12px; margin-left: 6px; }}
.lc-progress {{ display: flex; gap: 3px; margin-left: auto; }}
.lc-progress i {{ width: 12px; height: 12px; border: 1px solid var(--ink); display: block; }}
.lc-progress i.on {{ background: var(--ink); }}
@media (max-width: 600px) {{ .lc-status {{ flex-wrap: wrap; }} .lc-progress {{ margin-left: 23px; }} }}

/* readout (sheet data) */
.lc-readout {{ display: grid; grid-template-columns: repeat(6, 1fr); border: 1px solid var(--ink); background: var(--sheet);
               margin: 18px 0 0; }}
.lc-cell {{ padding: 12px 14px; border-right: 1px solid var(--rule); min-width: 0; }}
.lc-cell:last-child {{ border-right: none; }}
.lc-k {{ display: block; font-family: var(--mono); font-size: 10px; letter-spacing: .12em; text-transform: uppercase;
         color: var(--muted); }}
.lc-v {{ display: block; font-family: var(--serif); font-size: 22px; line-height: 1.15; margin: 4px 0 3px; }}
.lc-s {{ display: block; font-size: 12px; color: var(--muted); line-height: 1.35; }}
.lc-spark {{ display: flex; align-items: flex-end; gap: 2px; height: 30px; margin: 6px 0 3px; }}
.lc-spark span {{ flex: 1; background: var(--ink); min-height: 2px; position: relative; }}
.lc-spark span.none {{ background: none; border-bottom: 2px solid var(--red); }}
.lc-notes {{ margin: 10px 0 0; padding: 0; list-style: none; }}
.lc-notes li {{ border-left: 3px solid var(--red); background: #f6e9e2; padding: 7px 12px; margin-top: 6px; font-size: 13px; }}
@media (max-width: 1000px) {{ .lc-readout {{ grid-template-columns: repeat(3, 1fr); }}
  .lc-cell:nth-child(3) {{ border-right: none; }} .lc-cell:nth-child(-n+3) {{ border-bottom: 1px solid var(--rule); }} }}
@media (max-width: 600px) {{ .lc-readout {{ grid-template-columns: repeat(2, 1fr); }}
  .lc-cell {{ border-right: 1px solid var(--rule) !important; border-bottom: 1px solid var(--rule); }}
  .lc-cell:nth-child(2n) {{ border-right: none !important; }} }}

#lc-download {{ margin-top: 12px; width: auto !important; max-width: max-content; align-self: flex-start;
                 font-family: var(--mono) !important; font-size: 13px !important; padding: 10px 18px !important;
                 border: 1px solid var(--ink) !important; background: var(--sheet) !important; }}
#lc-download:hover {{ background: var(--ink) !important; color: var(--paper) !important; }}

/* plates */
.lc-plate {{ margin: 34px 0 0; }}
.lc-plate-head {{ display: flex; align-items: baseline; gap: 14px; border-bottom: 1px solid var(--ink);
                  padding-bottom: 6px; margin-bottom: 14px; flex-wrap: wrap; }}
.lc-plate-head b {{ font-family: var(--mono); font-size: 11px; letter-spacing: .14em; text-transform: uppercase;
                    color: var(--red); font-weight: 500; }}
.lc-plate-head h2 {{ font-family: var(--serif); font-weight: 400; font-size: 30px; margin: 0; line-height: 1.1; }}
.lc-plate-head span {{ margin-left: auto; font-size: 13px; color: var(--muted); }}
.lc-collar {{ display: flex; justify-content: space-between; font-family: var(--mono); font-size: 11px; color: var(--muted);
              margin: 0 0 6px; gap: 10px; flex-wrap: wrap; }}
.lc-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px 16px; }}
@media (max-width: 900px) {{ .lc-grid {{ grid-template-columns: repeat(3, 1fr); }} }}
@media (max-width: 600px) {{ .lc-grid {{ grid-template-columns: repeat(2, 1fr); gap: 14px 10px; }} }}
.lc-tile {{ margin: 0; min-width: 0; }}
.lc-tile figcaption {{ display: flex; justify-content: space-between; align-items: baseline; font-size: 12px;
                       color: var(--muted); margin-bottom: 4px; gap: 6px; }}
.lc-tile figcaption b {{ font-family: var(--mono); font-size: 13px; color: var(--ink); letter-spacing: .06em;
                         text-transform: uppercase; font-weight: 600; }}
.lc-tile figcaption em {{ font-style: normal; color: var(--red); }}
.lc-img {{ aspect-ratio: 1 / 1; border: 1px solid var(--ink); display: flex; align-items: center; justify-content: center;
           background: repeating-linear-gradient(135deg, transparent 0 6px, rgba(27,36,32,.09) 6px 7px), var(--sheet); }}
.lc-img img {{ width: 100%; height: 100%; object-fit: contain; display: block; }}
.lc-img p {{ margin: 0; padding: 12px; text-align: center; font-size: 12px; color: var(--muted); background: var(--sheet); }}
.lc-tile.empty .lc-img {{ border-color: var(--rule); }}
.lc-grid.empty {{ grid-template-columns: repeat(6, 1fr); gap: 12px; }}
@media (max-width: 700px) {{ .lc-grid.empty {{ grid-template-columns: repeat(4, 1fr); }} }}
.lc-plate-foot {{ display: grid; grid-template-columns: 3fr 1fr; gap: 16px; margin-top: 18px; align-items: start; }}
@media (max-width: 900px) {{ .lc-plate-foot {{ grid-template-columns: 1fr; }} }}
.lc-legend {{ display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 13px; }}
.lc-legend span {{ display: inline-flex; align-items: center; gap: 7px; }}
.lc-legend i {{ width: 14px; height: 14px; border: 1px solid var(--ink); display: inline-block; flex: none; }}
.lc-legend small {{ font-family: var(--mono); color: var(--muted); font-size: 11px; }}
.lc-scale {{ font-family: var(--mono); font-size: 11px; color: var(--muted); }}
.lc-scale .bar {{ display: flex; height: 6px; border: 1px solid var(--ink); margin: 4px 0 2px; }}
.lc-scale .bar span {{ flex: 1; }} .lc-scale .bar span:first-child {{ background: var(--ink); }}
.lc-scale .ticks {{ display: flex; justify-content: space-between; }}

/* change plate */
.lc-two {{ display: grid; grid-template-columns: 5fr 7fr; gap: 28px; align-items: start; }}
@media (max-width: 900px) {{ .lc-two {{ grid-template-columns: 1fr; }} }}
.lc-sub {{ font-family: var(--mono); font-size: 11px; letter-spacing: .12em; text-transform: uppercase; color: var(--muted);
           margin: 0 0 10px; }}
.lc-bars {{ display: grid; grid-template-columns: minmax(120px, auto) 1fr 64px; gap: 7px 10px; align-items: center;
            font-size: 13px; }}
.lc-bars .name {{ display: flex; align-items: center; gap: 7px; }}
.lc-bars .name i {{ width: 11px; height: 11px; border: 1px solid var(--ink); flex: none; }}
.lc-bars .track {{ position: relative; height: 16px; }}
.lc-bars .track::before {{ content: ""; position: absolute; left: 50%; top: -3px; bottom: -3px; border-left: 1px solid var(--ink); }}
.lc-bars .track span {{ position: absolute; top: 2px; height: 12px; border: 1px solid var(--ink); }}
.lc-bars .val {{ font-family: var(--mono); text-align: right; font-size: 12px; }}
.lc-bars .from {{ grid-column: 2 / 4; font-size: 11px; color: var(--muted); margin-top: -6px; }}
.lc-chart svg {{ width: 100%; height: auto; display: block; overflow: visible; }}
.lc-heat {{ width: 100%; border-collapse: collapse; font-family: var(--mono); font-size: 11px; margin: 6px 0 0 !important; }}
.lc-heat, .lc-heat tr {{ border: none !important; }}
.lc-heat td, .lc-heat th {{ border-top: none !important; border-left: none !important; border-right: none !important; }}
.lc-heat td:first-child, .lc-heat th:first-child {{ width: 1%; }}
.lc-heat th {{ font-weight: 500; color: var(--muted); padding: 4px 2px; text-align: center; border-bottom: 1px solid var(--ink); }}
.lc-heat th:first-child, .lc-heat td:first-child {{ text-align: left; font-family: "IBM Plex Sans", sans-serif;
                                                    font-size: 12px; color: var(--ink); white-space: nowrap; padding-right: 10px; }}
.lc-heat td {{ text-align: center; padding: 5px 2px; border-bottom: 1px solid var(--rule); }}
.lc-scroll {{ overflow-x: auto; width: 0; min-width: 100%; }}
.lc-aside {{ font-family: var(--serif); font-style: italic; font-size: 15px; line-height: 1.45; color: var(--muted);
             border-left: 1px solid var(--ink); padding-left: 14px; margin: 18px 0 0; max-width: 820px; }}

/* Dynamic World plate */
.lc-strip {{ display: grid; grid-template-columns: repeat(12, 1fr); border: 1px solid var(--ink); }}
@media (max-width: 900px) {{ .lc-strip {{ grid-template-columns: repeat(6, 1fr); }} }}
.lc-strip div {{ padding: 8px 6px 9px; border-right: 1px solid var(--rule); min-height: 74px; }}
.lc-strip div:last-child {{ border-right: none; }}
.lc-strip b {{ display: block; font-family: var(--mono); font-size: 11px; letter-spacing: .06em; text-transform: uppercase; }}
.lc-strip strong {{ display: block; font-family: var(--serif); font-weight: 400; font-size: 24px; line-height: 1.2; margin-top: 4px; }}
.lc-strip small {{ display: block; font-size: 11px; color: var(--muted); line-height: 1.25; }}
.lc-strip .off strong {{ font-size: 14px; color: var(--muted); margin-top: 8px; }}
.lc-empty {{ border: 1px dashed var(--rule); padding: 16px 18px; color: var(--muted); font-size: 14px; background: var(--sheet); }}

/* tables and footer */
#lc-tables {{ margin-top: 34px; border: 1px solid var(--ink) !important; min-width: 0 !important; max-width: 100%; }}
#lc-tables .tabs, #lc-tables .tabitem, #lc-tables .block, #lc-tables .table-wrap {{ min-width: 0 !important; max-width: 100%; }}
#lc-tables .table-wrap {{ overflow-x: auto; }}
#lc-tables .tab-wrapper, #lc-tables .tab-container, #lc-tables .header-row {{ min-width: 0 !important; flex-wrap: wrap; }}
#lc-tables .label-wrap span, #lc-tables > button span {{ font-family: var(--mono); letter-spacing: .1em; text-transform: uppercase; font-size: 12px; }}
.lc-foot {{ margin: 48px 0 30px; border-top: 3px double var(--ink); padding-top: 14px; display: grid;
            grid-template-columns: 2fr 1fr; gap: 28px; font-size: 12px; color: var(--muted); line-height: 1.55; }}
@media (max-width: 800px) {{ .lc-foot {{ grid-template-columns: 1fr; }} }}
.lc-foot h4 {{ font-family: var(--mono); font-weight: 500; font-size: 10px; letter-spacing: .14em; text-transform: uppercase;
               color: var(--ink); margin: 0 0 4px; }}
.lc-foot p {{ margin: 0 0 10px; }}
"""

# ---------------------------------------------------------------- location map (Leaflet inside a gr.HTML)

PICKER_HTML = """
<div class="lc-pick">
  <form class="lc-search"><input type="search" placeholder="Search a place by name, e.g. Lalibela, Ethiopia" aria-label="Search a place">
    <button type="submit">Find</button></form>
  <div class="lc-search-msg"></div>
  <div class="lc-pick-map"></div>
  <div class="lc-pick-foot">
    <span class="lc-pick-label">Place at the centre of the frame</span>
    <span class="lc-pick-name">&nbsp;</span>
    <span class="lc-pick-status">Click the map, search a place or type the coordinates. The red frame is the area that
    will be mapped.</span>
    <span class="lc-pick-note">Background map: location reference only, not model input and not current imagery &middot; place search and
    names: Nominatim, &copy; OpenStreetMap contributors</span>
  </div>
</div>
"""

PICKER_JS = """
const host = element.querySelector(".lc-pick-map");
const status = element.querySelector(".lc-pick-status");
const placeName = element.querySelector(".lc-pick-name");
const form = element.querySelector(".lc-search");
const searchBox = form.querySelector("input");
const searchMsg = element.querySelector(".lc-search-msg");

function searchMessage(text) {
  searchMsg.textContent = text || "";
  searchMsg.style.display = text ? "block" : "none";
}
let map = null, frame = null, centre = null;
let nameTimer = null, lastName = "", request = 0;

function current() {
  const parts = String(props.value || "").split(",").map(Number);
  return {lat: parts[0], lon: parts[1], size: parts[2]};
}

function frameBounds(p) {
  // same square as the app: size x size km around the centre
  const half = p.size * 500;
  const dLat = half / 111320;
  const dLon = half / (111320 * Math.cos(p.lat * Math.PI / 180));
  return L.latLngBounds([p.lat - dLat, p.lon - dLon], [p.lat + dLat, p.lon + dLon]);
}

function showName(p) {
  // name of the place at the centre (OpenStreetMap Nominatim); waits a second, so it asks at most once per second
  const key = p.lat.toFixed(3) + "," + p.lon.toFixed(3);
  if (key === lastName) return;
  lastName = key;
  clearTimeout(nameTimer);
  nameTimer = setTimeout(() => {
    const id = ++request;
    placeName.textContent = "looking up the place name ...";
    fetch("https://nominatim.openstreetmap.org/reverse?format=jsonv2&zoom=10&accept-language=en&lat=" + p.lat + "&lon=" + p.lon)
      .then((response) => response.json())
      .then((data) => {
        if (id !== request) return;
        placeName.textContent = data && data.display_name ? data.display_name : "no place name here (open sea or a remote area)";
      })
      .catch(() => { if (id === request) placeName.textContent = "place name not available (no connection to OpenStreetMap)"; });
  }, 1000);
}

function draw(zoomTo) {
  const p = current();
  if (!map || !isFinite(p.lat) || !isFinite(p.lon) || !isFinite(p.size) || Math.abs(p.lat) > 84) return;
  const bounds = frameBounds(p);
  if (frame) {
    frame.setBounds(bounds);
    centre.setLatLng([p.lat, p.lon]);
  } else {
    frame = L.rectangle(bounds, {color: "#b5381c", weight: 2, fill: false, interactive: false}).addTo(map);
    centre = L.circleMarker([p.lat, p.lon], {radius: 3, color: "#b5381c", fillColor: "#b5381c",
                                             fillOpacity: 1, interactive: false}).addTo(map);
  }
  if (zoomTo || !map.getBounds().contains(bounds)) map.fitBounds(bounds.pad(0.8));
  status.innerHTML = "Centre <b>" + p.lat.toFixed(4) + ", " + p.lon.toFixed(4) + "</b> &middot; area <b>" +
                     p.size + " &times; " + p.size + " km</b> &middot; click the map to move it";
  showName(p);
}

function choose(lat, lon, zoomTo) {
  // new centre from the map or the search: tell the app (the latitude and longitude fields follow)
  const p = current();
  props.value = [lat.toFixed(4), lon.toFixed(4), isFinite(p.size) ? p.size : 10].join(",");
  draw(zoomTo);
  trigger("input");
}

function start() {
  map = L.map(host, {worldCopyJump: true, minZoom: 2});
  const satellite = L.tileLayer(
    "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2021_3857/default/GoogleMapsCompatible/{z}/{y}/{x}.jpg",
    {maxZoom: 17, attribution: '<a href="https://cloudless.eox.at" target="_blank">EOxCloudless</a> by EOX IT Services GmbH ' +
                               '(Contains modified Copernicus Sentinel data 2021)'});
  const streets = L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    {maxZoom: 19, attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank">OpenStreetMap</a> contributors'});
  satellite.addTo(map);
  L.control.layers({"Satellite photo 2021 (location reference only)": satellite, "Streets (OpenStreetMap)": streets}).addTo(map);
  L.control.scale({imperial: false}).addTo(map);
  map.setView([20, 0], 2);
  draw(true);
  map.attributionControl.setPrefix('<a href="https://leafletjs.com" target="_blank">Leaflet</a>');
  map.on("click", (event) => {
    const point = event.latlng.wrap();
    searchBox.value = "";
    searchMessage("");
    choose(point.lat, point.lng, false);
  });
  new ResizeObserver(() => map.invalidateSize()).observe(host);
}

function searchTexts(text) {
  // the text as typed (without notes in brackets), then shorter versions:
  // "Sundarbans mangroves, Bangladesh" -> "Sundarbans mangroves" -> "Sundarbans, Bangladesh"
  const texts = [];
  const add = (t) => {
    t = t.replace(/\\s+/g, " ").replace(/ ,/g, ",").trim().replace(/^,|,$/g, "").trim();
    if (t && !texts.includes(t)) texts.push(t);
  };
  const clean = text.replace(/\\([^)]*\\)/g, " ");
  const parts = clean.split(",");
  const words = parts[0].trim().split(" ");
  add(clean);
  add(parts[0]);
  if (words.length > 1 && parts.length > 1) add([words.slice(0, -1).join(" ")].concat(parts.slice(1)).join(","));
  return texts;
}

async function findPlace(texts, id) {
  // asks Nominatim for one text after the other (at most one request per second) until a place is found
  for (let i = 0; i < texts.length; i++) {
    if (i > 0) await new Promise((resolve) => setTimeout(resolve, 1100));
    if (id !== request) return null;
    const response = await fetch("https://nominatim.openstreetmap.org/search?format=jsonv2&limit=1&accept-language=en&q=" +
                                 encodeURIComponent(texts[i]));
    const results = await response.json();
    if (results.length) return results[0];
  }
  return undefined;
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const query = searchBox.value.trim();
  if (!query || !map) return;
  clearTimeout(nameTimer);          // a name lookup for the old place must not overwrite the search result
  const id = ++request;
  searchMessage("Searching \\u201c" + query + "\\u201d ...");
  findPlace(searchTexts(query), id)
    .then((found) => {
      if (id !== request || found === null) return;
      if (found === undefined) {
        searchMessage("\\u201c" + query + "\\u201d was not found, the place was not changed. Try the name of a town " +
                      "or region, or click the map.");
        return;
      }
      const lat = Number(found.lat), lon = Number(found.lon);
      lastName = lat.toFixed(3) + "," + lon.toFixed(3);
      placeName.textContent = found.display_name;
      searchBox.value = "";
      searchMessage("");
      choose(lat, lon, true);
    })
    .catch(() => {
      if (id === request) searchMessage("The search is not available right now (no connection to OpenStreetMap). Type the coordinates instead.");
    });
});

// the app changed the place (typed coordinates, example buttons, area size): move the frame
watch("value", () => {
  // typed coordinates, example buttons: an old search text no longer fits the place
  const p = current();
  if (isFinite(p.lat) && isFinite(p.lon) && p.lat.toFixed(3) + "," + p.lon.toFixed(3) !== lastName) {
    searchBox.value = "";
    searchMessage("");
  }
  draw(false);
});

let tries = 0;
(function waitForLeaflet() {
  if (window.L) { start(); return; }
  if (++tries > 150) {
    status.textContent = "The map could not be loaded (no connection to unpkg.com). Type the coordinates instead.";
    return;
  }
  setTimeout(waitForLeaflet, 100);
})();
"""


def picker_value(lat, lon, size_km):
    return f"{float(lat):.4f},{float(lon):.4f},{float(size_km):g}"


# ---------------------------------------------------------------- small helpers

def esc(text):
    return html.escape(str(text))


def month_label(label):
    # "2021-03" -> "Mar 2021"
    year, month = label.split("-")
    return f"{MONTHS[int(month) - 1]} {year}"


def degrees(value, positive, negative):
    # 49.4875 -> 49°29.3′ N
    letter = positive if value >= 0 else negative
    value = abs(value)
    whole = int(value)
    return f"{whole}°{(value - whole) * 60:04.1f}′ {letter}"


def line_color(hex_color):
    # very light class colours (yellow, pink, grey) are hard to see as thin lines on paper: darken them a bit
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    if luminance < 0.4:
        return hex_color
    ink = (0x1b / 255, 0x24 / 255, 0x20 / 255)
    mixed = [0.6 * c + 0.4 * k for c, k in zip((r, g, b), ink)]
    return "#" + "".join(f"{round(255 * c):02x}" for c in mixed)


def model_name(record, info):
    if record["mode"] == "ensemble":
        return "Ensemble", "4 models × 8 flips/rotations"
    return MODEL_NAMES.get(info["model"], info["model"]), "selected single model"


def map_png(class_map, colors, max_pixels=560):
    # small PNG of a class map for the page (nearest neighbour, no new colours); no-data pixels are transparent
    step = max(1, math.ceil(max(class_map.shape) / max_pixels))
    small = np.ascontiguousarray(class_map[::step, ::step])
    image = Image.frombytes("P", (small.shape[1], small.shape[0]), small.tobytes())
    palette = [0] * (256 * 3)
    for k, color in enumerate(colors):
        palette[3 * k:3 * k + 3] = [int(color[i:i + 2], 16) for i in (1, 3, 5)]
    image.putpalette(palette)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", transparency=IGNORE, optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


# ---------------------------------------------------------------- fixed parts of the page

def masthead():
    return """
<header class="lc-mast">
  <div class="lc-kicker"><span>Sentinel-1 &middot; C-band radar &middot; VV + VH &middot; 10 m</span>
       <span>Ishwari Thakur &middot; Deep Learning &middot; SRH Hochschule Heidelberg &middot; 2026</span></div>
  <h1>The Radar <em>Almanac</em></h1>
  <p class="lc-sub-title">Land cover, month by month, from Sentinel-1</p>
  <p class="lc-deck">Pick a place and a start month. The app downloads twelve monthly Sentinel-1 radar images, and a
  deep learning model draws a land cover map for each month and measures how the classes change. Radar sees through
  clouds and works at night, so rainy and winter months get a map too. All maps and numbers are <i>predictions</i>
  of the model.</p>
  <dl class="lc-facts">
    <div><dt>Where</dt><dd>60° S to 75° N, not across 180°</dd></div>
    <div><dt>Area</dt><dd>2 to 16 km square</dd></div>
    <div><dt>Classes</dt><dd>7 classes of ESA WorldCover</dd></div>
    <div><dt>Not mapped</dt><dd>snow and ice</dd></div>
  </dl>
</header>"""


def step(number, title, hint=""):
    return f'<div class="lc-step"><b>{number}</b><span>{esc(title)}</span><i>{esc(hint)}</i></div>'


def footer():
    return """
<div class="lc-foot">
  <div>
    <h4>How it works</h4>
    <p>For every month, all Sentinel-1 IW images of one orbit track are combined into a median image (VV and VH,
    10 m pixels). The model was trained on ESA WorldCover 2021 labels and gets only these two radar bands. Shares and
    changes are counted only on pixels that have radar data in every available month, so a gap in the data cannot
    look like a change. Changes are given in percentage points (pp): 30 % &rarr; 32 % is +2 pp.</p>
    <h4>Limits</h4>
    <p>Small month-to-month changes can come from seasonal radar effects (leaves, soil moisture, frost) and from
    model errors, not only from real land cover change. Snow and ice are not among the classes. Sentinel-1B stopped
    in December 2021 and Sentinel-1C only started in 2025, so months in between have fewer images and noisier maps.</p>
  </div>
  <div>
    <h4>Data</h4>
    <p>Contains modified Copernicus Sentinel data, processed with Google Earth Engine.<br>
    Training labels: ESA WorldCover 2021 (CC BY 4.0).<br>
    Reference: Dynamic World V1, Google &amp; World Resources Institute (CC BY 4.0).<br>
    Location map: EOxCloudless 2021 by EOX IT Services GmbH (CC BY-NC-SA 4.0), &copy; OpenStreetMap contributors.</p>
  </div>
</div>"""


def status_idle():
    return ('<div class="lc-status"><span class="lc-dot"></span><span>No request yet. Choose a place and a start month, '
            'then press <b>Create 12 monthly maps</b>.</span></div>')


def status_working(mode, compare_dw, state=None, seconds=0.0):
    # a row of twelve squares that fill in as the months are predicted
    duration = "about 1.5 minutes" if str(mode).lower().startswith("ensemble") else "about half a minute"
    extra = ", plus 1 to 2 minutes for Dynamic World" if compare_dw else ""
    state = state or {"fraction": 0.0, "text": "Starting"}
    done = min(12, max(0, round(state["fraction"] * 13) - 1))
    squares = "".join(f'<i class="{"on" if k < done else ""}"></i>' for k in range(12))
    return (f'<div class="lc-status working"><span class="lc-dot"></span><span>{esc(state["text"])} &hellip; '
            f'<small>usually {duration}{extra}</small></span><span class="lc-progress">{squares}</span>'
            f'<code>{seconds:.0f} s</code></div>')


def status_done(record, seconds):
    return (f'<div class="lc-status done"><span class="lc-dot"></span><span>Done in {seconds:.0f} s. '
            f'Results are saved in <code>{esc(record["result_folder"])}</code></span></div>')


def status_error(message):
    return (f'<div class="lc-status error"><span class="lc-dot"></span><span><b>No maps were made.</b> '
            f'{esc(message)}</span></div>')


def empty_plate():
    tiles = "".join(f'<figure class="lc-tile empty"><figcaption><b>{m}</b><span>&mdash;</span></figcaption>'
                    f'<div class="lc-img"></div></figure>' for m in MONTHS)
    return (f'<section class="lc-plate"><div class="lc-plate-head"><b>Plate 1</b><h2>Twelve monthly maps</h2>'
            f'<span>appear here after a request</span></div><div class="lc-grid empty">{tiles}</div></section>')


# ---------------------------------------------------------------- results

def readout(record, info, dw_table=None):
    lat, lon = record["lat"], record["lon"]
    direction, orbit = record["track"].split()
    name, detail = model_name(record, info)
    months = record["months"]
    scenes = record["scenes"]
    most = max(max(scenes), 1)
    bars = "".join(f'<span class="{"none" if s == 0 else ""}" style="height:{max(6, 100 * s / most):.0f}%" '
                   f'title="{month_label(m)}: {s} scenes"></span>' for m, s in zip(months, scenes))

    cells = [
        ("Place", f"{degrees(lat, 'N', 'S')}<br>{degrees(lon, 'E', 'W')}",
         f"{lat:.4f}, {lon:.4f} &middot; {record['size_km']:g} &times; {record['size_km']:g} km"),
        ("Period", f"{month_label(months[0])} &ndash;<br>{month_label(months[-1])}",
         f"{record['months_with_map']} of 12 months mapped"),
        ("Radar track", f"{direction.capitalize()}<br>orbit {esc(orbit)}",
         "same track, viewing angle and pixel grid in every month"),
        ("Scenes per month", f'<div class="lc-spark">{bars}</div>',
         f"{min(scenes)} to {max(scenes)} images per monthly median"),
        ("Counted area", f"{record['counted_area_km2']:.1f} km²",
         f"{100 * record['counted_fraction']:.1f} % of the image has radar data in every mapped month"),
        ("Model", esc(name), f"{detail} &middot; benchmark mIoU {info['test_miou']:.3f} on six test areas "
                             f"(not the accuracy at this place) &middot; "
                             f"{record['months_with_map']} maps in {record['prediction_seconds']:.0f} s"),
    ]
    html_cells = "".join(f'<div class="lc-cell"><span class="lc-k">{k}</span><span class="lc-v">{v}</span>'
                         f'<span class="lc-s">{s}</span></div>' for k, v, s in cells)

    notes = []
    if record["missing_months"]:
        notes.append("No radar image on this track in " + ", ".join(month_label(m) for m in record["missing_months"])
                     + ". Missing months have no map and no percentages. Changes involving a missing month are "
                     "not available; the total change needs both the first and the last month.")
    if record["low_coverage_months"]:
        notes.append("Radar covers less than 90 % of the area in " +
                     ", ".join(month_label(m) for m in record["low_coverage_months"]) + ".")
    if record["counted_fraction"] < 0.9:
        notes.append(f"Only {100 * record['counted_fraction']:.1f} % of the area has radar data in every mapped month. "
                     "All shares and changes describe only this part.")
    if dw_table is not None and "status" in dw_table:
        failed = [month_label(m) for m, s in zip(dw_table["month"], dw_table["status"]) if str(s).startswith("download failed")]
        if failed:
            notes.append("Dynamic World could not be downloaded for " + ", ".join(failed) +
                         " (the Sentinel-1 results are not affected).")
    note_html = ('<ul class="lc-notes">' + "".join(f"<li>{n}</li>" for n in notes) + "</ul>") if notes else ""
    return f'<section class="lc-readout">{html_cells}</section>{note_html}'


def map_plate(record, info, share_table):
    classes, colors = info["classes"], info["colors"]
    folder = Path(record["result_folder"])
    grid = record["grid"]
    width_km = grid["width"] * grid["transform"][0] / 1000

    tiles = []
    for label, scenes in zip(record["months"], record["scenes"]):
        path = folder / f"lulc_{label}.tif"
        name, year = month_label(label).split()
        if label in record["missing_months"] or not path.exists():
            picture = '<p>no radar image<br>on this track</p>'
            right = "0 scenes"
        else:
            with rasterio.open(path) as src:
                class_map = src.read(1)
            picture = f'<img src="{map_png(class_map, colors)}" alt="Land cover {esc(name)} {year}">'
            right = f"{scenes} scenes"
            if label in record["low_coverage_months"]:
                right = f"<em>low coverage</em> &middot; {right}"
        tiles.append(f'<figure class="lc-tile"><figcaption><span><b>{name}</b> {year}</span><span>{right}</span>'
                     f'</figcaption><div class="lc-img">{picture}</div></figure>')

    # coordinates of the north-west and south-east corner, like on a printed map sheet
    t = grid["transform"]
    xs = [t[2], t[2] + grid["width"] * t[0]]
    ys = [t[5], t[5] + grid["height"] * t[4]]
    lons, lats = warp_transform(grid["crs"], "EPSG:4326", xs, ys)
    epsg = grid["crs"].split(":")[-1]
    zone = f"UTM zone {epsg[-2:]}{'N' if epsg.startswith('326') else 'S'}" if epsg[:3] in ("326", "327") else grid["crs"]

    mean_share = share_table[classes].mean()
    legend = "".join(f'<span><i style="background:{c}"></i>{esc(n)} <small>{mean_share[n]:.1f} %</small></span>'
                     for n, c in zip(classes, colors))
    scale_km = next(s for s in [5, 2, 1, 0.5, 0.2] if s <= width_km / 2)
    scale = (f'<div class="lc-scale"><div style="width:{100 * scale_km / width_km:.1f}%"><div class="bar"><span></span>'
             f'<span></span></div><div class="ticks"><span>0</span><span>{scale_km:g} km</span></div></div>'
             f'one map = {width_km:.1f} km across &middot; north up</div>')

    return f"""
<section class="lc-plate">
  <div class="lc-plate-head"><b>Plate 1</b><h2>Twelve monthly maps</h2>
       <span>predicted land cover &middot; {esc(record['track'].lower())} track</span></div>
  <div class="lc-collar"><span>NW {degrees(lats[0], 'N', 'S')} &nbsp;{degrees(lons[0], 'E', 'W')}</span>
       <span>map grid {esc(zone)} &middot; pixel 10 m &middot; hatched = no radar data &middot; previews are reduced
       for the page, the GeoTIFF downloads keep the full 10 m grid</span></div>
  <div class="lc-grid">{''.join(tiles)}</div>
  <div class="lc-collar" style="justify-content:flex-end;margin-top:6px">
       <span>SE {degrees(lats[1], 'N', 'S')} &nbsp;{degrees(lons[1], 'E', 'W')}</span></div>
  <div class="lc-plate-foot"><div><p class="lc-sub">Legend &middot; mean share over the mapped months</p>
       <div class="lc-legend">{legend}</div></div>{scale}</div>
</section>"""


def total_bars(share_table, total_table, classes, colors):
    total = total_table[classes].iloc[0]
    first, last = share_table[classes].iloc[0], share_table[classes].iloc[-1]
    if total.isna().all():
        return ('<div class="lc-empty">No total change: the first or the last month has no map. '
                'The month-to-month changes are shown below.</div>')
    largest = max(float(np.nanmax(np.abs(total.to_numpy(dtype=float)))), 0.5) * 1.15
    rows = []
    for name, color in zip(classes, colors):
        value = total[name]
        width = 50 * abs(value) / largest
        left = 50 if value >= 0 else 50 - width
        rows.append(f'<span class="name"><i style="background:{color}"></i>{esc(name)}</span>'
                    f'<span class="track"><span style="left:{left:.2f}%;width:{width:.2f}%;background:{color}"></span></span>'
                    f'<span class="val">{value:+.2f} pp</span>'
                    f'<span></span><span class="from">{first[name]:.1f} % &rarr; {last[name]:.1f} %</span>')
    return f'<div class="lc-bars">{"".join(rows)}</div>'


def line_chart(share_table, classes, colors):
    labels = list(share_table["month"])
    values = share_table[classes].to_numpy(dtype=float)
    width, height = 660, 300
    left, right, top, bottom = 40, 176, 12, 36
    top_value = max(10, math.ceil(np.nanmax(values) / 10) * 10)

    def x(i):
        return left + i * (width - left - right) / (len(labels) - 1)

    def y(v):
        return top + (1 - v / top_value) * (height - top - bottom)

    parts = []
    for v in range(0, top_value + 1, 10):
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="{RULE}"/>'
                     f'<text x="{left - 8}" y="{y(v) + 4:.1f}" text-anchor="end">{v}</text>')
    for i, label in enumerate(labels):
        name, year = month_label(label).split()
        parts.append(f'<text x="{x(i):.1f}" y="{height - bottom + 16}" text-anchor="middle">{name[0]}</text>')
        if i == 0 or label.endswith("-01"):
            parts.append(f'<text x="{x(i):.1f}" y="{height - bottom + 30}" text-anchor="middle" fill="{MUTED}">{year}</text>')

    ends = []
    for k, (name, color) in enumerate(zip(classes, colors)):
        stroke = line_color(color)
        segment, segments = [], []
        for i, v in enumerate(values[:, k]):
            if np.isnan(v):
                if segment:
                    segments.append(segment)
                segment = []
            else:
                segment.append((x(i), y(v), v, labels[i]))
        if segment:
            segments.append(segment)
        for seg in segments:
            points = " ".join(f"{px:.1f},{py:.1f}" for px, py, _, _ in seg)
            parts.append(f'<polyline points="{points}" fill="none" stroke="{stroke}" stroke-width="1.8"/>')
            for px, py, v, label in seg:
                parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.6" fill="{SHEET}" stroke="{stroke}" stroke-width="1.4">'
                             f'<title>{esc(name)}, {month_label(label)}: {v:.2f} %</title></circle>')
        valid = np.where(~np.isnan(values[:, k]))[0]
        if len(valid):
            last = valid[-1]
            ends.append([y(values[last, k]), name, stroke, values[last, k], x(last)])

    # labels at the end of the lines, pushed apart so they do not overlap
    ends.sort(key=lambda e: e[0])
    gap = 14
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
    overflow = ends[-1][0] - (height - bottom) if ends else 0
    if overflow > 0:
        for e in ends:
            e[0] -= overflow
        for i in range(len(ends) - 2, -1, -1):
            ends[i][0] = min(ends[i][0], ends[i + 1][0] - gap)
    label_x = width - right + 12
    for label_y, name, stroke, value, line_end in ends:
        short = name.split(" (")[0].split(" /")[0]
        parts.append(f'<text x="{label_x}" y="{label_y + 4:.1f}" fill="{INK}"><tspan fill="{stroke}">&#9632;</tspan> '
                     f'{esc(short)} <tspan fill="{MUTED}">{value:.1f}</tspan></text>')

    return (f'<div class="lc-chart"><svg viewBox="0 0 {width} {height}" role="img" aria-label="Class share per month" '
            f'font-family="IBM Plex Mono, monospace" font-size="10.5" fill="{MUTED}">{"".join(parts)}</svg></div>')


def heat_table(change_table, classes):
    values = change_table[classes].to_numpy(dtype=float)
    cap = max(2.0, float(np.nanmax(np.abs(values))) if not np.isnan(values).all() else 2.0)
    heads = "".join(f"<th>{month_label(c.split(' → ')[1]).split()[0]}</th>" for c in change_table["change"])
    rows = []
    for k, name in enumerate(classes):
        cells = []
        for i in range(values.shape[0]):
            v = values[i, k]
            if np.isnan(v):
                cells.append('<td title="no map in one of the two months">&middot;</td>')
                continue
            strength = min(abs(v) / cap, 1)
            color = RED if v > 0 else BLUE
            alpha = 0 if abs(v) < 0.05 else 0.10 + 0.75 * strength
            text = SHEET if alpha > 0.55 else INK
            rgb = ",".join(str(int(color[j:j + 2], 16)) for j in (1, 3, 5))
            shown = "0.0" if abs(v) < 0.05 else f"{v:+.1f}"
            cells.append(f'<td style="background:rgba({rgb},{alpha:.2f});color:{text}" '
                         f'title="{esc(change_table["change"].iloc[i])}: {v:+.2f} pp">{shown}</td>')
        rows.append(f"<tr><td>{esc(name.split(' (')[0])}</td>{''.join(cells)}</tr>")
    return (f'<div class="lc-scroll"><table class="lc-heat"><tr><th>change into &rarr;</th>{heads}</tr>'
            f'{"".join(rows)}</table></div>')


def change_plate(share_table, change_table, total_table, info):
    classes, colors = info["classes"], info["colors"]
    first, last = share_table["month"].iloc[0], share_table["month"].iloc[-1]
    return f"""
<section class="lc-plate">
  <div class="lc-plate-head"><b>Plate 2</b><h2>How the predicted classes change</h2>
       <span>percentage points (pp) of the counted area</span></div>
  <div class="lc-two">
    <div><p class="lc-sub">Total change &middot; {month_label(first)} &rarr; {month_label(last)}</p>
         {total_bars(share_table, total_table, classes, colors)}</div>
    <div><p class="lc-sub">Share of each class per month (%)</p>{line_chart(share_table, classes, colors)}</div>
  </div>
  <p class="lc-sub" style="margin-top:26px">Month-to-month change (pp) &middot; red = more, blue = less than the month before</p>
  {heat_table(change_table, classes)}
  <p class="lc-aside">Predicted differences may reflect real land cover change, seasonal radar conditions (wet soil, leaves, frost)
  or classification errors. Persistence over several months alone does not confirm a real change; independent dated
  evidence is needed.</p>
</section>"""


DW_STATUS = {
    "no Sentinel-1 map": ("no radar map", "no Sentinel-1 image this month"),
    "no Dynamic World image": ("no image", "no Sentinel-2 image this month"),
    "no pixel valid in both maps": ("cloudy", "no cloud-free pixel in the counted area"),
}


def dw_plate(dw_table):
    head = ('<div class="lc-plate-head"><b>Plate 3</b><h2>Check against Dynamic World</h2>'
            '<span>independent map from Sentinel-2, never used for training</span></div>')
    if dw_table is None:
        return (f'<section class="lc-plate">{head}<div class="lc-empty">Not requested for this run. Tick '
                '<b>Also compare with Dynamic World</b> and create the maps again to see, month by month, how often '
                'the radar model gives the same class as Dynamic World (Google &amp; World Resources Institute).</div>'
                '</section>')
    cells, agreements = [], []
    for _, row in dw_table.iterrows():
        name, year = month_label(row["month"]).split()
        status = str(row["status"])
        if status == "ok":
            agreements.append(row["agreement %"])
            cells.append(f'<div style="background:rgba(27,36,32,{0.04 + 0.16 * (row["agreement %"] - 50) / 50:.2f})">'
                         f'<b>{name}</b><small>{year}</small><strong>{row["agreement %"]:.0f}%</strong>'
                         f'<small>same class &middot; compared on {row["compared area %"]:.0f} % of the area</small></div>')
        else:
            short, long = DW_STATUS.get(status, ("failed", "download failed"))
            cells.append(f'<div class="off" title="{esc(status)}"><b>{name}</b><small>{year}</small>'
                         f'<strong>{short}</strong><small>{long}</small></div>')
    if agreements:
        summary = (f"Mean agreement over the {len(agreements)} months with cloud-free Dynamic World pixels: "
                   f"<b>{np.mean(agreements):.1f} %</b>.")
    else:
        summary = "No month had cloud-free Dynamic World pixels in the counted area."
    return f"""
<section class="lc-plate">{head}
  <div class="lc-strip">{''.join(cells)}</div>
  <p class="lc-aside">{summary} Agreement is the share of pixels where both maps give the same class, counted only on
  pixels that are cloud-free in Dynamic World. Dynamic World is optical, so it has gaps where it was cloudy; the radar
  model does not. Both are predictions from different satellites, so agreement is not accuracy.</p>
</section>"""


def make_zip(record, files):
    # one download with all files of the request
    folder = Path(record["result_folder"])
    for old in folder.glob("*.zip"):
        old.unlink()
    path = folder / f"lulc_{folder.name}.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in files:
            archive.write(file, arcname=Path(file).name)
    return path, len(files)
