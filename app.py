# app.py
# Streamlit UI for PS-SMaRT (unstable_slope_hazard_pipeline.py)

from __future__ import annotations

import os
import re
import time
import json
import base64
from pathlib import Path
from typing import Optional, Dict
from concurrent.futures import ThreadPoolExecutor

import streamlit as st
import numpy as np
import pandas as pd

# Optional heavy libs (guarded)
try:
    import rasterio
except Exception:
    rasterio = None

try:
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import matplotlib.patches as mpatches
except Exception:
    gpd = None
    plt = None

# ---- import your pipeline ----
from unstable_slope_hazard_pipeline import run_pipeline, BRAND as PIPE_BRAND

# -----------------------------------------------------------------------------
# Branding
# -----------------------------------------------------------------------------
BRAND   = "PS-SMaRT"  # keep in sync with pipeline
TAGLINE = "Persistent Scatterer–Soil Moisture Analysis for Risk & Triggering"

def _brand_svg_b64() -> str:
    svg = """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
      <rect width="64" height="64" rx="12" fill="#0B6E99"/>
      <rect x="10" y="14" width="16" height="12" rx="2" fill="#ffffff"/>
      <rect x="28" y="14" width="10" height="12" rx="2" fill="#D6ECF4"/>
      <rect x="40" y="10" width="14" height="8" rx="2" fill="#ffffff" opacity="0.85"/>
      <rect x="40" y="24" width="14" height="8" rx="2" fill="#ffffff" opacity="0.85"/>
      <rect x="24" y="26" width="6" height="12" rx="2" fill="#ffffff"/>
      <path d="M36 36c6 0 12 6 12 12" stroke="#ffffff" stroke-width="2" fill="none" opacity="0.9"/>
      <path d="M36 40c4 0 8 4 8 8" stroke="#D6ECF4" stroke-width="2" fill="none"/>
      <path d="M46 36c4 6 6 9 6 12a6 6 0 1 1-12 0c0-3 2-6 6-12z" fill="#86C5DA" stroke="#ffffff" stroke-width="1"/>
    </svg>
    """
    return base64.b64encode(svg.encode("utf-8")).decode("ascii")

ICON_B64 = _brand_svg_b64()

st.set_page_config(
    page_title=f"{BRAND} — Landslide Hazard Toolkit",
    page_icon="🛰️💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Hide Streamlit default chrome
st.markdown(
    """
    <style>
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
      header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Single header: larger logo + bigger title + caption
st.markdown(
    f"""
    <style>
      .pssmart-header {{
        display:flex; align-items:center; gap:14px; margin-bottom:0.75rem;
      }}
      .pssmart-title {{
        font-size: clamp(2.0rem, 3.2vw + 1rem, 3.2rem);  /* responsive, larger */
        font-weight: 800;
        line-height: 1.05;
        letter-spacing: 0.2px;
      }}
      .pssmart-tagline {{
        margin-top: -0.35rem;
        color: #445;
        opacity: 0.95;
        font-size: clamp(1.0rem, 0.6vw + 0.8rem, 1.15rem);
        font-weight: 500;
      }}
    </style>
    <div class="pssmart-header">
      <img src="data:image/svg+xml;base64,{ICON_B64}" width="48" height="48" />
      <div class="pssmart-title">{BRAND} — Landslide Hazard Toolkit</div>
    </div>
    <div class="pssmart-tagline">{TAGLINE}</div>
    """,
    unsafe_allow_html=True,
)

# st.markdown(
#     f"""
#     <div style="display:flex;align-items:center;gap:10px;margin-bottom:0.5rem">
#       <img src="data:image/svg+xml;base64,{ICON_B64}" width="32" height="32" />
#       <div style="font-size:1.6rem;font-weight:700">{BRAND} — Landslide Hazard Toolkit</div>
#     </div>
#     <div style="margin-top:-0.5rem;color:#445;opacity:0.9">{TAGLINE}</div>
#     """,
#     unsafe_allow_html=True,
# )

# -----------------------------------------------------------------------------
# Help / Theory (equations)
# -----------------------------------------------------------------------------
def render_help():
    st.header("Theory & Methods — PS-SMaRT")
    st.caption("Persistent Scatterer–Soil Moisture Analysis for Risk & Triggering")

    # -----------------------------
    # 0) Notation & frames
    # -----------------------------
    st.subheader("0) Notation & Coordinate Frames")
    st.markdown(
        "- ENU = local East–North–Up right-handed frame.\n"
        "- Heading \(h\) in degrees (clockwise from North). Incidence \(\theta\) measured **from vertical**.\n"
        "- Slope \(S\) in degrees **from horizontal**; Aspect \(A\) in degrees **clockwise from North**."
    )
    st.latex(r"\phi = \mathrm{deg}^{-1}\!\left((h+90)\bmod 360\right)")
    st.latex(r"\theta = \mathrm{deg}^{-1}\!\left(\text{incidence from vertical}\right)")
    st.latex(r"A_r = \mathrm{deg}^{-1}(A), \qquad S_r = \mathrm{deg}^{-1}(S)")

    st.markdown("---")

    # -----------------------------
    # Step 0) LOS → downslope projection
    # -----------------------------
    st.subheader("Step 0 — LOS → Downslope Projection (optional)")

    st.markdown("**(a) LOS unit vector in ENU (toward sensor)**")
    st.latex(r"""
    \mathbf{l} =
    \begin{bmatrix}
      -\sin\phi\,\sin\theta\\
      -\cos\phi\,\sin\theta\\
      \ \cos\theta
    \end{bmatrix}
    """)

    st.markdown("**(b) Downslope unit vector from aspect/slope**")
    st.latex(r"""
    \mathbf{d} =
    \begin{bmatrix}
      \sin A_r\,\cos S_r\\
      \cos A_r\,\cos S_r\\
      -\sin S_r
    \end{bmatrix}
    """)

    st.markdown("**(c) Sensitivity and projection**")
    st.latex(r"s = \mathbf{l}\cdot\mathbf{d}")
    st.latex(r"v_{\parallel} = \dfrac{v_{\mathrm{LOS}}}{\max(|s|,\ \varepsilon)}")
    st.latex(r"y_{\parallel}(t) = \dfrac{y_{\mathrm{LOS}}(t)}{\max(|s|,\ \varepsilon)}")

    st.markdown("**(d) Circular (aspect) bilinear averaging**")
    st.latex(r"\bar{A}=\arg\!\left(\sum_{i=1}^{4} w_i\,e^{\,j A_i}\right),\quad w_i\!\ge 0,\ \sum_i w_i=1")

    st.markdown("**(e) Acceptance criteria**")
    st.latex(r"S \ge S_{\min}, \qquad |s| \ge s_{\min}")

    st.markdown("---")

    # -----------------------------
    # Step A) Filtering
    # -----------------------------
    st.subheader("Step A — Filtering of Projected Points")
    st.latex(r"S(x,y) \ge S_{\min} \quad\text{and}\quad |v_{\parallel}(x,y)| \ge v_{\min}")

    st.markdown("---")

    # -----------------------------
    # Step B) DBSCAN clustering
    # -----------------------------
    st.subheader("Step B — Spatial Clustering (DBSCAN)")
    st.markdown(
        "Let \( \mathcal{P}=\{(x_i,y_i)\} \) in a metric CRS (meters). For distance \(d\) and parameters "
        r"\(\varepsilon,\ \text{min\_samples}\):"
    )
    st.latex(r"N_\varepsilon(p) = \{\,q\in\mathcal{P}\ :\ d(p,q)\le \varepsilon\,\}")
    st.latex(r"\text{core}(p)\iff |N_\varepsilon(p)|\ \ge\ \text{min\_samples}")
    st.markdown("Clusters are maximal density-connected sets; label \( -1 \) denotes noise.")

    st.markdown("---")

    # -----------------------------
    # Step C) Cluster polygons & stats
    # -----------------------------
    st.subheader("Step C — Cluster Polygons & Statistics")
    st.markdown("**Convex hull** of cluster \(C\):")
    st.latex(r"H_C=\mathrm{hull}\!\left(\{(x_i,y_i)\in C\}\right)")
    st.markdown("**Polished hull** (buffer–union, radius \(r\)):")
    st.latex(r"H_C'=\left(\bigcup_{i\in C} B_{r}(p_i)\right)^{\circ}")
    st.markdown("**Descriptive statistics over \(v_{\parallel}\)**: mean, std, min, max; polygon area.")

    st.markdown("---")

    # -----------------------------
    # Step D) Wet-anomaly overlap
    # -----------------------------
    st.subheader("Step D — Wet-Anomaly Overlap (optional)")
    st.markdown("Contingency table on sampled valid pixels:")
    st.latex(r"""
    \begin{array}{c|cc}
      & \text{Inside slopes} & \text{Outside slopes}\\\hline
      \text{Anomaly} & A & B\\
      \text{No anomaly} & C & D
    \end{array}
    """)
    st.markdown("**Chi-square statistic**")
    st.latex(r"\chi^2 = \sum_{i,j}\frac{(O_{ij}-E_{ij})^2}{E_{ij}},\quad E_{ij}=\frac{(\text{row}_i)(\text{col}_j)}{A+B+C+D}")
    st.markdown("**Matthews correlation / \( \phi \)**")
    st.latex(r"\phi=\mathrm{MCC}=\dfrac{AD-BC}{\sqrt{(A+B)(A+C)(B+D)(C+D)}}")

    st.markdown("---")

    # -----------------------------
    # Step E) TWI inside vs outside
    # -----------------------------
    st.subheader("Step E — TWI Inside vs Outside (optional)")
    st.markdown("Welch’s t-test (unequal variances):")
    st.latex(r"t = \dfrac{\bar{T}_{\mathrm{in}}-\bar{T}_{\mathrm{out}}}{\sqrt{\dfrac{s^2_{\mathrm{in}}}{n_{\mathrm{in}}}+\dfrac{s^2_{\mathrm{out}}}{n_{\mathrm{out}}}}}")
    st.latex(r"""
    \nu \approx
    \frac{\left(\frac{s^2_{\mathrm{in}}}{n_{\mathrm{in}}}+\frac{s^2_{\mathrm{out}}}{n_{\mathrm{out}}}\right)^2}
         {\frac{s^4_{\mathrm{in}}}{n_{\mathrm{in}}^{2}(n_{\mathrm{in}}-1)}+\frac{s^4_{\mathrm{out}}}{n_{\mathrm{out}}^{2}(n_{\mathrm{out}}-1)}}
    """)

    st.markdown("---")

    # -----------------------------
    # Step F) Hazard index & classes
    # -----------------------------
    st.subheader("Step F — Hazard Index & Classes")
    st.markdown("**Robust normalization (per layer \(X\))**")
    st.latex(r"X'=\mathrm{clip}\!\left(\dfrac{X-\mathrm{P5}(X)}{\mathrm{P95}(X)-\mathrm{P5}(X)+\varepsilon},\ 0,\ 1\right)")
    st.markdown("**Blend (mean over available layers)**")
    st.latex(r"HI=\dfrac{1}{K}\sum_{k=1}^{K} X'_k")
    st.markdown("**Class thresholds**")
    st.latex(r"\text{Low: } HI\le 0.20,\qquad \text{Moderate: } 0.20<HI\le 0.40,\qquad \text{High: } HI>0.40")

    st.markdown("**Zonal (polygon) classification**")
    st.latex(r"\overline{HI}_{\Omega} = \dfrac{1}{|\Omega|}\iint_{\Omega} HI(x,y)\,dx\,dy")

    st.markdown("---")

    # -----------------------------
    # Workflow
    # -----------------------------
    st.subheader("Workflow (End-to-End)")
    st.code(
        "  (Raw PS LOS CSV) + Slope + Aspect [+ Heading/Incidence rasters]\n"
        "                 │\n"
        "                 ▼\n"
        " [0] LOS→downslope (s = l·d), filter by |s| & slope\n"
        "                 │\n"
        "                 ├──▶ (Projected points CSV)  (if provided, skip [0])\n"
        "                 ▼\n"
        " [A] Filter by slope ≥ S_min and |v∥| ≥ v_min\n"
        "                 ▼\n"
        " [B] DBSCAN clustering in metric CRS\n"
        "                 ▼\n"
        " [C] Cluster polygons + stats\n"
        "            ┌──────────────┴──────────────┐\n"
        "            ▼                             ▼\n"
        "   [D] Anomaly overlap (χ², MCC)   [E] TWI in/out t-test\n"
        "            └──────────────┬──────────────┘\n"
        "                           ▼\n"
        " [F] Hazard index (robust blend) + Hazard class raster\n"
        "                           ▼\n"
        "   Zonal hazard per polygon + zonal stats + provenance/logs",
        language="text",
    )

    # -----------------------------
    # Practical guidance (brief)
    # -----------------------------
    st.subheader("Practical Guidance")
    st.markdown(
        "- **Units/CRS:** clustering uses the slope raster CRS; it should be projected (meters). "
        "Incidence is measured **from vertical**; velocities in mm/yr.\n"
        "- **Starting values:** \(S_{\min}\!=10^\circ\); \(|v_{\min}|\!=5\)–\(10\) mm/yr; "
        "DBSCAN \(\varepsilon\) ≈ 1–2× mean PS spacing; `min_samples` 5–10.\n"
        "- **Numerics:** use \(\varepsilon\approx 10^{-3}\) to stabilize division by \(s\); "
        "use circular averaging for aspect to avoid wraparound at \(360^\circ\).\n"
        "- **Caveat:** downslope projection assumes dominantly downslope motion; complex kinematics may deviate."
    )


def _guess_col_like(gdf, must_have_words):
    """
    Return the first column whose normalized name contains all words in must_have_words.
    Normalization: lowercase, remove underscores.
    """
    words = [w.lower() for w in must_have_words]
    for c in gdf.columns:
        norm = c.lower().replace("_", "")
        if all(w in norm for w in words):
            return c
    return None

def _guess_hazard_class_col(gdf):
    # direct match or shapefile truncations
    c = _guess_col_like(gdf, ["hazard", "class"])
    if c: return c
    # a few common legacy/truncated variants just in case
    for cand in ["hazard_cla", "hazard_cl", "hz_class", "haz_class", "hzclass", "hazclass", "h_class"]:
        if cand in gdf.columns:
            return cand
    return None

def _guess_mean_hazard_col(gdf):
    # mean hazard column used when we need to reconstruct classes
    c = _guess_col_like(gdf, ["mean", "haz"])
    if c: return c
    for cand in ["mean_hazar", "mean_haz", "meanhaz", "mn_hazard"]:
        if cand in gdf.columns:
            return cand
    return None

# -----------------------------------------------------------------------------
# Preview helpers
# -----------------------------------------------------------------------------
CLASS_COLORS = {"Low": "#2ecc71", "Moderate": "#f1c40f", "High": "#e74c3c"}
CLASS_COLORS_ORDERED = [CLASS_COLORS["Low"], CLASS_COLORS["Moderate"], CLASS_COLORS["High"]]

def preview_hazard_class_raster_categorical(path: Path, title: str):
    if not rasterio or not plt:
        st.info("Raster preview requires rasterio & matplotlib.")
        return
    try:
        with rasterio.open(path) as ds:
            arr = ds.read(1).astype(float)
            nodata = ds.nodata
            if nodata is not None:
                arr[arr == nodata] = np.nan
        cmap = ListedColormap(CLASS_COLORS_ORDERED)
        norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5], cmap.N)
        fig = plt.figure(figsize=(6, 4)); ax = fig.add_subplot(111)
        im = ax.imshow(arr, cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(title); ax.axis("off")
        cbar = fig.colorbar(im, ax=ax, ticks=[1,2,3], shrink=0.8)
        cbar.ax.set_yticklabels(["Low","Moderate","High"])
        st.pyplot(fig); plt.close(fig)
    except Exception as e:
        st.warning(f"Preview failed for {path.name}: {e}")

def preview_vectors(path: Path, title: str, color: str = "#8e44ad"):
    if not gpd or not plt:
        st.info("Vector preview requires geopandas & matplotlib.")
        return
    try:
        gdf = gpd.read_file(path)
        if gdf.empty:
            st.info(f"{title}: no features to preview.")
            return
        fig = plt.figure(figsize=(6,4)); ax = fig.add_subplot(111)
        gdf.plot(ax=ax, facecolor="none", edgecolor=color, linewidth=1.0)
        ax.set_title(title); ax.axis("off")
        st.pyplot(fig); plt.close(fig)
    except Exception as e:
        st.warning(f"Preview failed for {path.name}: {e}")

def preview_classified_polygons_color(path: Path, title: str):
    """Color polygons by hazard class. Handles shapefile field-name truncation."""
    if not gpd or not plt:
        st.info("Vector preview requires geopandas and matplotlib. Skipping preview.")
        return
    try:
        gdf = gpd.read_file(path)
        if gdf.empty:
            st.info(f"{title}: no features to preview.")
            return

        # Try to find an existing hazard-class column (handles truncation)
        hc_col = _guess_hazard_class_col(gdf)

        # If missing, try to reconstruct from mean-hazard numeric field
        if hc_col is None:
            mh_col = _guess_mean_hazard_col(gdf)
            if mh_col is not None and pd.api.types.is_numeric_dtype(gdf[mh_col]):
                st.info("Field 'hazard_class' not found; deriving classes from mean hazard.")
                gdf["_hc"] = pd.cut(
                    gdf[mh_col].astype(float),
                    bins=[-0.01, 0.20, 0.40, 1.00],
                    labels=["Low", "Moderate", "High"]
                ).astype(str)
                hc_col = "_hc"

        if hc_col is None:
            st.warning("No 'hazard_class' (or derivable mean hazard) found; plotting unclassified polygons.")
            preview_vectors(path, title)  # fallback outline
            return

        # Normalize labels and draw
        gdf[hc_col] = gdf[hc_col].astype(str).str.title()
        fig, ax = plt.subplots(figsize=(6, 4))
        CLASS_COLORS = {"Low": "#2ecc71", "Moderate": "#f1c40f", "High": "#e74c3c"}
        for cls, sub in gdf.groupby(hc_col):
            color = CLASS_COLORS.get(cls, "#95a5a6")
            sub.plot(ax=ax, facecolor=color, edgecolor="black", linewidth=0.3, label=cls)

        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="lower right", frameon=True)
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.warning(f"Preview failed for {path.name}: {e}")


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------
def persist_upload(uploaded_file, work_dir: Path, name: Optional[str] = None) -> Optional[Path]:
    if uploaded_file is None:
        return None
    fname = name if name else uploaded_file.name
    out_path = work_dir / fname
    with open(out_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return out_path

def safe_read_text(path: Path, max_bytes: int = 1_000_000) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(max_bytes)
    except Exception as e:
        return f"[log not available: {e}]"

def download_button_for(path: Path, label: Optional[str] = None):
    if not path or not path.exists():
        return
    label = label or f"Download {path.name}"
    with open(path, "rb") as f:
        st.download_button(label, data=f, file_name=path.name, mime="application/octet-stream")

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
st.sidebar.title("Inputs & Parameters")
page = st.sidebar.radio("Page", ["Run", "Help / Theory"], index=0)

mode = st.sidebar.radio(
    "Input Mode",
    ("Raw time-series → project (LOS→downslope)", "Already-projected points CSV"),
    index=0,
)

work_root = st.sidebar.text_input(
    "Output Folder (absolute path)",
    value=str(Path.home() / "hazard_pipeline_outputs"),
)

with st.sidebar.expander("Required rasters & CSV"):
    slope_tif_u = st.file_uploader("Slope (.tif) • required", type=["tif", "tiff"])
    if mode.startswith("Raw"):
        aspect_tif_u = st.file_uploader("Aspect (.tif) • required", type=["tif", "tiff"])
        raw_csv_u = st.file_uploader("Raw time-series CSV • required (LAT,LON,VEL,D*)", type=["csv"])
        points_csv_u = None
    else:
        aspect_tif_u = None
        raw_csv_u = None
        points_csv_u = st.file_uploader("Projected points CSV • required (VEL_REPR)", type=["csv"])

with st.sidebar.expander("Optional layers"):
    anomaly_tif_u = st.file_uploader("Wet anomaly raster (.tif) • optional", type=["tif", "tiff"])
    twi_tif_u     = st.file_uploader("TWI raster (.tif) • optional", type=["tif", "tiff"])
    pre_unstable_u= st.file_uploader("Pre-existing unstable polygons (.shp/.gpkg/.geojson) • optional",
                                     type=["shp","gpkg","json","geojson"])

with st.sidebar.expander("Projection parameters (if projecting)"):
    default_heading_deg = st.number_input("Default heading (deg from North, clockwise)",
                                          value=-10.82, step=0.01, format="%.2f")
    default_incidence_deg = st.number_input("Default incidence (deg FROM vertical)",
                                            value=31.53, step=0.01, format="%.2f")
    aspect_convention = st.selectbox("Aspect convention", ["north_cw", "east_ccw"], index=0)
    downhill_sign = st.selectbox(
        "Downslope sign convention",
        ["Positive (+)", "Negative (−)"],
        index=0,
        help="The 'downslope = negative'. Choose Negative (−) to match it.",
    )
    require_negative_v = st.checkbox(
        "Keep only negative projected velocities (V_projected < 0)",
        value=False,
        help="Enable to replicate the selection rule.",
    )

with st.sidebar.expander("Filtering & clustering"):
    slope_threshold_deg   = st.number_input("Min slope (deg)", value=10.0, step=0.5, format="%.1f")
    velocity_threshold_abs= st.number_input("|VEL_REPR| threshold (mm/yr)", value=10.0, step=1.0, format="%.1f")
    dbscan_eps_m          = st.number_input("DBSCAN eps (m)", value=100.0, step=5.0, format="%.1f")
    dbscan_min_pts        = st.number_input("DBSCAN min_samples", value=5, step=1, min_value=1)
    polygon_polish_buffer = st.number_input("Polygon polish buffer (m)", value=0.0, step=1.0, format="%.1f")

clean_outputs = st.sidebar.checkbox("Clean output folder before run", value=False)
run_clicked   = st.sidebar.button("Run Pipeline", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# Main content
# -----------------------------------------------------------------------------
if page == "Help / Theory":
    render_help()
    st.stop()

# st.title(f"{BRAND} — Landslide Hazard Toolkit")
# st.caption(TAGLINE)

with st.expander("About PS-SMaRT", expanded=True):
    st.markdown(
        "- **Purpose**: fuse PS-InSAR deformation (projected along local downslope) "
        "with soil-moisture/wet-anomaly and terrain indices to screen unstable slopes.\n"
        "- **Core outputs**: hazard index/class rasters; unstable polygons with zonal hazard class; "
        "statistics and provenance."
    )

# Working directory (single fixed outputs folder)
work_root_path = Path(work_root).expanduser().resolve()
work_root_path.mkdir(parents=True, exist_ok=True)
out_dir = work_root_path / "outputs"
out_dir.mkdir(parents=True, exist_ok=True)
st.info(f"Output directory: `{out_dir}`")

# Persist uploads
persisted: Dict[str, Optional[Path]] = {}
persisted["slope_tif"]   = persist_upload(slope_tif_u, out_dir, name="slope.tif") if slope_tif_u else None
persisted["aspect_tif"]  = persist_upload(aspect_tif_u, out_dir, name="aspect.tif") if aspect_tif_u else None
persisted["raw_csv"]     = persist_upload(raw_csv_u, out_dir, name="raw_timeseries.csv") if raw_csv_u else None
persisted["points_csv"]  = persist_upload(points_csv_u, out_dir, name="points_projected.csv") if points_csv_u else None
persisted["anomaly_tif"] = persist_upload(anomaly_tif_u, out_dir, name="anomaly.tif") if anomaly_tif_u else None
persisted["twi_tif"]     = persist_upload(twi_tif_u, out_dir, name="twi.tif") if twi_tif_u else None
persisted["pre_unstable"]= persist_upload(pre_unstable_u, out_dir, name="pre_unstable.gpkg") if pre_unstable_u else None

def ready_to_run() -> bool:
    if not persisted["slope_tif"]:
        st.error("Slope raster is required."); return False
    if mode.startswith("Raw"):
        if not persisted["aspect_tif"] or not persisted["raw_csv"]:
            st.error("For Raw mode, Aspect raster and Raw CSV are required."); return False
    else:
        if not persisted["points_csv"]:
            st.error("For Projected mode, the Projected points CSV is required."); return False
    return True

# -----------------------------------------------------------------------------
# Run with progress bar
# -----------------------------------------------------------------------------
if run_clicked:
    if ready_to_run():
        # Optional clean
        if clean_outputs:
            # remove everything in out_dir
            for child in out_dir.iterdir():
                try:
                    if child.is_dir():
                        import shutil; shutil.rmtree(child)
                    else:
                        child.unlink(missing_ok=True)
                except Exception:
                    pass

        st.subheader("Run status")
        progress = st.progress(0)
        stage_txt = st.empty()
        log_area = st.empty()

        # Map log lines to coarse progress
        stage_markers = [
            (re.compile(r"Unstable-slope hazard pipeline", re.I), 5,  "Initializing…"),
            (re.compile(r"Projected points saved", re.I),          20, "Projection complete"),
            (re.compile(r"Filtered points retained", re.I),         35, "Filtering complete"),
            (re.compile(r"DBSCAN found", re.I),                     55, "Clustering complete"),
            (re.compile(r"Saved: points_csv", re.I),                65, "Polygons & stats saved"),
            (re.compile(r"Anomaly overlap", re.I),                  75, "Anomaly stats done"),
            (re.compile(r"TWI inside vs outside", re.I),            80, "TWI stats done"),
            (re.compile(r"Hazard index/class written", re.I),       95, "Hazard rasters done"),
            (re.compile(r"Pipeline complete", re.I),               100, "Finished"),
        ]

        def update_progress_from_log(text: str, current: int) -> int:
            newp = current
            for rx, p, label in stage_markers:
                if rx.search(text) and p > newp:
                    newp = p
                    stage_txt.info(label)
            return newp

        # Submit pipeline in a worker thread so we can update the UI
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(
                run_pipeline,
                out_dir=str(out_dir),
                file_prefix=BRAND,
                # choose mode
                points_csv=str(persisted["points_csv"]) if (not mode.startswith("Raw") and persisted["points_csv"]) else None,
                raw_timeseries_csv=str(persisted["raw_csv"]) if (mode.startswith("Raw") and persisted["raw_csv"]) else None,
                # core rasters
                slope_tif=str(persisted["slope_tif"]),
                aspect_tif=str(persisted["aspect_tif"]) if persisted["aspect_tif"] else None,
                aspect_convention=aspect_convention,
                # geometry defaults
                default_heading_deg=float(default_heading_deg),
                default_incidence_deg=float(default_incidence_deg),
                # clustering / thresholds
                slope_threshold_deg=float(slope_threshold_deg),
                velocity_threshold_abs=float(velocity_threshold_abs),
                dbscan_eps_m=float(dbscan_eps_m),
                dbscan_min_pts=int(dbscan_min_pts),
                polygon_polish_buffer=float(polygon_polish_buffer),
                # optional layers
                anomaly_tif=str(persisted["anomaly_tif"]) if persisted["anomaly_tif"] else None,
                twi_tif=str(persisted["twi_tif"]) if persisted["twi_tif"] else None,
                pre_unstable_shp=str(persisted["pre_unstable"]) if persisted["pre_unstable"] else None,
                # slide-style options
                downhill_negative=(downhill_sign == "Negative (−)"),
                require_negative_v=bool(require_negative_v),
                clean_out_dir=False,
            )

            log_path = out_dir / f"{BRAND}_pipeline.log"
            pval = 0
            last_tail = ""
            while not fut.done():
                if log_path.exists():
                    txt = safe_read_text(log_path, max_bytes=500_000)
                    pval = update_progress_from_log(txt, pval)
                    progress.progress(min(max(pval, 1), 99))
                    # show last ~40 lines
                    lines = txt.splitlines()[-40:]
                    tail = "\n".join(lines)
                    if tail != last_tail:
                        log_area.code(tail, language="text")
                        last_tail = tail
                time.sleep(0.5)

            # finalize
            exc = fut.exception()
            if exc:
                progress.progress(100)
                stage_txt.error("Pipeline failed.")
                st.exception(exc)
                st.stop()
            else:
                progress.progress(100)
                stage_txt.success("Pipeline finished successfully.")

        # -----------------------------------------------------------------------------
        # Logs & provenance
        # -----------------------------------------------------------------------------
        log_path = out_dir / f"{BRAND}_pipeline.log"
        prov_path = out_dir / f"{BRAND}_provenance.json"

        cols = st.columns(2)
        with cols[0]:
            st.subheader("Log")
            if log_path.exists():
                st.code(safe_read_text(log_path), language="text")
                download_button_for(log_path, "Download log")
            else:
                st.info("No log file found.")

        with cols[1]:
            st.subheader("Provenance")
            if prov_path.exists():
                prov_text = safe_read_text(prov_path)
                try:
                    st.json(json.loads(prov_text))
                except Exception:
                    st.code(prov_text, language="json")
                download_button_for(prov_path, "Download provenance.json")
            else:
                st.info("No provenance file found.")

        # -----------------------------------------------------------------------------
        # Outputs table + previews
        # -----------------------------------------------------------------------------
        st.subheader("Outputs")
        prefix = BRAND + "_"
        outputs = {
            "points_shp": out_dir / f"{prefix}robust_unstable_points.shp",
            "points_csv": out_dir / f"{prefix}robust_unstable_points.csv",
            "clusters_shp": out_dir / f"{prefix}robust_unstable_cluster_polygons.shp",
            "cluster_stats_csv": out_dir / f"{prefix}robust_unstable_cluster_stats.csv",
            "contingency_xlsx": out_dir / f"{prefix}contingency_table.xlsx",
            "anomaly_summary_xlsx": out_dir / f"{prefix}anomaly_overlap_summary.xlsx",
            "twi_summary_csv": out_dir / f"{prefix}twi_inside_outside_summary.csv",
            "hazard_index_tif": out_dir / f"{prefix}hazard_index.tif",
            "hazard_class_tif": out_dir / f"{prefix}hazard_class.tif",
            "unstable_classified_shp": out_dir / f"{prefix}unstable_slope_classified.shp",
            "hazard_zonal_stats_xlsx": out_dir / f"{prefix}hazard_zonal_stats.xlsx",
            "provenance_json": out_dir / f"{prefix}provenance.json",
            "pipeline_log": out_dir / f"{prefix}pipeline.log",
        }

        table_data = []
        for k, p in outputs.items():
            exists = p.exists()
            table_data.append({"Output": k, "Exists": "Yes" if exists else "No", "Path": str(p if exists else "")})
        try:
            st.dataframe(pd.DataFrame(table_data), use_container_width=True)
        except Exception as e:
            st.warning(f"DataFrame preview unavailable ({e}). Showing plain text instead.")
            st.text(pd.DataFrame(table_data).to_string(index=False))

        st.markdown("**Downloads**")
        dcols = st.columns(3); i = 0
        for k, p in outputs.items():
            if p.exists():
                with dcols[i % 3]:
                    download_button_for(p, label=f"Download {p.name}")
                i += 1

        pcols = st.columns(2)
        if outputs["hazard_index_tif"].exists():
            with pcols[0]:
                if rasterio and plt:
                    with rasterio.open(outputs["hazard_index_tif"]) as ds:
                        arr = ds.read(1).astype(float)
                        if ds.nodata is not None: arr[arr == ds.nodata] = np.nan
                    fig = plt.figure(figsize=(6,4)); ax = fig.add_subplot(111)
                    im = ax.imshow(arr, interpolation="nearest"); ax.set_title("Hazard Index"); ax.axis("off")
                    fig.colorbar(im, ax=ax, shrink=0.8); st.pyplot(fig); plt.close(fig)
        if outputs["hazard_class_tif"].exists():
            with pcols[1]:
                preview_hazard_class_raster_categorical(outputs["hazard_class_tif"], "Hazard Class (Low/Moderate/High)")

        pcols2 = st.columns(2)
        if outputs["clusters_shp"].exists():
            with pcols2[0]:
                preview_vectors(outputs["clusters_shp"], "Unstable Cluster Polygons", color="#8e44ad")
        if outputs["unstable_classified_shp"].exists():
            with pcols2[1]:
                preview_classified_polygons_color(outputs["unstable_classified_shp"], "Unstable Polygons (Zonal Class)")

else:
    st.info("Configure inputs on the left and click **Run Pipeline** to begin.")
    st.caption("Tip: point the Output Folder to a fast local drive (e.g., D:\\) for large runs on Windows.")
