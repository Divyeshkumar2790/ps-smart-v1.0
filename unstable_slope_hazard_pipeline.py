# -*- coding: utf-8 -*-
from __future__ import annotations
r"""
PS-SMaRT: Unstable-Slope Landslide Hazard Pipeline
(with optional LOS→downslope projection and hazard blending)
----------------------------------------------------------------

Dependencies (pip/conda):
  geopandas rasterio shapely pyproj scikit-learn scipy pandas numpy affine

Typical CLI:
  # (A) Using ALREADY-PROJECTED points CSV:
  python unstable_slope_hazard_pipeline.py ^
      --out_dir "D:\Work\Pipeline_Outputs" ^
      --points_csv "D:\...\projected_timeseries_SAO_DESC.csv" ^
      --slope_tif "D:\...\Slope_1m.tif"

  # (B) Project raw LOS → downslope first, then full pipeline:
  python unstable_slope_hazard_pipeline.py ^
      --out_dir "D:\Work\Pipeline_Outputs" ^
      --raw_timeseries_csv "D:\...\SAO_DESC.csv" ^
      --slope_tif "D:\...\Slope_1m.tif" ^
      --aspect_tif "D:\...\Aspect_1m.tif" ^
      --default_heading_deg -10.82 ^
      --default_incidence_deg 31.53 ^
      --downhill_negative ^
      --require_negative_v

Outputs (prefixed with BRAND, default 'PS-SMaRT'):
  - PS-SMaRT_robust_unstable_points.(shp|csv)
  - PS-SMaRT_robust_unstable_cluster_polygons.shp
  - PS-SMaRT_robust_unstable_cluster_stats.csv
  - PS-SMaRT_hazard_index.tif, PS-SMaRT_hazard_class.tif
  - PS-SMaRT_unstable_slope_classified.shp
  - PS-SMaRT_hazard_zonal_stats.xlsx
  - PS-SMaRT_contingency_table.xlsx, PS-SMaRT_anomaly_overlap_summary.xlsx (if anomaly)
  - PS-SMaRT_twi_inside_outside_summary.csv (if TWI)
  - PS-SMaRT_pipeline.log, PS-SMaRT_provenance.json
"""

# ---------- stdlib ----------
import os
import math
import json
import argparse
import logging
import shutil
import platform
from typing import Optional, Tuple, Dict, Union
from pathlib import Path
from datetime import datetime, timezone

# ---------- third-party ----------
from affine import Affine
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry import Point, MultiPoint
from shapely.ops import unary_union
from shapely.strtree import STRtree
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling
from sklearn.cluster import DBSCAN
from scipy.stats import chi2_contingency
from sklearn.metrics import matthews_corrcoef

# ---------- branding ----------
BRAND = "PS-SMaRT"
PS_PREFIX = BRAND + "_"


# =========================
# Utilities & Logging
# =========================
def _ensure_dir(p: Union[str, Path]) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def clean_dir(p: Union[str, Path]) -> None:
    """Remove all contents of a directory (but keep the directory)."""
    d = Path(p)
    if not d.exists():
        d.mkdir(parents=True, exist_ok=True)
        return
    for child in d.iterdir():
        try:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink(missing_ok=True)
        except Exception:
            # best-effort cleanup; continue
            pass


def setup_logging(out_dir: Union[str, Path], file_prefix: str = BRAND) -> logging.Logger:
    out_dir = _ensure_dir(out_dir)
    log_path = out_dir / f"{file_prefix}_pipeline.log"
    logger = logging.getLogger("hazard_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    ch = logging.StreamHandler(); ch.setFormatter(fmt); ch.setLevel(logging.INFO)
    fh = logging.FileHandler(str(log_path), mode="w", encoding="utf-8")
    fh.setFormatter(fmt); fh.setLevel(logging.INFO)
    logger.addHandler(ch); logger.addHandler(fh)
    logger.info("[%s] Logging to %s", file_prefix, log_path)
    return logger


# =========================
# Sampling helpers
# =========================
def inverse_map(transform: Affine, x: float, y: float) -> Tuple[float, float]:
    col, row = ~transform * (x, y)
    return float(row), float(col)


def bilinear_sample_scalar(ds: rasterio.io.DatasetReader, x: float, y: float) -> float:
    """Bilinear sampling for scalars. Returns np.nan if OOB or nodata."""
    nodata = ds.nodata
    transform = ds.transform
    rows, cols = ds.height, ds.width

    r_f, c_f = inverse_map(transform, x, y)
    r0 = int(math.floor(r_f)); c0 = int(math.floor(c_f))
    r1 = r0 + 1;              c1 = c0 + 1
    if r0 < 0 or c0 < 0 or r1 >= rows or c1 >= cols:
        return np.nan

    dr = r_f - r0; dc = c_f - c0
    v00 = ds.read(1, window=((r0, r0+1), (c0, c0+1)))[0, 0]
    v10 = ds.read(1, window=((r1, r1+1), (c0, c0+1)))[0, 0]
    v01 = ds.read(1, window=((r0, r0+1), (c1, c1+1)))[0, 0]
    v11 = ds.read(1, window=((r1, r1+1), (c1, c1+1)))[0, 0]
    vals = np.array([v00, v10, v01, v11], dtype=float)
    if nodata is not None:
        vals[vals == nodata] = np.nan
    if not np.all(np.isfinite(vals)):
        return np.nan
    v00, v10, v01, v11 = vals
    return float(v00*(1-dr)*(1-dc) + v10*dr*(1-dc) + v01*(1-dr)*dc + v11*dr*dc)


def bilinear_sample_aspect(ds: rasterio.io.DatasetReader, x: float, y: float) -> float:
    """Bilinear sampling for ASPECT (circular) via vector averaging, degrees 0..360."""
    nodata = ds.nodata
    transform = ds.transform
    rows, cols = ds.height, ds.width

    r_f, c_f = inverse_map(transform, x, y)
    r0 = int(math.floor(r_f)); c0 = int(math.floor(c_f))
    r1 = r0 + 1;              c1 = c0 + 1
    if r0 < 0 or c0 < 0 or r1 >= rows or c1 >= cols:
        return np.nan

    dr = r_f - r0; dc = c_f - c0
    A00 = ds.read(1, window=((r0, r0+1), (c0, c0+1)))[0, 0]
    A10 = ds.read(1, window=((r1, r1+1), (c0, c0+1)))[0, 0]
    A01 = ds.read(1, window=((r0, r0+1), (c1, c1+1)))[0, 0]
    A11 = ds.read(1, window=((r1, r1+1), (c1, c1+1)))[0, 0]
    angles = np.array([A00, A10, A01, A11], dtype=float)
    if nodata is not None:
        angles[angles == nodata] = np.nan
    if not np.all(np.isfinite(angles)):
        return np.nan

    ang = np.deg2rad(angles)
    cosc, sinc = np.cos(ang), np.sin(ang)
    w00 = (1-dr)*(1-dc); w10 = dr*(1-dc); w01 = (1-dr)*dc; w11 = dr*dc
    C = w00*cosc[0] + w10*cosc[1] + w01*cosc[2] + w11*cosc[3]
    S = w00*sinc[0] + w10*sinc[1] + w01*sinc[2] + w11*sinc[3]
    return float((math.degrees(math.atan2(S, C)) + 360.0) % 360.0)


# =========================
# Geometry: LOS & slope vectors
# =========================
def los_unit_vector(heading_deg: float, incidence_from_vertical_deg: float) -> np.ndarray:
    """LOS unit vector (toward sensor) in ENU, incidence measured FROM VERTICAL."""
    phi = math.radians((heading_deg + 90.0) % 360.0)   # look azimuth
    theta = math.radians(incidence_from_vertical_deg)  # from vertical
    l_e = -math.sin(phi) * math.sin(theta)
    l_n = -math.cos(phi) * math.sin(theta)
    l_u =  math.cos(theta)
    v = np.array([l_e, l_n, l_u], dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def downslope_unit_vector(aspect_deg_north_cw: float, slope_deg_from_horizontal: float) -> np.ndarray:
    """Downslope unit vector in ENU (aspect: 0°=North, clockwise)."""
    A = math.radians(aspect_deg_north_cw)
    S = math.radians(slope_deg_from_horizontal)
    s_e = math.sin(A) * math.cos(S)
    s_n = math.cos(A) * math.cos(S)
    s_u = -math.sin(S)
    v = np.array([s_e, s_n, s_u], dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def convert_aspect_to_north_cw(aspect_deg: float, convention: str) -> float:
    """Return aspect in 0°=North, clockwise."""
    if np.isnan(aspect_deg):
        return np.nan
    if convention == "north_cw":
        return float(aspect_deg % 360.0)
    elif convention == "east_ccw":
        return float((90.0 - aspect_deg) % 360.0)
    else:
        raise ValueError(f"Unsupported ASPECT_CONVENTION={convention}")


# =========================
# STRtree helper (indices vs geometries)
# =========================
def _tree_candidates(cix, geoms):
    """Normalize STRtree.query results to geometries (handles index/geom variants)."""
    if cix is None:
        return ()
    arr = np.asarray(cix, dtype=object)
    if arr.size == 0:
        return ()
    if np.issubdtype(arr.dtype, np.integer) or isinstance(arr.flat[0], (int, np.integer)):
        return (geoms[int(i)] for i in arr.tolist())
    return arr.tolist()


# =========================
# STEP 0: LOS → downslope projection (optional)
# =========================
def project_points_along_slope(
    raw_timeseries_csv: str,
    slope_tif: str,
    aspect_tif: str,
    out_csv: str,
    aspect_convention: str = "north_cw",
    heading_raster: Optional[str] = None,
    incidence_raster: Optional[str] = None,
    default_heading_deg: float = -10.82,  # DESC default (clockwise from North)
    default_incidence_deg: float = 31.53, # FROM VERTICAL
    min_slope_deg: float = 10.0,
    min_sensitivity: float = 0.25,
    eps_div: float = 1e-3,
    logger: Optional[logging.Logger] = None,
    *,
    downhill_negative: bool = False,   # make downslope values negative if True
    require_negative_v: bool = False,  # keep only v_parallel < 0 if True
) -> str:
    """
    Reads raw time-series CSV (LAT, LON, VEL, D* columns) and writes a projected CSV
    with time-series and velocity along local downslope direction.
    """
    log = logger or logging.getLogger("hazard_pipeline")
    df = pd.read_csv(raw_timeseries_csv, sep=None, engine="python")
    for rc in ["ID", "LAT", "LON", "VEL"]:
        if rc not in df.columns:
            raise ValueError(f"Required column '{rc}' not found in {raw_timeseries_csv}")
    time_cols = [c for c in df.columns if c.startswith("D")]

    slope_ds  = rasterio.open(slope_tif)
    aspect_ds = rasterio.open(aspect_tif)
    if aspect_ds.crs != slope_ds.crs:
        raise ValueError("Aspect CRS differs from slope CRS.")
    crs = slope_ds.crs

    heading_ds = rasterio.open(heading_raster) if heading_raster else None
    incid_ds   = rasterio.open(incidence_raster) if incidence_raster else None
    if heading_ds and heading_ds.crs != crs:
        raise ValueError("Heading raster CRS differs.")
    if incid_ds and incid_ds.crs != crs:
        raise ValueError("Incidence raster CRS differs.")

    from pyproj import Transformer
    transformer = Transformer.from_crs("epsg:4326", crs, always_xy=True)

    out_rows = []
    total = len(df); kept = 0
    for _, r in df.iterrows():
        lat, lon = float(r["LAT"]), float(r["LON"])
        vel_los  = float(r["VEL"])
        x, y = transformer.transform(lon, lat)

        # quick bounds
        if not (slope_ds.bounds.left <= x <= slope_ds.bounds.right and
                slope_ds.bounds.bottom <= y <= slope_ds.bounds.top):
            continue

        slope_val = bilinear_sample_scalar(slope_ds, x, y)
        aspect_val_raw = bilinear_sample_aspect(aspect_ds, x, y)
        if not np.isfinite(slope_val) or not np.isfinite(aspect_val_raw):
            continue
        if slope_val < min_slope_deg:
            continue
        aspect_val = convert_aspect_to_north_cw(aspect_val_raw, aspect_convention)

        # heading/incidence at point
        heading = bilinear_sample_scalar(heading_ds, x, y) if heading_ds else default_heading_deg
        incid   = bilinear_sample_scalar(incid_ds, x, y)   if incid_ds   else default_incidence_deg
        if not np.isfinite(heading) or not np.isfinite(incid):
            continue

        l_vec = los_unit_vector(heading, incid)
        s_vec = downslope_unit_vector(aspect_val, slope_val)
        sens  = float(np.dot(l_vec, s_vec))
        if abs(sens) < max(min_sensitivity, eps_div):
            continue

        denom = sens if abs(sens) >= eps_div else math.copysign(eps_div, sens)
        scale = -1.0 if downhill_negative else 1.0
        ts_vals = r[time_cols].to_numpy(dtype=float, copy=False)
        repr_series = scale * (ts_vals / denom)
        vel_repr    = scale * (vel_los / denom)

        if require_negative_v and not (vel_repr < 0):
            continue

        out = {
            "ID": r["ID"], "LAT": lat, "LON": lon,
            "VEL_LOS": vel_los, "VEL_REPR": vel_repr,
            "SLOPE": slope_val, "ASPECT": aspect_val, "SENSITIVITY": sens,
            "HEADING_DEG": heading, "INCIDENCE_DEG": incid,
            "LOOK_AZ_DEG": (heading + 90.0) % 360.0
        }
        for c, v in zip(time_cols, repr_series):
            out[c] = v
        out_rows.append(out); kept += 1

    slope_ds.close(); aspect_ds.close()
    if heading_ds: heading_ds.close()
    if incid_ds: incid_ds.close()

    if kept == 0:
        raise RuntimeError("Projection produced zero points (check thresholds/inputs).")
    out_df = pd.DataFrame(out_rows)
    _ensure_dir(Path(out_csv).parent)
    out_df.to_csv(out_csv, index=False)
    log.info("Projected points saved: %s (kept %d / %d)", out_csv, kept, total)
    return out_csv


# =========================
# STEP A: Load & filter points (already projected)
# =========================
def load_and_filter_points(points_csv: str, slope_tif: str,
                           slope_deg_min: float, vel_abs_min: float,
                           logger: logging.Logger) -> gpd.GeoDataFrame:
    df = pd.read_csv(points_csv, sep=None, engine="python")
    for rc in ["ID", "LAT", "LON", "VEL_REPR"]:
        if rc not in df.columns:
            raise ValueError(f"Required column '{rc}' missing in {points_csv}")

    with rasterio.open(slope_tif) as ds:
        crs = ds.crs; bounds = ds.bounds
        from pyproj import Transformer
        transformer = Transformer.from_crs("epsg:4326", crs, always_xy=True)

        rows = []
        for _, r in df.iterrows():
            lon, lat = float(r["LON"]), float(r["LAT"])
            vel = float(r["VEL_REPR"])
            if not np.isfinite(vel):
                continue
            x, y = transformer.transform(lon, lat)
            if not (bounds.left <= x <= bounds.right and bounds.bottom <= y <= bounds.top):
                continue
            slope_val = bilinear_sample_scalar(ds, x, y)
            if not np.isfinite(slope_val):
                continue
            if slope_val >= slope_deg_min and abs(vel) >= vel_abs_min:
                rows.append({"ID": r["ID"], "LAT": lat, "LON": lon, "X": x, "Y": y,
                             "VEL_REPR": vel, "SLOPE": slope_val})

    if not rows:
        logger.warning("No points met slope/velocity thresholds.")
        return gpd.GeoDataFrame(columns=["ID","LAT","LON","X","Y","VEL_REPR","SLOPE","geometry"], crs=crs)

    gdf = gpd.GeoDataFrame(
        rows,
        geometry=[Point(xy) for xy in zip([k["X"] for k in rows], [k["Y"] for k in rows])],
        crs=crs
    )
    logger.info("Filtered points retained: %d", len(gdf))
    return gdf


# =========================
# STEP B: DBSCAN clustering on XY
# =========================
def cluster_points_dbscan(gdf_pts: gpd.GeoDataFrame, eps_m: float, min_samples: int,
                          logger: logging.Logger) -> gpd.GeoDataFrame:
    if gdf_pts.empty:
        return gdf_pts.assign(CLUSTER_ID=-1)
    XY = np.vstack([gdf_pts["X"].to_numpy(), gdf_pts["Y"].to_numpy()]).T
    db = DBSCAN(eps=eps_m, min_samples=min_samples)
    labels = db.fit_predict(XY)
    gdf = gdf_pts.copy()
    gdf["CLUSTER_ID"] = labels
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    logger.info("DBSCAN found %d clusters (eps=%.1fm, min_samples=%d)", n_clusters, eps_m, min_samples)
    return gdf


# =========================
# STEP C: Polygons & stats
# =========================
def clusters_to_polygons_and_stats(gdf_pts: gpd.GeoDataFrame, polish_buffer: float = 0.0):
    clusters = gdf_pts[gdf_pts["CLUSTER_ID"] != -1]
    if clusters.empty:
        return gpd.GeoDataFrame(columns=["CLUSTER_ID","geometry"], crs=gdf_pts.crs), pd.DataFrame()

    poly_rows, stats_rows = [], []
    for cid, grp in clusters.groupby("CLUSTER_ID"):
        mp = MultiPoint(list(grp.geometry))
        hull = mp.convex_hull
        if hull.geom_type in ("LineString", "Point"):
            hull = hull.buffer(0.1)  # meters (projected CRS)
        if polish_buffer > 0:
            hull = unary_union([pt.buffer(polish_buffer) for pt in grp.geometry]).buffer(0)
        hull = hull.buffer(0)  # clean topology

        stats_rows.append({
            "CLUSTER_ID": cid,
            "Num_MPs": int(len(grp)),
            "Mean_VEL_REPR": float(np.nanmean(grp["VEL_REPR"])),
            "Std_VEL_REPR": float(np.nanstd(grp["VEL_REPR"])),
            "Min_VEL_REPR": float(np.nanmin(grp["VEL_REPR"])),
            "Max_VEL_REPR": float(np.nanmax(grp["VEL_REPR"])),
            "Cluster_Area_m2": float(hull.area)
        })
        poly_rows.append({"CLUSTER_ID": cid, "geometry": hull})

    return gpd.GeoDataFrame(poly_rows, crs=gdf_pts.crs), pd.DataFrame(stats_rows)


# =========================
# STEP D: Anomaly overlap stats (centroid-based)
# =========================
def anomaly_overlap_stats(anomaly_tif: str, unstable_polys: gpd.GeoDataFrame,
                          out_dir: Union[str, Path], logger: logging.Logger,
                          out_prefix: str = PS_PREFIX) -> Dict[str, float]:
    try:
        if not anomaly_tif or not os.path.exists(anomaly_tif):
            logger.info("Skipping anomaly overlap: anomaly_tif missing.")
            return {}
        if unstable_polys is None or unstable_polys.empty:
            logger.info("Skipping anomaly overlap: unstable polygons are empty.")
            return {}

        with rasterio.open(anomaly_tif) as src:
            arr = src.read(1)
            tr = src.transform
            r_crs = src.crs
            nodata = src.nodata

        if unstable_polys.crs is None and r_crs is None:
            logger.warning("Both anomaly raster and polygons have undefined CRS. Skipping overlap.")
            return {}

        work_crs = r_crs if r_crs is not None else unstable_polys.crs
        polys = unstable_polys
        if polys.crs != work_crs:
            try:
                polys = polys.to_crs(work_crs)
            except Exception as e:
                logger.error("Failed to reproject polygons to anomaly CRS: %s", e)
                return {}

        polys = polys[polys.geometry.notnull()].copy()
        if polys.empty:
            logger.info("Skipping overlap: all polygons are null.")
            return {}
        polys["geometry"] = polys.geometry.buffer(0)
        polys = polys[~polys.geometry.is_empty]
        if polys.empty:
            logger.info("Skipping overlap: all polygons empty after fixing.")
            return {}

        geoms = list(polys.geometry)
        tree = STRtree(geoms)

        if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
            valid = np.isfinite(arr) & (arr != nodata)
        else:
            valid = np.isfinite(arr)

        if not np.any(valid):
            logger.info("Skipping overlap: no valid pixels in anomaly raster.")
            return {}

        rr, cc = np.where(valid)
        anom_present = (arr > 0)

        max_samples = 2_000_000
        if rr.size > max_samples:
            sel = np.random.default_rng(42).choice(rr.size, size=max_samples, replace=False)
            rr, cc = rr[sel], cc[sel]

        y_anom, y_slope = [], []
        for r, c in zip(rr, cc):
            x, y = tr * (c + 0.5, r + 0.5)
            pt = Point(x, y)
            y_anom.append(int(anom_present[r, c]))
            cix = tree.query(pt)
            candidates = _tree_candidates(cix, geoms)
            inside = any(g.contains(pt) for g in candidates)
            y_slope.append(int(inside))

        y_anom = np.asarray(y_anom, dtype=int)
        y_slope = np.asarray(y_slope, dtype=int)
        A = int(np.sum((y_anom == 1) & (y_slope == 1)))
        B = int(np.sum((y_anom == 1) & (y_slope == 0)))
        C = int(np.sum((y_anom == 0) & (y_slope == 1)))
        D = int(np.sum((y_anom == 0) & (y_slope == 0)))

        table = np.array([[A, B], [C, D]], dtype=int)
        chi2, p, dof, _ = chi2_contingency(table)
        mcc = matthews_corrcoef(y_slope, y_anom) if (A + B + C + D) > 0 else np.nan

        df_cont = pd.DataFrame(table, columns=["In Slopes", "Outside Slopes"], index=["Anomaly", "No Anomaly"])
        out_dir = _ensure_dir(out_dir)
        df_cont.to_excel(out_dir / f"{out_prefix}contingency_table.xlsx")
        pd.DataFrame(
            {"Metric": ["Chi2", "p_value", "dof", "Phi/MCC", "A", "B", "C", "D"],
             "Value":  [chi2,    p,        dof,   mcc,       A,   B,   C,   D]}
        ).to_excel(out_dir / f"{out_prefix}anomaly_overlap_summary.xlsx", index=False)

        logger.info("Anomaly overlap: chi2=%.3f, p=%.3g, MCC=%.3f (A=%d B=%d C=%d D=%d)", chi2, p, mcc, A, B, C, D)
        return {"chi2": chi2, "p": p, "dof": dof, "mcc": mcc, "A": A, "B": B, "C": C, "D": D}

    except Exception as e:
        logger.exception("anomaly_overlap_stats failed: %s", e)
        return {}


# =========================
# STEP E: TWI analysis
# =========================
def twi_inside_outside(twi_tif: str, unstable_polys: gpd.GeoDataFrame,
                       out_dir: Union[str, Path], logger: logging.Logger,
                       out_prefix: str = PS_PREFIX) -> Dict[str, float]:
    if not twi_tif or not os.path.exists(twi_tif) or unstable_polys.empty:
        logger.info("Skipping TWI analysis (missing inputs)."); return {}
    with rasterio.open(twi_tif) as src:
        twi = src.read(1); tr = src.transform; crs = src.crs; nodata = src.nodata
        polys = unstable_polys.to_crs(crs)
        masked, _ = mask(src, [geom for geom in polys.geometry], crop=False)
        inside = masked[0].astype(float)
        if nodata is not None:
            inside[inside == nodata] = np.nan
        twi_in = inside[np.isfinite(inside)].ravel()

        valid = np.isfinite(twi) & ((twi != nodata) if nodata is not None else True)
        rows, cols = np.where(valid)

        geoms = list(polys.geometry)
        tree = STRtree(geoms)
        rng = np.random.default_rng(42)
        order = rng.permutation(len(rows))
        twi_out = []
        for i in order:
            r, c = rows[i], cols[i]
            x, y = tr * (c + 0.5, r + 0.5)
            pt = Point(x, y)
            cix = tree.query(pt)
            candidates = _tree_candidates(cix, geoms)
            if not any(g.contains(pt) for g in candidates):
                twi_out.append(twi[r, c])
                if len(twi_out) >= len(twi_in): break

    import scipy.stats as st
    t_stat, p_val = st.ttest_ind(twi_in, twi_out, equal_var=False, nan_policy="omit")
    pd.DataFrame({"Metric":["Mean TWI (unstable)","Mean TWI (stable)","T-stat","p-value","N_in","N_out"],
                  "Value":[np.nanmean(twi_in), np.nanmean(twi_out), t_stat, p_val, len(twi_in), len(twi_out)]}
                ).to_csv(Path(out_dir) / f"{out_prefix}twi_inside_outside_summary.csv", index=False)
    logger.info("TWI inside vs outside: t=%.3f, p=%.3g (N=%d/%d)", t_stat, p_val, len(twi_in), len(twi_out))
    return {"t":t_stat, "p":p_val, "n_in":len(twi_in), "n_out":len(twi_out)}


# =========================
# STEP F: Hazard products
# =========================
def resample_to_match(src_array, src_transform, src_crs, ref_shape, ref_transform, ref_crs):
    dest = np.empty(ref_shape, dtype=np.float32)
    reproject(source=src_array, destination=dest,
              src_transform=src_transform, src_crs=src_crs,
              dst_transform=ref_transform, dst_crs=ref_crs,
              resampling=Resampling.bilinear)
    return dest


def robust_minmax_norm(arr: np.ndarray, pmin=5, pmax=95) -> np.ndarray:
    valid = arr[np.isfinite(arr)]
    if valid.size < 10: return np.full_like(arr, np.nan, dtype=float)
    lo = np.percentile(valid, pmin); hi = np.percentile(valid, pmax)
    out = (arr - lo) / (hi - lo + 1e-12)
    return np.clip(out, 0, 1)


def write_geotiff(path, arr, ref_meta, dtype, nodata):
    meta = ref_meta.copy()
    meta.update(count=1, dtype=dtype, nodata=nodata)
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(arr.astype(dtype), 1)


def hazard_index_products(out_dir: Union[str, Path], unstable_polys: gpd.GeoDataFrame,
                          wet_tif: Optional[str], twi_tif: Optional[str], slope_tif: str,
                          logger: logging.Logger, out_prefix: str = PS_PREFIX):
    out_dir = _ensure_dir(out_dir)
    with rasterio.open(slope_tif) as sds:
        slope = sds.read(1).astype(np.float32)
        if sds.nodata is not None: slope[slope == sds.nodata] = np.nan
        s_tr, s_crs, s_meta = sds.transform, sds.crs, sds.meta.copy()
    slope_norm = robust_minmax_norm(slope)

    if wet_tif and os.path.exists(wet_tif):
        with rasterio.open(wet_tif) as wds:
            wet = wds.read(1).astype(np.float32)
            if wds.nodata is not None: wet[wet == wds.nodata] = np.nan
            wet_res = resample_to_match(wet, wds.transform, wds.crs, slope.shape, s_tr, s_crs)
        wet_norm = robust_minmax_norm(wet_res)
    else:
        wet_norm = np.full_like(slope_norm, np.nan)

    if twi_tif and os.path.exists(twi_tif):
        with rasterio.open(twi_tif) as tds:
            twi = tds.read(1).astype(np.float32)
            if tds.nodata is not None: twi[twi == tds.nodata] = np.nan
            twi_res = resample_to_match(twi, tds.transform, tds.crs, slope.shape, s_tr, s_crs)
        twi_norm = robust_minmax_norm(twi_res)
    else:
        twi_norm = np.full_like(slope_norm, np.nan)

    layers = []
    for a in (wet_norm, twi_norm, slope_norm):
        if np.any(np.isfinite(a)):
            layers.append(a)

    if len(layers) == 0:
        logger.warning("No valid hazard layers found (all-NaN). Writing empty hazard rasters.")
        hazard_index = np.full_like(slope_norm, np.nan, dtype=float)
    else:
        stack = np.stack(layers, axis=0)
        hazard_index = np.nanmean(stack, axis=0)

    hazard_class = np.zeros_like(hazard_index, dtype=np.uint8)
    hazard_class[hazard_index <= 0.20] = 1
    hazard_class[(hazard_index > 0.20) & (hazard_index <= 0.40)] = 2
    hazard_class[hazard_index > 0.40] = 3

    hz_idx_tif = Path(out_dir) / f"{out_prefix}hazard_index.tif"
    hz_cls_tif = Path(out_dir) / f"{out_prefix}hazard_class.tif"
    write_geotiff(str(hz_idx_tif), hazard_index, s_meta, "float32", -9999.0)  # distinct nodata
    write_geotiff(str(hz_cls_tif), hazard_class, s_meta, "uint8", 0)

    unstable_classified = ""
    hz_zonal_xlsx = ""

    if not unstable_polys.empty:
        up = unstable_polys.to_crs(s_crs)
        zs = []
        with rasterio.open(hz_idx_tif) as hz:
            hz_nodata = hz.nodata
            for i, geom in enumerate(up.geometry):
                try:
                    masked, _ = mask(hz, [geom], crop=False)
                    vals = masked[0]
                    if hz_nodata is not None:
                        vals = vals[np.isfinite(vals) & (vals != hz_nodata)]
                    else:
                        vals = vals[np.isfinite(vals)]
                    mean_val = float(np.nanmean(vals)) if vals.size else np.nan
                except Exception:
                    mean_val = np.nan
                zs.append({"idx": i, "mean_hazard": mean_val, "geometry": geom})
        gdf_zonal = gpd.GeoDataFrame(zs, crs=s_crs)
        gdf_zonal["hazard_class"] = pd.cut(
            gdf_zonal["mean_hazard"], bins=[-0.01,0.20,0.40,1.00],
            labels=["Low","Moderate","High"]
        ).astype(str)
        unstable_classified = Path(out_dir) / f"{out_prefix}unstable_slope_classified.shp"
        gdf_zonal.to_file(unstable_classified)
        hz_zonal_xlsx = Path(out_dir) / f"{out_prefix}hazard_zonal_stats.xlsx"
        gdf_zonal.drop(columns="geometry").to_excel(hz_zonal_xlsx, index=False)

    logger.info("Hazard index/class written; zonal classification completed.")
    return str(hz_idx_tif), str(hz_cls_tif), str(unstable_classified), str(hz_zonal_xlsx)


# =========================
# Provenance (clean, semantic)
# =========================
def _write_provenance(
    out_dir: Union[str, Path],
    points_csv: Optional[str],
    raw_timeseries_csv: Optional[str],
    slope_tif: str,
    aspect_tif: Optional[str],
    aspect_convention: str,
    heading_raster: Optional[str],
    incidence_raster: Optional[str],
    default_heading_deg: float,
    default_incidence_deg: float,
    slope_threshold_deg: float,
    velocity_threshold_abs: float,
    dbscan_eps_m: float,
    dbscan_min_pts: int,
    polygon_polish_buffer: float,
    anomaly_tif: Optional[str],
    twi_tif: Optional[str],
    pre_unstable_shp: Optional[str],
    outputs: Dict[str, str],
    *,
    file_prefix: str = BRAND,
    downhill_negative: bool = False,
    require_negative_v: bool = False,
) -> None:
    # minimal raster meta for reproducibility
    try:
        with rasterio.open(slope_tif) as _s:
            slope_meta = {
                "crs": str(_s.crs),
                "width": _s.width,
                "height": _s.height,
                "dtype": str(_s.dtypes[0]),
                "transform": list(_s.transform)
            }
    except Exception:
        slope_meta = {}

    params = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "brand": {"name": BRAND, "file_prefix": file_prefix},
        "host": {
            "python": platform.python_version(),
            "system": platform.system(),
            "release": platform.release()
        },
        "inputs": {
            "out_dir": str(Path(out_dir).resolve()),
            "points_csv": points_csv,
            "raw_timeseries_csv": raw_timeseries_csv,
            "slope_tif": slope_tif,
            "aspect_tif": aspect_tif,
            "aspect_convention": aspect_convention,
            "heading_raster": heading_raster,
            "incidence_raster": incidence_raster
        },
        "projection_defaults": {
            "default_heading_deg": float(default_heading_deg),
            "default_incidence_deg": float(default_incidence_deg),
            "downhill_negative": bool(downhill_negative),
            "require_negative_v": bool(require_negative_v),
        },
        "processing": {
            "slope_threshold_deg": float(slope_threshold_deg),
            "velocity_threshold_abs_mm_yr": float(velocity_threshold_abs),
            "dbscan_eps_m": float(dbscan_eps_m),
            "dbscan_min_pts": int(dbscan_min_pts),
            "polygon_polish_buffer_m": float(polygon_polish_buffer)
        },
        "optional_layers": {
            "anomaly_tif": anomaly_tif,
            "twi_tif": twi_tif,
            "pre_unstable_shp": pre_unstable_shp
        },
        "raster_meta": {"slope": slope_meta},
        "outputs": outputs
    }

    def _drop_none(d):
        if isinstance(d, dict):
            return {k: _drop_none(v) for k, v in d.items() if v is not None}
        if isinstance(d, list):
            return [_drop_none(v) for v in d if v is not None]
        return d

    params = _drop_none(params)
    out_dir = _ensure_dir(out_dir)
    with open(Path(out_dir) / f"{file_prefix}_provenance.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)


# =========================
# Orchestrator
# =========================
def run_pipeline(
    out_dir: Union[str, Path],
    *,
    file_prefix: str = BRAND,                      # e.g., "PS-SMaRT"
    # Option 1: already projected CSV
    points_csv: Optional[str] = None,
    # Option 2: generate projected CSV from raw
    raw_timeseries_csv: Optional[str] = None,
    slope_tif: str = "",
    aspect_tif: Optional[str] = None,
    aspect_convention: str = "north_cw",
    heading_raster: Optional[str] = None,
    incidence_raster: Optional[str] = None,
    default_heading_deg: float = -10.82,
    default_incidence_deg: float = 31.53,  # from vertical
    # Filtering & clustering
    slope_threshold_deg: float = 10.0,
    velocity_threshold_abs: float = 10.0,  # mm/yr on |VEL_REPR|
    dbscan_eps_m: float = 100.0,
    dbscan_min_pts: int = 5,
    polygon_polish_buffer: float = 0.0,
    # Optional analysis layers
    anomaly_tif: Optional[str] = None,
    twi_tif: Optional[str] = None,
    pre_unstable_shp: Optional[str] = None,
    # Slide-style options
    downhill_negative: bool = False,
    require_negative_v: bool = False,
    # housekeeping
    clean_out_dir: bool = False,
):
    """
    Run the PS-SMaRT unstable-slope hazard pipeline.

    Parameters
    ----------
    out_dir : str | Path
        Destination folder for all outputs (rasters, vectors, logs, provenance).
    file_prefix : str
        Prefix applied to all artifact names (default: BRAND, e.g., "PS-SMaRT").
    """
    out_dir = _ensure_dir(out_dir)
    if clean_out_dir:
        clean_dir(out_dir)

    logger = setup_logging(out_dir, file_prefix)
    logger.info("[%s] === Unstable-slope hazard pipeline ===", file_prefix)

    # Basic input sanity
    if not slope_tif:
        raise ValueError("slope_tif is required.")
    if not (points_csv or raw_timeseries_csv):
        raise ValueError("Provide either points_csv (already projected) OR raw_timeseries_csv + aspect_tif.")
    if raw_timeseries_csv and not aspect_tif:
        raise ValueError("aspect_tif is required when raw_timeseries_csv is provided.")

    # STEP 0: projection if requested
    if raw_timeseries_csv:
        projected_csv = str(Path(out_dir) / f"{file_prefix}_points_projected.csv")
        points_csv = project_points_along_slope(
            raw_timeseries_csv=raw_timeseries_csv,
            slope_tif=slope_tif,
            aspect_tif=aspect_tif,
            out_csv=projected_csv,
            aspect_convention=aspect_convention,
            heading_raster=heading_raster,
            incidence_raster=incidence_raster,
            default_heading_deg=default_heading_deg,
            default_incidence_deg=default_incidence_deg,
            logger=logger,
            downhill_negative=downhill_negative,
            require_negative_v=require_negative_v,
        )
    elif not points_csv:
        raise ValueError("Provide either points_csv (already projected) or raw_timeseries_csv + slope/aspect.")

    # STEP A: filter
    gdf_pts = load_and_filter_points(points_csv, slope_tif, slope_threshold_deg, velocity_threshold_abs, logger)
    if gdf_pts.empty:
        logger.warning("No filtered points. Exiting.")
        _write_provenance(
            out_dir, points_csv, raw_timeseries_csv, slope_tif, aspect_tif,
            aspect_convention, heading_raster, incidence_raster,
            default_heading_deg, default_incidence_deg,
            slope_threshold_deg, velocity_threshold_abs,
            dbscan_eps_m, dbscan_min_pts, polygon_polish_buffer,
            anomaly_tif, twi_tif, pre_unstable_shp,
            outputs={},
            file_prefix=file_prefix,
            downhill_negative=downhill_negative,
            require_negative_v=require_negative_v,
        )
        return

    # STEP B: cluster
    gdf_pts = cluster_points_dbscan(gdf_pts, dbscan_eps_m, dbscan_min_pts, logger)

    # STEP C: polygons + stats
    polys, stats = clusters_to_polygons_and_stats(gdf_pts, polish_buffer=polygon_polish_buffer)

    # Save clustered outputs (guard empty sets) with prefix
    pts_out   = Path(out_dir) / f"{file_prefix}_robust_unstable_points.shp"
    clust_out = Path(out_dir) / f"{file_prefix}_robust_unstable_cluster_polygons.shp"
    stats_out = Path(out_dir) / f"{file_prefix}_robust_unstable_cluster_stats.csv"
    csv_pts   = Path(out_dir) / f"{file_prefix}_robust_unstable_points.csv"

    clustered_pts = gdf_pts[gdf_pts["CLUSTER_ID"] != -1]
    if not clustered_pts.empty:
        clustered_pts.to_file(pts_out)
    else:
        pts_out = Path("")  # mark missing

    if not polys.empty:
        polys.to_file(clust_out)
    else:
        clust_out = Path("")

    if not stats.empty:
        stats.to_csv(stats_out, index=False)
    else:
        stats_out = Path("")

    gdf_pts.drop(columns="geometry").to_csv(csv_pts, index=False)
    logger.info("Saved: points_csv=%s | clusters_shp=%s | stats_csv=%s", csv_pts, clust_out, stats_out)

    # Choose unstable polygons layer for subsequent analyses
    if pre_unstable_shp and os.path.exists(pre_unstable_shp):
        unstable = gpd.read_file(pre_unstable_shp).to_crs(gdf_pts.crs)
    else:
        unstable = polys

    # STEP D/E: analyses
    if anomaly_tif: anomaly_overlap_stats(anomaly_tif, unstable, out_dir, logger, out_prefix=file_prefix + "_")
    if twi_tif:     twi_inside_outside(twi_tif, unstable, out_dir, logger, out_prefix=file_prefix + "_")

    # STEP F: hazard maps + unique per-polygon class
    hz_idx_tif, hz_cls_tif, unstable_classified, hz_zonal_xlsx = hazard_index_products(
        out_dir, unstable, anomaly_tif, twi_tif, slope_tif, logger, out_prefix=file_prefix + "_"
    )

    # Outputs dictionary (stringified)
    def _s(p): return str(p) if p else ""
    outputs = {
        "points_shp": _s(pts_out),
        "points_csv": _s(csv_pts),
        "clusters_shp": _s(clust_out),
        "cluster_stats_csv": _s(stats_out),
        "hazard_index_tif": _s(hz_idx_tif),
        "hazard_class_tif": _s(hz_cls_tif),
        "unstable_classified_shp": _s(unstable_classified),
        "hazard_zonal_stats_xlsx": _s(hz_zonal_xlsx),
        "contingency_xlsx": _s(Path(out_dir) / f"{file_prefix}_contingency_table.xlsx"),
        "anomaly_summary_xlsx": _s(Path(out_dir) / f"{file_prefix}_anomaly_overlap_summary.xlsx"),
        "twi_summary_csv": _s(Path(out_dir) / f"{file_prefix}_twi_inside_outside_summary.csv"),
        "pipeline_log": _s(Path(out_dir) / f"{file_prefix}_pipeline.log"),
        "provenance_json": _s(Path(out_dir) / f"{file_prefix}_provenance.json"),
    }

    # Provenance
    _write_provenance(
        out_dir, points_csv, raw_timeseries_csv, slope_tif, aspect_tif,
        aspect_convention, heading_raster, incidence_raster,
        default_heading_deg, default_incidence_deg,
        slope_threshold_deg, velocity_threshold_abs,
        dbscan_eps_m, dbscan_min_pts, polygon_polish_buffer,
        anomaly_tif, twi_tif, pre_unstable_shp,
        outputs=outputs,
        file_prefix=file_prefix,
        downhill_negative=downhill_negative,
        require_negative_v=require_negative_v,
    )

    logger.info("Pipeline complete. Outputs in: %s", out_dir)


# =========================
# CLI
# =========================
def build_arg_parser():
    p = argparse.ArgumentParser(
        description=f"{BRAND}: Unstable-slope landslide hazard pipeline (with optional LOS→downslope projection)."
    )

    # I/O & out_dir
    p.add_argument("--out_dir", required=True, help="Destination folder for outputs.")
    p.add_argument("--file_prefix", default=BRAND, help="Prefix for all outputs (default: PS-SMaRT).")
    p.add_argument("--clean_out_dir", action="store_true", help="Empty out_dir before writing outputs.")

    # Option 1: already projected
    p.add_argument("--points_csv", default=None)

    # Option 2: generate projected from raw
    p.add_argument("--raw_timeseries_csv", default=None)
    p.add_argument("--slope_tif", required=True)
    p.add_argument("--aspect_tif", default=None)
    p.add_argument("--aspect_convention", default="north_cw", choices=["north_cw","east_ccw"])
    p.add_argument("--heading_raster", default=None)
    p.add_argument("--incidence_raster", default=None)
    p.add_argument("--default_heading_deg", type=float, default=-10.82)
    p.add_argument("--default_incidence_deg", type=float, default=31.53)

    # Thresholds / clustering
    p.add_argument("--slope_threshold_deg", type=float, default=10.0)
    p.add_argument("--velocity_threshold_abs", type=float, default=10.0)
    p.add_argument("--dbscan_eps_m", type=float, default=100.0)
    p.add_argument("--dbscan_min_pts", type=int, default=5)
    p.add_argument("--polygon_polish_buffer", type=float, default=0.0)

    # Optional analysis layers
    p.add_argument("--anomaly_tif", default=None)
    p.add_argument("--twi_tif", default=None)
    p.add_argument("--pre_unstable_shp", default=None)

    # Slide-style options
    p.add_argument("--downhill_negative", action="store_true",
                   help="Make downslope values negative (matches S3-02 slide convention).")
    p.add_argument("--require_negative_v", action="store_true",
                   help="Keep only points with v_parallel < 0 after projection.")

    return p


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # --- CLI path ---
        args = build_arg_parser().parse_args()
        if not args.points_csv and not args.raw_timeseries_csv:
            raise SystemExit("Provide either --points_csv or --raw_timeseries_csv (+ --aspect_tif).")
        if args.raw_timeseries_csv and not args.aspect_tif:
            raise SystemExit("--aspect_tif is required when using --raw_timeseries_csv.")

        run_pipeline(
            out_dir=args.out_dir,
            file_prefix=args.file_prefix,
            points_csv=args.points_csv,
            raw_timeseries_csv=args.raw_timeseries_csv,
            slope_tif=args.slope_tif,
            aspect_tif=args.aspect_tif,
            aspect_convention=args.aspect_convention,
            heading_raster=args.heading_raster,
            incidence_raster=args.incidence_raster,
            default_heading_deg=args.default_heading_deg,
            default_incidence_deg=args.default_incidence_deg,
            slope_threshold_deg=args.slope_threshold_deg,
            velocity_threshold_abs=args.velocity_threshold_abs,
            dbscan_eps_m=args.dbscan_eps_m,
            dbscan_min_pts=args.dbscan_min_pts,
            polygon_polish_buffer=args.polygon_polish_buffer,
            anomaly_tif=args.anomaly_tif if args.anomaly_tif and os.path.exists(args.anomaly_tif) else None,
            twi_tif=args.twi_tif if args.twi_tif and os.path.exists(args.twi_tif) else None,
            pre_unstable_shp=args.pre_unstable_shp if args.pre_unstable_shp and os.path.exists(args.pre_unstable_shp) else None,
            downhill_negative=bool(args.downhill_negative),
            require_negative_v=bool(args.require_negative_v),
            clean_out_dir=bool(args.clean_out_dir),
        )
    else:
        # --- Example (edit for local testing) ---
        run_pipeline(
            out_dir=r"D:\Temp\ps_smart_outputs",
            file_prefix=BRAND,
            points_csv=None,  # using RAW → projection
            raw_timeseries_csv=r"D:\Data\CSK_DESC.csv",
            slope_tif=r"D:\Data\Slope_1m.tif",
            aspect_tif=r"D:\Data\Aspect_1m.tif",
            aspect_convention="north_cw",
            default_heading_deg=-10.5461,   # deg from North (clockwise)
            default_incidence_deg=33.929,   # deg FROM VERTICAL
            slope_threshold_deg=10.0,
            velocity_threshold_abs=5.0,
            dbscan_eps_m=50.0,
            dbscan_min_pts=10,
            polygon_polish_buffer=0.0,
            anomaly_tif=None,
            twi_tif=None,
            pre_unstable_shp=None,
            downhill_negative=False,
            require_negative_v=False,
            clean_out_dir=False,
        )
