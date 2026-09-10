"""
WATERSCOPE-AI — Sentinel-2 Multi-Temporal Change Detection Module
Provides modular, memory-conscious raster processing for multi-temporal (T0 vs T1)
Sentinel-2 biophysical analysis:
1. Spatial alignment validation across temporal pairs.
2. NDVI difference (T1 - T0) and classified vegetation dynamics.
3. NDWI difference (T1 - T0) and classified moisture dynamics.
4. Relative change calculation where mathematically valid.
5. Dynamic surface water classification (water gain, water loss, persistent water, persistent land),
   deriving surface areas directly from raster pixel resolutions.
6. Memory-conscious GeoTIFF export (LZW compression).
7. Structured JSON summary serialization.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
import json
import numpy as np
import rasterio
from rasterio.transform import from_origin

DEFAULT_NODATA = -9999.0

# ─────────────────────────────────────────────────────────────────────────────
# 1. Spatial Pair Validation
# ─────────────────────────────────────────────────────────────────────────────
def validate_temporal_pair(meta_t0: Dict[str, Any], meta_t1: Dict[str, Any]) -> None:
    """
    Strictly verifies that T0 and T1 observation rasters share identical dimensions,
    CRS, affine transform, and spatial resolution.
    Fails with ValueError if any spatial discrepancy is detected.
    """
    if meta_t0.get("width") != meta_t1.get("width"):
        raise ValueError(
            f"Temporal width mismatch: T0 has {meta_t0.get('width')} px, "
            f"but T1 has {meta_t1.get('width')} px."
        )
    if meta_t0.get("height") != meta_t1.get("height"):
        raise ValueError(
            f"Temporal height mismatch: T0 has {meta_t0.get('height')} px, "
            f"but T1 has {meta_t1.get('height')} px."
        )
    if str(meta_t0.get("crs")) != str(meta_t1.get("crs")):
        raise ValueError(
            f"Temporal CRS mismatch: T0 is {meta_t0.get('crs')}, "
            f"but T1 is {meta_t1.get('crs')}."
        )
    if meta_t0.get("transform") != meta_t1.get("transform"):
        raise ValueError("Temporal affine transform mismatch between T0 and T1 rasters.")
    if meta_t0.get("res") != meta_t1.get("res"):
        raise ValueError(
            f"Temporal resolution mismatch: T0 has {meta_t0.get('res')}, "
            f"but T1 has {meta_t1.get('res')}."
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Difference & Relative Change Arithmetic
# ─────────────────────────────────────────────────────────────────────────────
def calculate_ndvi_difference(
    ndvi_t0: np.ndarray,
    ndvi_t1: np.ndarray,
    nodata_value: float = DEFAULT_NODATA,
) -> np.ndarray:
    """
    Computes absolute NDVI change:
        NDVI_change = NDVI_T1 - NDVI_T0
    Preserves nodata and NaN values cleanly.
    """
    if ndvi_t0.shape != ndvi_t1.shape:
        raise ValueError(f"Shape mismatch: T0 {ndvi_t0.shape} vs T1 {ndvi_t1.shape}")

    diff = np.full(ndvi_t0.shape, nodata_value, dtype=np.float32)
    valid = (
        (ndvi_t0 != nodata_value) & np.isfinite(ndvi_t0) &
        (ndvi_t1 != nodata_value) & np.isfinite(ndvi_t1)
    )
    diff[valid] = ndvi_t1[valid] - ndvi_t0[valid]
    return diff


def calculate_ndwi_difference(
    ndwi_t0: np.ndarray,
    ndwi_t1: np.ndarray,
    nodata_value: float = DEFAULT_NODATA,
) -> np.ndarray:
    """
    Computes absolute NDWI change:
        NDWI_change = NDWI_T1 - NDWI_T0
    Preserves nodata and NaN values cleanly.
    """
    if ndwi_t0.shape != ndwi_t1.shape:
        raise ValueError(f"Shape mismatch: T0 {ndwi_t0.shape} vs T1 {ndwi_t1.shape}")

    diff = np.full(ndwi_t0.shape, nodata_value, dtype=np.float32)
    valid = (
        (ndwi_t0 != nodata_value) & np.isfinite(ndwi_t0) &
        (ndwi_t1 != nodata_value) & np.isfinite(ndwi_t1)
    )
    diff[valid] = ndwi_t1[valid] - ndwi_t0[valid]
    return diff


def calculate_relative_change(
    raster_t0: np.ndarray,
    raster_t1: np.ndarray,
    nodata_value: float = DEFAULT_NODATA,
    eps: float = 1e-4,
) -> np.ndarray:
    """
    Calculates percentage relative change:
        Relative_Change (%) = ((T1 - T0) / (|T0| + eps)) * 100
    Only calculated for valid, non-nodata cells.
    """
    if raster_t0.shape != raster_t1.shape:
        raise ValueError(f"Shape mismatch: T0 {raster_t0.shape} vs T1 {raster_t1.shape}")

    rel_diff = np.full(raster_t0.shape, nodata_value, dtype=np.float32)
    valid = (
        (raster_t0 != nodata_value) & np.isfinite(raster_t0) &
        (raster_t1 != nodata_value) & np.isfinite(raster_t1)
    )
    rel_diff[valid] = ((raster_t1[valid] - raster_t0[valid]) / (np.abs(raster_t0[valid]) + eps)) * 100.0
    return rel_diff


# ─────────────────────────────────────────────────────────────────────────────
# 3. Categorical Change Classifications (Documented Heuristic Thresholds)
# ─────────────────────────────────────────────────────────────────────────────
# NDVI change classes:
# 1: Significant Vegetation Loss / Decline (delta_ndvi < threshold_decline)
# 2: Stable / Negligible Vegetation Change (threshold_decline <= delta_ndvi <= threshold_gain)
# 3: Significant Vegetation Growth / Regrowth (delta_ndvi > threshold_gain)
# 255: NoData
NDVI_CHANGE_CLASSES = {
    1: "Significant Vegetation Loss / Degradation",
    2: "Stable / Negligible Vegetation Change",
    3: "Significant Vegetation Growth / Regrowth",
    255: "NoData",
}

def classify_ndvi_change(
    ndvi_diff: np.ndarray,
    threshold_decline: float = -0.10,
    threshold_gain: float = 0.10,
    nodata_value: float = DEFAULT_NODATA,
) -> np.ndarray:
    """
    Classifies continuous NDVI difference into qualitative dynamic categories using
    configurable heuristic thresholds:
        - Loss (1): delta_ndvi < threshold_decline (default: -0.10)
        - Stable (2): threshold_decline <= delta_ndvi <= threshold_gain
        - Gain (3): delta_ndvi > threshold_gain (default: +0.10)
        - NoData (255)
    """
    classified = np.full(ndvi_diff.shape, 255, dtype=np.uint8)
    valid = (ndvi_diff != nodata_value) & np.isfinite(ndvi_diff)

    loss_mask = valid & (ndvi_diff < threshold_decline)
    gain_mask = valid & (ndvi_diff > threshold_gain)
    stable_mask = valid & (ndvi_diff >= threshold_decline) & (ndvi_diff <= threshold_gain)

    classified[loss_mask] = 1
    classified[stable_mask] = 2
    classified[gain_mask] = 3
    return classified


# NDWI change classes:
# 1: Desiccation / Drying (delta_ndwi < threshold_dry)
# 2: Stable Moisture / Negligible Change (threshold_dry <= delta_ndwi <= threshold_wet)
# 3: Inundation / Moistening (delta_ndwi > threshold_wet)
# 255: NoData
NDWI_CHANGE_CLASSES = {
    1: "Desiccation / Drying",
    2: "Stable Moisture / Negligible Change",
    3: "Inundation / Moistening",
    255: "NoData",
}

def classify_ndwi_change(
    ndwi_diff: np.ndarray,
    threshold_dry: float = -0.10,
    threshold_wet: float = 0.10,
    nodata_value: float = DEFAULT_NODATA,
) -> np.ndarray:
    """
    Classifies continuous NDWI difference into qualitative dynamic categories using
    configurable heuristic thresholds:
        - Drying (1): delta_ndwi < threshold_dry (default: -0.10)
        - Stable (2): threshold_dry <= delta_ndwi <= threshold_wet
        - Wetting / Inundation (3): delta_ndwi > threshold_wet (default: +0.10)
        - NoData (255)
    """
    classified = np.full(ndwi_diff.shape, 255, dtype=np.uint8)
    valid = (ndwi_diff != nodata_value) & np.isfinite(ndwi_diff)

    dry_mask = valid & (ndwi_diff < threshold_dry)
    wet_mask = valid & (ndwi_diff > threshold_wet)
    stable_mask = valid & (ndwi_diff >= threshold_dry) & (ndwi_diff <= threshold_wet)

    classified[dry_mask] = 1
    classified[stable_mask] = 2
    classified[wet_mask] = 3
    return classified


# ─────────────────────────────────────────────────────────────────────────────
# 4. Temporal Water Dynamics (Water Gain / Loss / Persistent)
# ─────────────────────────────────────────────────────────────────────────────
WATER_DYNAMICS_CATEGORIES = {
    0: "Persistent Land / Non-Water",
    1: "Persistent Water",
    2: "Water Gain (Inundation / New Water Body)",
    3: "Water Loss (Desiccation / Dried Up)",
    255: "NoData",
}

def calculate_temporal_water_dynamics(
    ndwi_t0: np.ndarray,
    ndwi_t1: np.ndarray,
    water_threshold: float = 0.0,
    pixel_area_m2: float = 100.0,
    nodata_value: float = DEFAULT_NODATA,
) -> Dict[str, Any]:
    """
    Classifies water surface change between two NDWI rasters:
    - 0: Persistent Non-Water (<= water_threshold at T0 and T1)
    - 1: Persistent Water (> water_threshold at T0 and T1)
    - 2: Water Gain (<= water_threshold at T0, > water_threshold at T1)
    - 3: Water Loss (> water_threshold at T0, <= water_threshold at T1)
    - 255: NoData

    Calculates land and water surface areas dynamically using the supplied pixel_area_m2.
    """
    if ndwi_t0.shape != ndwi_t1.shape:
        raise ValueError(f"Shape mismatch: T0 {ndwi_t0.shape} vs T1 {ndwi_t1.shape}")

    valid = (
        (ndwi_t0 != nodata_value) & np.isfinite(ndwi_t0) &
        (ndwi_t1 != nodata_value) & np.isfinite(ndwi_t1)
    )

    dynamics = np.full(ndwi_t0.shape, 255, dtype=np.uint8)

    w0 = ndwi_t0 > water_threshold
    w1 = ndwi_t1 > water_threshold

    c0 = valid & (~w0) & (~w1)  # Persistent Land
    c1 = valid & w0 & w1        # Persistent Water
    c2 = valid & (~w0) & w1     # Water Gain
    c3 = valid & w0 & (~w1)     # Water Loss

    dynamics[c0] = 0
    dynamics[c1] = 1
    dynamics[c2] = 2
    dynamics[c3] = 3

    valid_count = int(np.count_nonzero(valid))
    total_count = int(ndwi_t0.size)

    def stats_for(mask_cond, name):
        count = int(np.count_nonzero(mask_cond))
        pct = (count / valid_count * 100.0) if valid_count > 0 else 0.0
        area_m2 = count * pixel_area_m2
        return {
            "name": name,
            "pixel_count": count,
            "percentage": round(pct, 2),
            "area_m2": round(area_m2, 2),
            "area_hectares": round(area_m2 / 10000.0, 4),
        }

    return {
        "water_threshold": water_threshold,
        "pixel_area_m2": pixel_area_m2,
        "total_pixels": total_count,
        "valid_pixels": valid_count,
        "nodata_pixels": total_count - valid_count,
        "categories": {
            "persistent_land": stats_for(c0, "Persistent Land / Non-Water"),
            "persistent_water": stats_for(c1, "Persistent Water"),
            "water_gain": stats_for(c2, "Water Gain (Inundated / Constructed)"),
            "water_loss": stats_for(c3, "Water Loss (Desiccated / Demolished)"),
        },
        "dynamics_raster": dynamics,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Statistical Aggregations
# ─────────────────────────────────────────────────────────────────────────────
def calculate_change_statistics(
    diff_array: np.ndarray,
    threshold_negligible: float = 0.05,
    nodata_value: float = DEFAULT_NODATA,
) -> Dict[str, Any]:
    """
    Computes distribution statistics for a difference raster (min, max, mean, std,
    increased pixels, decreased pixels, negligible change pixels).
    """
    valid_mask = (diff_array != nodata_value) & np.isfinite(diff_array)
    valid_data = diff_array[valid_mask]
    total_pixels = int(diff_array.size)
    valid_pixels = int(valid_data.size)

    if valid_pixels == 0:
        return {
            "valid_pixels": 0,
            "nodata_pixels": total_pixels,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "pixels_increased": 0,
            "pixels_decreased": 0,
            "pixels_negligible": 0,
        }

    increased = int(np.count_nonzero(valid_data > threshold_negligible))
    decreased = int(np.count_nonzero(valid_data < -threshold_negligible))
    negligible = valid_pixels - (increased + decreased)

    return {
        "total_pixels": total_pixels,
        "valid_pixels": valid_pixels,
        "nodata_pixels": total_pixels - valid_pixels,
        "min": round(float(np.min(valid_data)), 4),
        "max": round(float(np.max(valid_data)), 4),
        "mean": round(float(np.mean(valid_data)), 4),
        "std": round(float(np.std(valid_data)), 4),
        "pixels_increased": increased,
        "percentage_increased": round((increased / valid_pixels) * 100.0, 2),
        "pixels_decreased": decreased,
        "percentage_decreased": round((decreased / valid_pixels) * 100.0, 2),
        "pixels_negligible": negligible,
        "percentage_negligible": round((negligible / valid_pixels) * 100.0, 2),
        "threshold_negligible": threshold_negligible,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. GeoTIFF Writer & JSON Summary
# ─────────────────────────────────────────────────────────────────────────────
def write_temporal_change_geotiff(
    output_path: Union[str, Path],
    array: np.ndarray,
    transform: rasterio.Affine,
    crs: Union[str, rasterio.crs.CRS],
    nodata_value: Optional[Union[int, float]] = DEFAULT_NODATA,
    dtype: Optional[str] = None,
) -> Path:
    """
    Writes a continuous or categorical difference raster to GeoTIFF with LZW compression.
    """
    out_p = Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    h, w = array.shape[:2]
    target_dtype = dtype or str(array.dtype)

    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": target_dtype,
        "crs": crs,
        "transform": transform,
        "nodata": nodata_value,
        "compress": "lzw",
    }

    with rasterio.open(out_p, "w", **profile) as dst:
        dst.write(array.astype(target_dtype), 1)

    return out_p


def write_temporal_change_summary(
    output_path: Union[str, Path],
    summary_data: Dict[str, Any],
) -> Path:
    """
    Serializes structured temporal change results into JSON format.
    """
    out_p = Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    return out_p
