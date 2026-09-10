"""
WATERSCOPE-AI — Satellite Temporal Dynamics Module
Analyzes multi-temporal raster differences (NDVI/NDWI changes) and classifies water dynamics:
Persistent Water, Water Gain (Inundation/Filling), Water Loss (Desiccation/Drying), and Land.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union
import numpy as np
import rasterio
from rasterio.transform import from_origin

DEFAULT_NODATA = -9999.0

# Dynamic Water Class Codes
WATER_DYNAMICS_CLASSES: Dict[int, str] = {
    0: "Persistent Land / Non-Water",
    1: "Persistent Water",
    2: "Water Gain (Inundated / Constructed)",
    3: "Water Loss (Desiccated / Demolished)",
    255: "NoData",
}

def compute_temporal_difference(
    raster_t0: np.ndarray,
    raster_t1: np.ndarray,
    nodata_value: float = DEFAULT_NODATA
) -> np.ndarray:
    """
    Computes pixel-wise difference: (Raster_T1 - Raster_T0).

    Args:
        raster_t0: Initial time raster (2D float).
        raster_t1: Subsequent time raster (2D float).
        nodata_value: Value representing nodata.

    Returns:
        np.ndarray: Difference raster with nodata preserved.
    """
    if raster_t0.shape != raster_t1.shape:
        raise ValueError(
            f"Shape mismatch: T0 {raster_t0.shape} vs T1 {raster_t1.shape}"
        )

    diff = np.full(raster_t0.shape, nodata_value, dtype=np.float32)
    valid = (
        (raster_t0 != nodata_value) & np.isfinite(raster_t0) &
        (raster_t1 != nodata_value) & np.isfinite(raster_t1)
    )

    diff[valid] = raster_t1[valid] - raster_t0[valid]
    return diff

def detect_water_dynamics(
    ndwi_t0: np.ndarray,
    ndwi_t1: np.ndarray,
    threshold: float = 0.0,
    pixel_area_m2: float = 100.0,
    nodata_value: float = DEFAULT_NODATA
) -> Dict[str, Any]:
    """
    Classifies temporal water surface changes into 4 dynamic categories:
    - 0: Persistent Non-Water (<= threshold at T0 and T1)
    - 1: Persistent Water (> threshold at T0 and T1)
    - 2: Water Gain (<= threshold at T0, > threshold at T1)
    - 3: Water Loss (> threshold at T0, <= threshold at T1)
    - 255: NoData
    """
    if ndwi_t0.shape != ndwi_t1.shape:
        raise ValueError(f"Shape mismatch: {ndwi_t0.shape} vs {ndwi_t1.shape}")

    valid = (
        (ndwi_t0 != nodata_value) & np.isfinite(ndwi_t0) &
        (ndwi_t1 != nodata_value) & np.isfinite(ndwi_t1)
    )

    dynamics = np.full(ndwi_t0.shape, 255, dtype=np.uint8)

    w0 = ndwi_t0 > threshold
    w1 = ndwi_t1 > threshold

    # Apply classifications
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

def save_dynamics_geotiff(
    output_path: Union[str, Path],
    dynamics_raster: np.ndarray,
    transform: Optional[rasterio.Affine] = None,
    crs: Optional[Union[str, rasterio.crs.CRS]] = "EPSG:4326"
) -> Path:
    """
    Saves the temporal dynamics classification raster (uint8) as a GeoTIFF.
    """
    out_p = Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    h, w = dynamics_raster.shape[:2]
    if transform is None:
        transform = from_origin(85.0, 20.0, 0.0001, 0.0001)

    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": "uint8",
        "crs": crs,
        "transform": transform,
        "nodata": 255,
        "compress": "lzw",
    }

    with rasterio.open(out_p, "w", **profile) as dst:
        dst.write(dynamics_raster.astype(np.uint8), 1)

    return out_p
