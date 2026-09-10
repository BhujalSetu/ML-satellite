"""
WATERSCOPE-AI — Normalized Difference Vegetation Index (NDVI) Module
Calculates NDVI, filters nodata/zero-division, computes vegetation statistics, and saves GeoTIFFs.

Formula:
NDVI = (NIR - Red) / (NIR + Red)
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union
import numpy as np
import rasterio
from rasterio.transform import from_origin

DEFAULT_NODATA = -9999.0

def calculate_ndvi(
    nir: np.ndarray,
    red: np.ndarray,
    nodata_value: float = DEFAULT_NODATA
) -> np.ndarray:
    """
    Calculates the Normalized Difference Vegetation Index (NDVI) from NIR and Red bands.

    Args:
        nir: Near-Infrared band array (2D numpy array, numeric).
        red: Red band array (2D numpy array, numeric).
        nodata_value: Value to assign to invalid or zero-denominator pixels.

    Returns:
        np.ndarray: NDVI array (float32), with values in [-1.0, 1.0] and nodata_value for invalid cells.
    """
    if nir.shape != red.shape:
        raise ValueError(f"Shape mismatch: NIR {nir.shape} vs Red {red.shape}")

    nir_f = nir.astype(np.float32)
    red_f = red.astype(np.float32)

    numerator = nir_f - red_f
    denominator = nir_f + red_f

    # Create output array initialized to nodata_value
    ndvi = np.full(nir.shape, nodata_value, dtype=np.float32)

    # Valid mask: denominator != 0 and both bands are finite and non-negative (unless reflectance calibrated)
    valid = (denominator != 0) & np.isfinite(numerator) & np.isfinite(denominator)

    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi_calc = numerator / denominator
        # Clip to valid mathematical NDVI bounds [-1.0, 1.0]
        ndvi[valid] = np.clip(ndvi_calc[valid], -1.0, 1.0)

    return ndvi

def compute_ndvi_statistics(
    ndvi: np.ndarray,
    nodata_value: float = DEFAULT_NODATA
) -> Dict[str, Any]:
    """
    Computes statistical indicators and vegetation canopy classification from an NDVI raster.

    Args:
        ndvi: 2D NDVI array.
        nodata_value: Assigned nodata value.

    Returns:
        Dict[str, Any]: Dictionary of statistical metrics.
    """
    valid_mask = (ndvi != nodata_value) & np.isfinite(ndvi)
    valid_pixels = int(np.count_nonzero(valid_mask))
    total_pixels = int(ndvi.size)

    if valid_pixels == 0:
        return {
            "valid_pixels": 0,
            "total_pixels": total_pixels,
            "mean": None,
            "min": None,
            "max": None,
            "std": None,
            "dense_vegetation_pct": 0.0,
            "sparse_vegetation_pct": 0.0,
            "barren_soil_pct": 0.0,
            "water_proxy_pct": 0.0,
        }

    valid_vals = ndvi[valid_mask]

    dense_veg = int(np.count_nonzero(valid_vals >= 0.4))
    sparse_veg = int(np.count_nonzero((valid_vals >= 0.2) & (valid_vals < 0.4)))
    barren = int(np.count_nonzero((valid_vals >= 0.0) & (valid_vals < 0.2)))
    water_proxy = int(np.count_nonzero(valid_vals < 0.0))

    return {
        "valid_pixels": valid_pixels,
        "total_pixels": total_pixels,
        "nodata_pixels": total_pixels - valid_pixels,
        "mean": round(float(np.mean(valid_vals)), 4),
        "min": round(float(np.min(valid_vals)), 4),
        "max": round(float(np.max(valid_vals)), 4),
        "std": round(float(np.std(valid_vals)), 4),
        "dense_vegetation_pct": round((dense_veg / valid_pixels) * 100.0, 2),
        "sparse_vegetation_pct": round((sparse_veg / valid_pixels) * 100.0, 2),
        "barren_soil_pct": round((barren / valid_pixels) * 100.0, 2),
        "water_proxy_pct": round((water_proxy / valid_pixels) * 100.0, 2),
    }

def save_geotiff(
    output_path: Union[str, Path],
    data: np.ndarray,
    transform: Optional[rasterio.Affine] = None,
    crs: Optional[Union[str, rasterio.crs.CRS]] = "EPSG:4326",
    nodata: float = DEFAULT_NODATA
) -> Path:
    """
    Saves a 2D float32 array as a georeferenced single-band GeoTIFF.
    """
    out_p = Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    h, w = data.shape[:2]
    if transform is None:
        # Default mock 10m pixel transform
        transform = from_origin(85.0, 20.0, 0.0001, 0.0001)

    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "compress": "lzw",
    }

    with rasterio.open(out_p, "w", **profile) as dst:
        dst.write(data.astype(np.float32), 1)

    return out_p
