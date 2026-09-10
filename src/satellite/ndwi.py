"""
WATERSCOPE-AI — Normalized Difference Water Index (NDWI) Module
Calculates NDWI for surface water delineation, handles nodata/zero-division,
extracts water binary masks, computes water surface statistics, and saves GeoTIFFs.

Formulas:
McFeeters (1996): NDWI = (Green - NIR) / (Green + NIR)  [Default: Water > 0]
Gao (1996) / MNDWI: (NIR - SWIR) / (NIR + SWIR)
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union
import numpy as np
import rasterio
from rasterio.transform import from_origin

DEFAULT_NODATA = -9999.0

def calculate_ndwi(
    band_primary: np.ndarray,
    band_secondary: np.ndarray,
    nodata_value: float = DEFAULT_NODATA,
    method: str = "mcfeeters"
) -> np.ndarray:
    """
    Calculates the Normalized Difference Water Index (NDWI).

    Args:
        band_primary: Green band (for McFeeters) or NIR band (for Gao).
        band_secondary: NIR band (for McFeeters) or SWIR band (for Gao).
        nodata_value: Value assigned to zero-division or invalid cells.
        method: 'mcfeeters' (Green - NIR) / (Green + NIR) or 'gao' (NIR - SWIR) / (NIR + SWIR).

    Returns:
        np.ndarray: NDWI array (float32) in [-1.0, 1.0] with nodata_value for invalid cells.
    """
    if band_primary.shape != band_secondary.shape:
        raise ValueError(
            f"Shape mismatch: Band 1 {band_primary.shape} vs Band 2 {band_secondary.shape}"
        )

    b1_f = band_primary.astype(np.float32)
    b2_f = band_secondary.astype(np.float32)

    numerator = b1_f - b2_f
    denominator = b1_f + b2_f

    ndwi = np.full(band_primary.shape, nodata_value, dtype=np.float32)

    valid = (denominator != 0) & np.isfinite(numerator) & np.isfinite(denominator)

    with np.errstate(divide="ignore", invalid="ignore"):
        ndwi_calc = numerator / denominator
        ndwi[valid] = np.clip(ndwi_calc[valid], -1.0, 1.0)

    return ndwi

def classify_water_mask(
    ndwi: np.ndarray,
    threshold: float = 0.0,
    nodata_value: float = DEFAULT_NODATA
) -> np.ndarray:
    """
    Generates a binary water mask where NDWI > threshold.

    Returns:
        np.ndarray: uint8 mask:
            1 = Water Body
            0 = Non-Water (Land/Vegetation)
            255 = NoData
    """
    mask = np.zeros(ndwi.shape, dtype=np.uint8)
    valid = (ndwi != nodata_value) & np.isfinite(ndwi)

    mask[valid & (ndwi > threshold)] = 1
    mask[~valid] = 255
    return mask

def compute_ndwi_statistics(
    ndwi: np.ndarray,
    threshold: float = 0.0,
    pixel_area_m2: float = 100.0,  # e.g., 10m x 10m for Sentinel-2
    nodata_value: float = DEFAULT_NODATA
) -> Dict[str, Any]:
    """
    Computes statistical indicators and water surface area from an NDWI raster.
    """
    valid_mask = (ndwi != nodata_value) & np.isfinite(ndwi)
    valid_pixels = int(np.count_nonzero(valid_mask))
    total_pixels = int(ndwi.size)

    if valid_pixels == 0:
        return {
            "valid_pixels": 0,
            "total_pixels": total_pixels,
            "water_pixels": 0,
            "water_percentage": 0.0,
            "estimated_water_area_m2": 0.0,
            "estimated_water_area_hectares": 0.0,
            "mean": None,
            "min": None,
            "max": None,
            "std": None,
        }

    valid_vals = ndwi[valid_mask]
    water_pixels = int(np.count_nonzero(valid_vals > threshold))
    water_pct = (water_pixels / valid_pixels) * 100.0
    water_area_m2 = water_pixels * pixel_area_m2
    water_area_ha = water_area_m2 / 10000.0

    return {
        "valid_pixels": valid_pixels,
        "total_pixels": total_pixels,
        "nodata_pixels": total_pixels - valid_pixels,
        "water_pixels": water_pixels,
        "water_percentage": round(water_pct, 2),
        "estimated_water_area_m2": round(water_area_m2, 2),
        "estimated_water_area_hectares": round(water_area_ha, 4),
        "mean": round(float(np.mean(valid_vals)), 4),
        "min": round(float(np.min(valid_vals)), 4),
        "max": round(float(np.max(valid_vals)), 4),
        "std": round(float(np.std(valid_vals)), 4),
    }

def save_water_mask_geotiff(
    output_path: Union[str, Path],
    mask: np.ndarray,
    transform: Optional[rasterio.Affine] = None,
    crs: Optional[Union[str, rasterio.crs.CRS]] = "EPSG:4326"
) -> Path:
    """
    Saves a water classification mask (uint8) as a GeoTIFF.
    """
    out_p = Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    h, w = mask.shape[:2]
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
        dst.write(mask.astype(np.uint8), 1)

    return out_p
