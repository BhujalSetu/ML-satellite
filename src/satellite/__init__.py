"""
WATERSCOPE-AI — Satellite Remote Sensing Package
Provides reusable raster operations for multispectral vegetation and water indices (NDVI, NDWI),
nodata/zero-division handling, water delineation masks, temporal dynamics, GeoTIFF export,
and Copernicus Data Space STAC client for Sentinel-2 L2A scene discovery.
"""

from .ndvi import calculate_ndvi, compute_ndvi_statistics, save_geotiff as save_ndvi_geotiff
from .ndwi import (
    calculate_ndwi,
    classify_water_mask,
    compute_ndwi_statistics,
    save_water_mask_geotiff,
)
from .temporal import (
    compute_temporal_difference,
    detect_water_dynamics,
    save_dynamics_geotiff,
    WATER_DYNAMICS_CLASSES,
)
from .stac_client import (
    CopernicusSTACClient,
    SentinelSceneMetadata,
    BandAsset,
)

__all__ = [
    "calculate_ndvi",
    "compute_ndvi_statistics",
    "save_ndvi_geotiff",
    "calculate_ndwi",
    "classify_water_mask",
    "compute_ndwi_statistics",
    "save_water_mask_geotiff",
    "compute_temporal_difference",
    "detect_water_dynamics",
    "save_dynamics_geotiff",
    "WATER_DYNAMICS_CLASSES",
    "CopernicusSTACClient",
    "SentinelSceneMetadata",
    "BandAsset",
]
