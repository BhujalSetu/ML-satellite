"""
WATERSCOPE-AI — Satellite Remote Sensing Pipeline
Supports two execution modes:
1. Synthetic mode: Generates realistic multi-spectral mock data and demonstrates temporal dynamics.
2. Real mode: Ingests and processes real Sentinel-2 Level-2A 10m JP2 bands (B03, B04, B08),
   verifies spatial alignment, computes real NDVI, NDWI, and water masks, and exports GeoTIFFs
   preserving the true spatial CRS (EPSG:32645) and transform.
"""

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import rasterio
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.satellite.ndvi import calculate_ndvi, compute_ndvi_statistics, save_geotiff as save_ndvi_tiff
from src.satellite.ndwi import calculate_ndwi, classify_water_mask, compute_ndwi_statistics, save_water_mask_geotiff
from src.satellite.temporal import detect_water_dynamics, save_dynamics_geotiff
from src.satellite.aoi_workflow import (
    AOIBoundingBox,
    build_aoi,
    search_sentinel_scenes_for_aoi,
    audit_scene_download_access,
)
from src.satellite.temporal_change import (
    validate_temporal_pair,
    calculate_ndvi_difference,
    calculate_ndwi_difference,
    calculate_relative_change,
    classify_ndvi_change,
    classify_ndwi_change,
    calculate_temporal_water_dynamics,
    calculate_change_statistics,
    write_temporal_change_geotiff,
    write_temporal_change_summary,
    NDVI_CHANGE_CLASSES,
    NDWI_CHANGE_CLASSES,
    WATER_DYNAMICS_CATEGORIES,
)

# ─────────────────────────────────────────────────────────────────────────────
# Real Sentinel-2 Scene Constants & Metadata
# ─────────────────────────────────────────────────────────────────────────────
REAL_SCENE_METADATA = {
    "scene_id": "S2A_MSIL2A_20240221T044821_N0510_R076_T45QUC_20240221T075652",
    "satellite": "Sentinel-2A",
    "product_type": "L2A",
    "acquisition_datetime": "2024-02-21T04:48:21.024Z",
    "source": "Copernicus Data Space Ecosystem — manually downloaded Sentinel-2 L2A product",
}

DEFAULT_S2_DIR = ROOT / "data" / "satellite" / "sentinel2"
DEFAULT_B03_FILE = "T45QUC_20240221T044821_B03_10m.jp2"
DEFAULT_B04_FILE = "T45QUC_20240221T044821_B04_10m.jp2"
DEFAULT_B08_FILE = "T45QUC_20240221T044821_B08_10m.jp2"

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic Mode Helper Functions (Preserved)
# ─────────────────────────────────────────────────────────────────────────────
def generate_synthetic_watershed_raster(h: int = 512, w: int = 512):
    """
    Generates realistic multispectral surface reflectance arrays (scaled 0-10000 like Sentinel-2 L2A)
    simulating a watershed landscape with central water reservoir and agricultural fields.
    """
    y, x = np.mgrid[:h, :w]
    center_y, center_x = h // 2, w // 2

    dist_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    pond_t0 = dist_from_center < 60
    new_pond = np.sqrt((x - (center_x + 130)) ** 2 + (y - (center_y - 80)) ** 2) < 35
    pond_t1 = (dist_from_center < 100) | new_pond

    veg_zone = y < (h // 2 - 40)

    def make_bands(is_t1=False):
        water = pond_t1 if is_t1 else pond_t0

        green = np.full((h, w), 1800, dtype=np.float32)
        red = np.full((h, w), 2200, dtype=np.float32)
        nir = np.full((h, w), 2500, dtype=np.float32)

        green[veg_zone] = 1200
        red[veg_zone] = 500
        nir[veg_zone] = 4800 if is_t1 else 3800

        green[water] = 450
        red[water] = 200
        nir[water] = 90

        rng = np.random.default_rng(seed=42 if not is_t1 else 99)
        noise = rng.normal(0, 30, (h, w)).astype(np.float32)
        green += noise
        red += noise
        nir += noise

        green[:5, :] = -9999
        green[:, :5] = -9999
        red[:5, :] = -9999
        red[:, :5] = -9999
        nir[:5, :] = -9999
        nir[:, :5] = -9999

        return green, red, nir

    g0, r0, n0 = make_bands(is_t1=False)
    g1, r1, n1 = make_bands(is_t1=True)

    return (g0, r0, n0), (g1, r1, n1)

def run_synthetic_pipeline(output_dir: Path):
    """Executes the existing synthetic pipeline demo."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 68)
    print("  WATERSCOPE-AI — Synthetic Satellite Pipeline Demo")
    print("=" * 68)
    print("\n[Architecture Note]")
    print("  Generating synthetic temporal scenes (T0 vs T1) for methodology demonstration.")

    transform = from_origin(85.8000, 20.2500, 0.0001, 0.0001)
    crs = "EPSG:4326"

    print("\n[Generating / Processing Temporal Multispectral Bands]")
    (g0, r0, n0), (g1, r1, n1) = generate_synthetic_watershed_raster(512, 512)

    print("\n[1. Calculating NDVI (Vegetation Dynamics)]")
    ndvi_t0 = calculate_ndvi(nir=n0, red=r0)
    ndvi_t1 = calculate_ndvi(nir=n1, red=r1)
    ndvi_t0_stats = compute_ndvi_statistics(ndvi_t0)
    ndvi_t1_stats = compute_ndvi_statistics(ndvi_t1)
    print(f"  T0 NDVI: Mean={ndvi_t0_stats['mean']}, Dense Veg={ndvi_t0_stats['dense_vegetation_pct']}%, Water Proxy={ndvi_t0_stats['water_proxy_pct']}%")
    print(f"  T1 NDVI: Mean={ndvi_t1_stats['mean']}, Dense Veg={ndvi_t1_stats['dense_vegetation_pct']}%, Water Proxy={ndvi_t1_stats['water_proxy_pct']}%")

    ndvi_path_t0 = save_ndvi_tiff(output_dir / "ndvi_t0.tif", ndvi_t0, transform=transform, crs=crs)
    ndvi_path_t1 = save_ndvi_tiff(output_dir / "ndvi_t1.tif", ndvi_t1, transform=transform, crs=crs)

    print("\n[2. Calculating NDWI (Surface Water Delineation)]")
    ndwi_t0 = calculate_ndwi(band_primary=g0, band_secondary=n0)
    ndwi_t1 = calculate_ndwi(band_primary=g1, band_secondary=n1)
    ndwi_t0_stats = compute_ndwi_statistics(ndwi_t0)
    ndwi_t1_stats = compute_ndwi_statistics(ndwi_t1)
    print(f"  T0 NDWI: Water Pixels={ndwi_t0_stats['water_pixels']} ({ndwi_t0_stats['water_percentage']}%), Est Area={ndwi_t0_stats['estimated_water_area_hectares']} ha")
    print(f"  T1 NDWI: Water Pixels={ndwi_t1_stats['water_pixels']} ({ndwi_t1_stats['water_percentage']}%), Est Area={ndwi_t1_stats['estimated_water_area_hectares']} ha")

    mask_t0 = classify_water_mask(ndwi_t0)
    mask_t1 = classify_water_mask(ndwi_t1)
    mask_path_t0 = save_water_mask_geotiff(output_dir / "water_mask_t0.tif", mask_t0, transform=transform, crs=crs)
    mask_path_t1 = save_water_mask_geotiff(output_dir / "water_mask_t1.tif", mask_t1, transform=transform, crs=crs)

    print("\n[3. Classifying Temporal Water Dynamics (T0 -> T1)]")
    dynamics = detect_water_dynamics(ndwi_t0=ndwi_t0, ndwi_t1=ndwi_t1)
    dyn_cats = dynamics["categories"]
    print(f"  Persistent Water : {dyn_cats['persistent_water']['pixel_count']} px ({dyn_cats['persistent_water']['percentage']}%) [{dyn_cats['persistent_water']['area_hectares']} ha]")
    print(f"  Water Gain       : {dyn_cats['water_gain']['pixel_count']} px ({dyn_cats['water_gain']['percentage']}%) [{dyn_cats['water_gain']['area_hectares']} ha]")
    print(f"  Water Loss       : {dyn_cats['water_loss']['pixel_count']} px ({dyn_cats['water_loss']['percentage']}%) [{dyn_cats['water_loss']['area_hectares']} ha]")
    print(f"  Persistent Land  : {dyn_cats['persistent_land']['pixel_count']} px ({dyn_cats['persistent_land']['percentage']}%) [{dyn_cats['persistent_land']['area_hectares']} ha]")

    dyn_path = save_dynamics_geotiff(output_dir / "water_dynamics.tif", dynamics["dynamics_raster"], transform=transform, crs=crs)

    summary = {
        "mode": "synthetic",
        "spatial_reference": {"crs": crs, "transform": [float(v) for v in list(transform)[:6]]},
        "ndvi_t0": ndvi_t0_stats,
        "ndvi_t1": ndvi_t1_stats,
        "ndwi_t0": ndwi_t0_stats,
        "ndwi_t1": ndwi_t1_stats,
        "temporal_water_dynamics": {
            k: {pk: pv for pk, pv in v.items() if pk != "dynamics_raster"}
            for k, v in dyn_cats.items()
        },
        "files_generated": [
            str(ndvi_path_t0), str(ndvi_path_t1),
            str(mask_path_t0), str(mask_path_t1),
            str(dyn_path)
        ]
    }
    json_path = output_dir / "satellite_analytics_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Analytics Summary]")
    print(f"  Saved JSON Report : {json_path}")
    print("=" * 68)

def run_synthetic_temporal_pipeline(output_dir: Path):
    """
    Executes a complete synthetic temporal change detection pipeline producing all 9 GeoTIFF
    artifacts and temporal_change_summary.json, clearly labeled as TEST/SYNTHETIC.
    """
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 72)
    print("  WATERSCOPE-AI — Synthetic Temporal Change Pipeline (TEST/SYNTHETIC)")
    print("=" * 72)
    print("[Notice: Operating on synthetic test arrays for temporal verification]")

    h, w = 512, 512
    pixel_size_deg = 0.0001
    transform = from_origin(85.8000, 20.2500, pixel_size_deg, pixel_size_deg)
    crs = "EPSG:4326"
    # Derived pixel area: at ~20N, 0.0001 deg is ~11m x 10.4m ~ 114 m2 (or standard 100 m2 test cell)
    pixel_area_m2 = 100.0

    (g0, r0, n0), (g1, r1, n1) = generate_synthetic_watershed_raster(h, w)

    # 1. Compute NDVI T0 and T1
    ndvi_t0 = calculate_ndvi(nir=n0, red=r0)
    ndvi_t1 = calculate_ndvi(nir=n1, red=r1)
    ndvi_t0_stats = compute_ndvi_statistics(ndvi_t0)
    ndvi_t1_stats = compute_ndvi_statistics(ndvi_t1)

    # 2. Compute NDVI Change and Classification
    ndvi_diff = calculate_ndvi_difference(ndvi_t0, ndvi_t1)
    ndvi_diff_stats = calculate_change_statistics(ndvi_diff, threshold_negligible=0.05)
    ndvi_change_class = classify_ndvi_change(ndvi_diff, threshold_decline=-0.10, threshold_gain=0.10)

    # 3. Compute NDWI T0 and T1
    ndwi_t0 = calculate_ndwi(band_primary=g0, band_secondary=n0)
    ndwi_t1 = calculate_ndwi(band_primary=g1, band_secondary=n1)
    ndwi_t0_stats = compute_ndwi_statistics(ndwi_t0, pixel_area_m2=pixel_area_m2)
    ndwi_t1_stats = compute_ndwi_statistics(ndwi_t1, pixel_area_m2=pixel_area_m2)

    # 4. Compute NDWI Change and Classification
    ndwi_diff = calculate_ndwi_difference(ndwi_t0, ndwi_t1)
    ndwi_diff_stats = calculate_change_statistics(ndwi_diff, threshold_negligible=0.05)
    ndwi_change_class = classify_ndwi_change(ndwi_diff, threshold_dry=-0.10, threshold_wet=0.10)

    # 5. Temporal Water Dynamics
    water_dynamics = calculate_temporal_water_dynamics(
        ndwi_t0=ndwi_t0,
        ndwi_t1=ndwi_t1,
        water_threshold=0.0,
        pixel_area_m2=pixel_area_m2,
    )
    dyn_cats = water_dynamics["categories"]

    # 6. Water Masks
    mask_t0 = classify_water_mask(ndwi_t0)
    mask_t1 = classify_water_mask(ndwi_t1)

    # 7. Write 9 GeoTIFF Output Artifacts
    p_ndvi_t0 = write_temporal_change_geotiff(out_dir / "ndvi_t0.tif", ndvi_t0, transform, crs, nodata_value=-9999.0, dtype="float32")
    p_ndvi_t1 = write_temporal_change_geotiff(out_dir / "ndvi_t1.tif", ndvi_t1, transform, crs, nodata_value=-9999.0, dtype="float32")
    p_ndvi_change = write_temporal_change_geotiff(out_dir / "ndvi_change.tif", ndvi_diff, transform, crs, nodata_value=-9999.0, dtype="float32")

    p_ndwi_t0 = write_temporal_change_geotiff(out_dir / "ndwi_t0.tif", ndwi_t0, transform, crs, nodata_value=-9999.0, dtype="float32")
    p_ndwi_t1 = write_temporal_change_geotiff(out_dir / "ndwi_t1.tif", ndwi_t1, transform, crs, nodata_value=-9999.0, dtype="float32")
    p_ndwi_change = write_temporal_change_geotiff(out_dir / "ndwi_change.tif", ndwi_diff, transform, crs, nodata_value=-9999.0, dtype="float32")

    p_mask_t0 = write_temporal_change_geotiff(out_dir / "water_mask_t0.tif", mask_t0, transform, crs, nodata_value=255, dtype="uint8")
    p_mask_t1 = write_temporal_change_geotiff(out_dir / "water_mask_t1.tif", mask_t1, transform, crs, nodata_value=255, dtype="uint8")
    p_water_change = write_temporal_change_geotiff(out_dir / "water_change.tif", water_dynamics["dynamics_raster"], transform, crs, nodata_value=255, dtype="uint8")

    # 8. Summary JSON
    summary = {
        "dataset_type": "TEST/SYNTHETIC",
        "spatial_reference": {"crs": crs, "transform": [float(v) for v in list(transform)[:6]]},
        "pixel_resolution_m": [10.0, 10.0],
        "pixel_area_m2": pixel_area_m2,
        "observations": {
            "T0": {"scene_id": "SYNTHETIC_WATERSHED_T0", "datetime": "2024-01-15T05:00:00Z", "cloud_cover_pct": 0.0},
            "T1": {"scene_id": "SYNTHETIC_WATERSHED_T1", "datetime": "2024-02-15T05:00:00Z", "cloud_cover_pct": 0.0},
        },
        "ndvi_statistics": {
            "T0": ndvi_t0_stats,
            "T1": ndvi_t1_stats,
            "difference": ndvi_diff_stats,
        },
        "ndwi_statistics": {
            "T0": ndwi_t0_stats,
            "T1": ndwi_t1_stats,
            "difference": ndwi_diff_stats,
        },
        "water_statistics": {
            "persistent_land": dyn_cats["persistent_land"],
            "persistent_water": dyn_cats["persistent_water"],
            "water_gain": dyn_cats["water_gain"],
            "water_loss": dyn_cats["water_loss"],
        },
        "processing_status": "Synthetic temporal change detection completed successfully.",
        "files_generated": [
            str(p_ndvi_t0), str(p_ndvi_t1), str(p_ndvi_change),
            str(p_ndwi_t0), str(p_ndwi_t1), str(p_ndwi_change),
            str(p_mask_t0), str(p_mask_t1), str(p_water_change),
        ]
    }
    json_path = out_dir / "temporal_change_summary.json"
    write_temporal_change_summary(json_path, summary)

    print(f"\n[Generated Temporal Output Artifacts]")
    print(f"  NDVI T0 / T1 / Change        : {p_ndvi_t0.name}, {p_ndvi_t1.name}, {p_ndvi_change.name}")
    print(f"  NDWI T0 / T1 / Change        : {p_ndwi_t0.name}, {p_ndwi_t1.name}, {p_ndwi_change.name}")
    print(f"  Water Mask T0 / T1 / Change  : {p_mask_t0.name}, {p_mask_t1.name}, {p_water_change.name}")
    print(f"  Summary Report JSON          : {json_path.name}")
    print("=" * 72)

# ─────────────────────────────────────────────────────────────────────────────
# Real Sentinel-2 Mode Verification & Processing
# ─────────────────────────────────────────────────────────────────────────────
def resolve_band_path(user_arg: Optional[str], default_filename: str) -> Path:
    """Resolves band path from command-line argument or auto-discovers default location."""
    if user_arg:
        p = Path(user_arg).resolve()
        if p.is_file():
            return p
        raise FileNotFoundError(f"Specified band file does not exist: {user_arg}")

    # 1. Check direct default path
    direct = DEFAULT_S2_DIR / default_filename
    if direct.is_file():
        return direct.resolve()

    # 2. Check search recursively in data/satellite/sentinel2
    if DEFAULT_S2_DIR.is_dir():
        matches = list(DEFAULT_S2_DIR.rglob(default_filename))
        if matches:
            return matches[0].resolve()

    raise FileNotFoundError(
        f"Default Sentinel-2 band file '{default_filename}' not found in {DEFAULT_S2_DIR}."
    )

def verify_band_alignment(band_paths: Dict[str, Path]) -> Dict[str, Any]:
    """
    Opens each band raster with Rasterio and strictly verifies that all bands share:
    - identical width
    - identical height
    - identical CRS
    - identical affine transform
    - identical resolution
    - compatible bounds

    Fails clearly with ValueError if any misalignment is detected.
    Does NOT silently resample or reproject.
    """
    ref_meta = None
    ref_band_name = None

    for band_name, path in band_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Band file '{band_name}' not found: {path}")

        with rasterio.open(path) as ds:
            current_meta = {
                "width": ds.width,
                "height": ds.height,
                "crs": ds.crs,
                "transform": ds.transform,
                "res": ds.res,
                "bounds": ds.bounds,
                "driver": ds.driver,
                "nodata": ds.nodata,
            }

            if ref_meta is None:
                ref_meta = current_meta
                ref_band_name = band_name
            else:
                if current_meta["width"] != ref_meta["width"]:
                    raise ValueError(
                        f"Width mismatch: {ref_band_name} has width {ref_meta['width']}, "
                        f"but {band_name} has {current_meta['width']}."
                    )
                if current_meta["height"] != ref_meta["height"]:
                    raise ValueError(
                        f"Height mismatch: {ref_band_name} has height {ref_meta['height']}, "
                        f"but {band_name} has {current_meta['height']}."
                    )
                if current_meta["crs"] != ref_meta["crs"]:
                    raise ValueError(
                        f"CRS mismatch: {ref_band_name} has CRS {ref_meta['crs']}, "
                        f"but {band_name} has {current_meta['crs']}."
                    )
                if current_meta["transform"] != ref_meta["transform"]:
                    raise ValueError(
                        f"Affine transform mismatch between {ref_band_name} and {band_name}."
                    )
                if current_meta["res"] != ref_meta["res"]:
                    raise ValueError(
                        f"Spatial resolution mismatch: {ref_band_name} has resolution {ref_meta['res']}, "
                        f"but {band_name} has {current_meta['res']}."
                    )
                # Bounds check tolerance: 0.001 m
                b_ref = ref_meta["bounds"]
                b_cur = current_meta["bounds"]
                if (abs(b_ref.left - b_cur.left) > 1e-3 or
                    abs(b_ref.bottom - b_cur.bottom) > 1e-3 or
                    abs(b_ref.right - b_cur.right) > 1e-3 or
                    abs(b_ref.top - b_cur.top) > 1e-3):
                    raise ValueError(
                        f"Spatial bounds mismatch between {ref_band_name} and {band_name}."
                    )

    return ref_meta

def run_real_pipeline(
    b03_arg: Optional[str] = None,
    b04_arg: Optional[str] = None,
    b08_arg: Optional[str] = None,
    output_dir_arg: Optional[str] = None,
    lat_arg: Optional[float] = None,
    lon_arg: Optional[float] = None,
    buffer_arg: float = 0.05,
    west_arg: Optional[float] = None,
    south_arg: Optional[float] = None,
    east_arg: Optional[float] = None,
    north_arg: Optional[float] = None,
    start_date: str = "2024-01-01",
    end_date: str = "2024-03-31",
    max_cloud: float = 20.0,
    limit: int = 5,
    temporal_mode: bool = False,
    b03_t1_arg: Optional[str] = None,
    b04_t1_arg: Optional[str] = None,
    b08_t1_arg: Optional[str] = None,
):
    """
    Executes real Sentinel-2 Level-2A workflow:
    1. If AOI parameters (lat/lon or bounding box) are supplied:
       - Validates geographic coordinates and builds AOI.
       - Queries the official Copernicus STAC API for Sentinel-2 Level-2A scenes.
       - Discovers low-cloud candidates, identifies B03/B04/B08 assets.
       - Audits download accessibility (reports authentication requirement transparently).
       - Saves structured STAC discovery report.
    2. Resolves band paths (from CLI arguments or local verified test scene).
    3. Strictly verifies spatial alignment across bands.
    4. Computes real NDVI and NDWI, delineates surface water, and exports georeferenced GeoTIFFs.
    """
    out_dir = Path(output_dir_arg or (ROOT / "outputs" / "satellite" / "real")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 72)
    print("  WATERSCOPE-AI — Real Sentinel-2 Level-2A Processing Pipeline")
    print("=" * 72)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 0: AOI & STAC Scene Discovery (if AOI provided)
    # ─────────────────────────────────────────────────────────────────────────
    has_aoi = (lat_arg is not None and lon_arg is not None) or all(
        c is not None for c in [west_arg, south_arg, east_arg, north_arg]
    )

    aoi_report = None
    if has_aoi:
        print("\n[AOI Specification & STAC Discovery]")
        aoi = build_aoi(
            lat=lat_arg,
            lon=lon_arg,
            buffer_deg=buffer_arg,
            west=west_arg,
            south=south_arg,
            east=east_arg,
            north=north_arg,
        )
        print(f"  Target AOI Footprint (WGS 84):")
        print(f"    West: {aoi.west:.5f}, South: {aoi.south:.5f}, East: {aoi.east:.5f}, North: {aoi.north:.5f}")

        print(f"\n  Querying Copernicus Data Space Ecosystem STAC...")
        print(f"    Date Range      : {start_date} to {end_date}")
        print(f"    Max Cloud Cover : <= {max_cloud}%")
        print(f"    Candidate Limit : {limit}")

        search_res = search_sentinel_scenes_for_aoi(
            aoi=aoi,
            date_range=(start_date, end_date),
            max_cloud_cover=max_cloud,
            limit=limit,
        )
        print(f"  Discovered Scenes : {search_res['total_scenes_found']}")

        if search_res["total_scenes_found"] > 0:
            print("\n  Top Matching Candidate Scenes:")
            for idx, sc in enumerate(search_res["scenes"][:3]):
                print(f"    {idx+1}. {sc['scene_id']} | Date: {sc['datetime'][:10]} | Cloud: {sc['cloud_cover_pct']}%")

            best = search_res["best_scene"]
            print(f"\n  Selected Optimal Scene : {best['scene_id']}")
            print(f"    Acquisition Datetime : {best['datetime']}")
            print(f"    Cloud Cover          : {best['cloud_cover_pct']}%")
            print(f"    Has Key Bands (3,4,8): {best['has_required_bands']}")

            # Check download authentication requirement
            b04_data = best["bands"]["B04_red"]
            https_url = b04_data.get("https_href") if b04_data else None

            print("\n  [Asset Download & Authentication Audit]")
            print(f"    Red (B04) S3 URI   : {b04_data.get('s3_href') if b04_data else 'N/A'}")
            print(f"    Red (B04) HTTPS    : {https_url or 'N/A'}")

            auth_limitation = (
                "Copernicus Data Space Ecosystem (CDSE) policy requires registered user authentication "
                "(OAuth2 / OIDC Bearer Token) or S3 EODATA credentials to download raw full-resolution .jp2 bands. "
                "The STAC catalog search and metadata discovery are 100% public and dynamic. "
                "Local verified Sentinel-2 Level-2A files are utilized for downstream spectral raster processing."
            )
            print(f"    CDSE Policy Notice : {auth_limitation}")

            aoi_report = {
                "aoi": aoi.to_dict(),
                "search_parameters": {
                    "start_date": start_date,
                    "end_date": end_date,
                    "max_cloud_cover": max_cloud,
                    "limit": limit,
                },
                "discovery_result": search_res,
                "download_authentication_required": True,
                "authentication_notice": auth_limitation,
            }

            aoi_report_path = out_dir / "aoi_stac_discovery_report.json"
            with open(aoi_report_path, "w", encoding="utf-8") as f:
                json.dump(aoi_report, f, indent=2)
            print(f"    Saved AOI Discovery Report: {aoi_report_path.name}")
        else:
            print("  Notice: No scenes found for the requested AOI and date range.")

    # 1. Resolve Band Paths (with local verified fallback)
    b03_path = resolve_band_path(b03_arg, DEFAULT_B03_FILE)
    b04_path = resolve_band_path(b04_arg, DEFAULT_B04_FILE)
    b08_path = resolve_band_path(b08_arg, DEFAULT_B08_FILE)

    band_paths = {
        "B03 (Green 10m)": b03_path,
        "B04 (Red 10m)": b04_path,
        "B08 (NIR 10m)": b08_path,
    }

    print("\n[Input Sentinel-2 Bands]")
    for name, p in band_paths.items():
        print(f"  {name:<18} : {p.name} ({p.stat().st_size / (1024*1024):.1f} MB)")
        print(f"                     Path: {p}")

    # 2. Strict Spatial Verification
    print("\n[Verifying Spatial Alignment]")
    ref_meta = verify_band_alignment(band_paths)
    crs = ref_meta["crs"]
    transform = ref_meta["transform"]
    w = ref_meta["width"]
    h = ref_meta["height"]
    res = ref_meta["res"]
    bounds = ref_meta["bounds"]
    pixel_area_m2 = float(res[0] * res[1])

    print(f"  Alignment Check : PASSED (All 3 bands strictly identical in geometry)")
    print(f"  CRS             : {crs} (UTM Zone 45N / WGS 84)")
    print(f"  Dimensions      : {w} x {h} pixels ({w * h:,} total cells)")
    print(f"  Spatial Res     : {res[0]}m x {res[1]}m ({pixel_area_m2:.1f} m2 per pixel)")
    print(f"  Bounding Extent : Left={bounds.left:.1f}, Bottom={bounds.bottom:.1f}, Right={bounds.right:.1f}, Top={bounds.top:.1f}")

    # 3. Read Red (B04) & NIR (B08) and Compute NDVI
    # Formula: NDVI = (NIR - Red) / (NIR + Red)
    print("\n[1. Processing Real NDVI — Normalized Difference Vegetation Index]")
    print("  Formula: NDVI = (B08 - B04) / (B08 + B04)")
    print("  Reading B04 (Red) & B08 (NIR) as float32 arrays...")

    with rasterio.open(b04_path) as src_red:
        b04 = src_red.read(1)
    with rasterio.open(b08_path) as src_nir:
        b08 = src_nir.read(1)

    print("  Calculating NDVI and masking nodata / zero-division cells...")
    ndvi = calculate_ndvi(nir=b08, red=b04)
    del b04  # Immediately free memory for Red band
    gc.collect()

    print("  Computing canopy coverage and vegetation statistics...")
    ndvi_stats = compute_ndvi_statistics(ndvi)
    print(f"    Valid Cells   : {ndvi_stats['valid_pixels']:,} / {ndvi_stats['total_pixels']:,} ({ndvi_stats['nodata_pixels']:,} nodata)")
    print(f"    NDVI Range    : [{ndvi_stats['min']:.4f}, {ndvi_stats['max']:.4f}] | Mean: {ndvi_stats['mean']:.4f} (Std: {ndvi_stats['std']:.4f})")
    print(f"    Dense Veg     : {ndvi_stats['dense_vegetation_pct']}% (NDVI >= 0.4)")
    print(f"    Sparse Veg    : {ndvi_stats['sparse_vegetation_pct']}% (0.2 <= NDVI < 0.4)")
    print(f"    Barren Soil   : {ndvi_stats['barren_soil_pct']}% (0.0 <= NDVI < 0.2)")
    print(f"    Water Proxy   : {ndvi_stats['water_proxy_pct']}% (NDVI < 0.0)")

    ndvi_out_path = out_dir / "ndvi_real.tif"
    print(f"  Saving georeferenced GeoTIFF: {ndvi_out_path.name}...")
    save_ndvi_tiff(ndvi_out_path, ndvi, transform=transform, crs=crs)
    print(f"  GeoTIFF saved ({ndvi_out_path.stat().st_size / (1024*1024):.1f} MB)")

    del ndvi  # Free NDVI array before computing NDWI
    gc.collect()

    # 4. Read Green (B03) & Compute NDWI
    # Formula: McFeeters (1996) NDWI = (Green - NIR) / (Green + NIR)
    print("\n[2. Processing Real NDWI — Normalized Difference Water Index]")
    print("  Formulation: McFeeters (1996) NDWI = (B03 - B08) / (B03 + B08) [Water threshold: NDWI > 0.0]")
    print("  Reading B03 (Green) as float32 array...")

    with rasterio.open(b03_path) as src_green:
        b03 = src_green.read(1)

    print("  Calculating NDWI and surface water delineation...")
    ndwi = calculate_ndwi(band_primary=b03, band_secondary=b08, method="mcfeeters")
    del b03, b08  # Free remaining raw reflectance bands
    gc.collect()

    print("  Computing surface water statistics...")
    ndwi_stats = compute_ndwi_statistics(ndwi, threshold=0.0, pixel_area_m2=pixel_area_m2)
    print(f"    Valid Cells   : {ndwi_stats['valid_pixels']:,} / {ndwi_stats['total_pixels']:,}")
    print(f"    NDWI Range    : [{ndwi_stats['min']:.4f}, {ndwi_stats['max']:.4f}] | Mean: {ndwi_stats['mean']:.4f}")
    print(f"    Water Pixels  : {ndwi_stats['water_pixels']:,} ({ndwi_stats['water_percentage']}%)")
    print(f"    Water Surface : {ndwi_stats['estimated_water_area_m2']:,} m2 ({ndwi_stats['estimated_water_area_hectares']:,} ha)")

    ndwi_out_path = out_dir / "ndwi_real.tif"
    print(f"  Saving georeferenced GeoTIFF: {ndwi_out_path.name}...")
    save_ndvi_tiff(ndwi_out_path, ndwi, transform=transform, crs=crs)
    print(f"  GeoTIFF saved ({ndwi_out_path.stat().st_size / (1024*1024):.1f} MB)")

    # 5. Generate and Save Binary Water Mask
    print("\n[3. Generating Real Surface Water Mask]")
    water_mask = classify_water_mask(ndwi, threshold=0.0)
    del ndwi  # Free NDWI array
    gc.collect()

    water_px = int(np.count_nonzero(water_mask == 1))
    non_water_px = int(np.count_nonzero(water_mask == 0))
    nodata_px = int(np.count_nonzero(water_mask == 255))
    total_valid = water_px + non_water_px
    water_pct = (water_px / total_valid * 100.0) if total_valid > 0 else 0.0

    water_mask_stats = {
        "threshold": 0.0,
        "total_pixels": int(water_mask.size),
        "water_pixels": water_px,
        "non_water_pixels": non_water_px,
        "nodata_pixels": nodata_px,
        "water_percentage": round(water_pct, 2),
        "estimated_water_area_m2": round(water_px * pixel_area_m2, 2),
        "estimated_water_area_hectares": round((water_px * pixel_area_m2) / 10000.0, 4),
    }

    water_mask_out_path = out_dir / "water_mask_real.tif"
    print(f"  Saving georeferenced Water Mask GeoTIFF: {water_mask_out_path.name}...")
    save_water_mask_geotiff(water_mask_out_path, water_mask, transform=transform, crs=crs)
    print(f"  Water Mask saved ({water_mask_out_path.stat().st_size / (1024*1024):.1f} MB)")
    del water_mask
    gc.collect()

    # 6. Temporal Analysis Handling
    print("\n[4. Temporal Analysis Status]")
    has_t1_bands = bool(b03_t1_arg and b04_t1_arg and b08_t1_arg)
    temporal_summary = None

    if temporal_mode and has_t1_bands:
        print("  Two real observations provided. Initiating real temporal comparison (T0 vs T1)...")
        temporal_out_dir = (Path(output_dir_arg).resolve() if output_dir_arg else (ROOT / "outputs" / "satellite" / "temporal"))
        temporal_out_dir.mkdir(parents=True, exist_ok=True)

        b03_t1_path = Path(b03_t1_arg).resolve()
        b04_t1_path = Path(b04_t1_arg).resolve()
        b08_t1_path = Path(b08_t1_arg).resolve()

        t1_paths = {
            "B03_T1": b03_t1_path,
            "B04_T1": b04_t1_path,
            "B08_T1": b08_t1_path,
        }
        meta_t1 = verify_band_alignment(t1_paths)
        validate_temporal_pair(ref_meta, meta_t1)

        print("  Processing T1 bands and biophysical differences...")
        with rasterio.open(b04_t1_path) as s_r1, rasterio.open(b08_t1_path) as s_n1:
            r1 = s_r1.read(1)
            n1 = s_n1.read(1)
        ndvi_t1 = calculate_ndvi(nir=n1, red=r1)
        del r1
        gc.collect()

        with rasterio.open(b03_t1_path) as s_g1:
            g1 = s_g1.read(1)
        ndwi_t1 = calculate_ndwi(band_primary=g1, band_secondary=n1)
        del g1, n1
        gc.collect()

        # Re-compute or re-read T0 for temporal diff
        with rasterio.open(b04_path) as s_r0, rasterio.open(b08_path) as s_n0:
            ndvi_t0 = calculate_ndvi(nir=s_n0.read(1), red=s_r0.read(1))
        with rasterio.open(b03_path) as s_g0, rasterio.open(b08_path) as s_n0:
            ndwi_t0 = calculate_ndwi(band_primary=s_g0.read(1), band_secondary=s_n0.read(1))

        ndvi_diff = calculate_ndvi_difference(ndvi_t0, ndvi_t1)
        ndwi_diff = calculate_ndwi_difference(ndwi_t0, ndwi_t1)
        water_dyn = calculate_temporal_water_dynamics(ndwi_t0, ndwi_t1, pixel_area_m2=pixel_area_m2)
        mask_t0 = classify_water_mask(ndwi_t0)
        mask_t1 = classify_water_mask(ndwi_t1)

        # Write temporal GeoTIFFs
        t_ndvi_t0 = write_temporal_change_geotiff(temporal_out_dir / "ndvi_t0.tif", ndvi_t0, transform, crs, -9999.0, "float32")
        t_ndvi_t1 = write_temporal_change_geotiff(temporal_out_dir / "ndvi_t1.tif", ndvi_t1, transform, crs, -9999.0, "float32")
        t_ndvi_ch = write_temporal_change_geotiff(temporal_out_dir / "ndvi_change.tif", ndvi_diff, transform, crs, -9999.0, "float32")

        t_ndwi_t0 = write_temporal_change_geotiff(temporal_out_dir / "ndwi_t0.tif", ndwi_t0, transform, crs, -9999.0, "float32")
        t_ndwi_t1 = write_temporal_change_geotiff(temporal_out_dir / "ndwi_t1.tif", ndwi_t1, transform, crs, -9999.0, "float32")
        t_ndwi_ch = write_temporal_change_geotiff(temporal_out_dir / "ndwi_change.tif", ndwi_diff, transform, crs, -9999.0, "float32")

        t_mask_t0 = write_temporal_change_geotiff(temporal_out_dir / "water_mask_t0.tif", mask_t0, transform, crs, 255, "uint8")
        t_mask_t1 = write_temporal_change_geotiff(temporal_out_dir / "water_mask_t1.tif", mask_t1, transform, crs, 255, "uint8")
        t_water_ch = write_temporal_change_geotiff(temporal_out_dir / "water_change.tif", water_dyn["dynamics_raster"], transform, crs, 255, "uint8")

        temporal_notice = "Real temporal change detection completed successfully across two verified observations."
        temporal_summary = {
            "status": "completed",
            "t0_scene": REAL_SCENE_METADATA["scene_id"],
            "t1_bands": [str(b03_t1_path), str(b04_t1_path), str(b08_t1_path)],
            "ndvi_change_stats": calculate_change_statistics(ndvi_diff),
            "ndwi_change_stats": calculate_change_statistics(ndwi_diff),
            "water_dynamics": water_dyn["categories"],
            "files_generated": [
                str(t_ndvi_t0), str(t_ndvi_t1), str(t_ndvi_ch),
                str(t_ndwi_t0), str(t_ndwi_t1), str(t_ndwi_ch),
                str(t_mask_t0), str(t_mask_t1), str(t_water_ch),
            ]
        }
        write_temporal_change_summary(temporal_out_dir / "temporal_change_summary.json", temporal_summary)
        print(f"  Temporal change results saved to: {temporal_out_dir}")
    else:
        temporal_notice = "Temporal comparison requires two compatible observations."
        print(f"  Notice: {temporal_notice}")
        print("  Project Truth: Exactly one real Sentinel-2 observation is locally available.")
        print("  Fabrication of synthetic T0/T1 comparisons on real data is strictly prohibited.")
        if temporal_mode:
            temp_status_dir = (Path(output_dir_arg).resolve() if output_dir_arg else (ROOT / "outputs" / "satellite" / "temporal"))
            temp_status_dir.mkdir(parents=True, exist_ok=True)
            temporal_summary = {
                "status": "unavailable",
                "notice": temporal_notice,
                "local_observations_available": 1,
                "required_observations": 2,
                "message": "Only one real Sentinel-2 Level-2A observation is currently stored locally. Provide a second observation or use --mode synthetic --temporal for synthetic methodology verification.",
                "cdse_download_notice": "Downloading additional real full-resolution band rasters from Copernicus Data Space requires user authentication.",
            }
            write_temporal_change_summary(temp_status_dir / "temporal_change_summary.json", temporal_summary)
            print(f"  Temporal status report written: {temp_status_dir / 'temporal_change_summary.json'}")

    # 7. Summary JSON Generation
    summary_json_data = {
        "scene_id": REAL_SCENE_METADATA["scene_id"],
        "satellite": REAL_SCENE_METADATA["satellite"],
        "product_type": REAL_SCENE_METADATA["product_type"],
        "acquisition_datetime": REAL_SCENE_METADATA["acquisition_datetime"],
        "source": REAL_SCENE_METADATA["source"],
        "bands": {
            "B03": str(b03_path),
            "B04": str(b04_path),
            "B08": str(b08_path),
        },
        "crs": str(crs),
        "resolution": [float(res[0]), float(res[1])],
        "width": int(w),
        "height": int(h),
        "bounds": {
            "left": float(bounds.left),
            "bottom": float(bounds.bottom),
            "right": float(bounds.right),
            "top": float(bounds.top),
        },
        "ndvi_statistics": ndvi_stats,
        "ndwi_statistics": ndwi_stats,
        "water_mask_statistics": water_mask_stats,
        "temporal_analysis_status": temporal_notice,
        "temporal_analysis": temporal_summary,
        "aoi_stac_discovery": aoi_report,
        "files_generated": [
            str(ndvi_out_path),
            str(ndwi_out_path),
            str(water_mask_out_path),
        ],
    }

    json_out_path = out_dir / "real_satellite_analytics_summary.json"
    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(summary_json_data, f, indent=2)

    print(f"\n[Generated Output Artifacts]")
    print(f"  1. Real NDVI GeoTIFF        : {ndvi_out_path}")
    print(f"  2. Real NDWI GeoTIFF        : {ndwi_out_path}")
    print(f"  3. Real Water Mask GeoTIFF  : {water_mask_out_path}")
    print(f"  4. Real Analytics Summary   : {json_out_path}")
    if aoi_report:
        print(f"  5. AOI STAC Discovery Report: {out_dir / 'aoi_stac_discovery_report.json'}")
    print("=" * 72)
    print("  Real Sentinel-2 Pipeline Completed Successfully")
    print("=" * 72)

# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="WATERSCOPE-AI: Satellite Remote Sensing Index & Water Processing Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["synthetic", "real"],
        default="synthetic",
        help="Pipeline mode: 'synthetic' (demo multispectral data) or 'real' (actual Sentinel-2 JP2 files)",
    )
    parser.add_argument(
        "--b03",
        type=str,
        default=None,
        help="Path to real Sentinel-2 B03 (Green 10m) JP2 file (real mode only)",
    )
    parser.add_argument(
        "--b04",
        type=str,
        default=None,
        help="Path to real Sentinel-2 B04 (Red 10m) JP2 file (real mode only)",
    )
    parser.add_argument(
        "--b08",
        type=str,
        default=None,
        help="Path to real Sentinel-2 B08 (NIR 10m) JP2 file (real mode only)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Destination directory for output GeoTIFFs and summary JSON",
    )

    # Location / AOI Configuration
    parser.add_argument(
        "--lat",
        type=float,
        default=None,
        help="Latitude of target center point (WGS 84, -90 to 90)",
    )
    parser.add_argument(
        "--lon",
        type=float,
        default=None,
        help="Longitude of target center point (WGS 84, -180 to 180)",
    )
    parser.add_argument(
        "--buffer",
        type=float,
        default=0.05,
        help="Half-width degree buffer around center point (~0.05 deg is ~5.5km buffer)",
    )
    parser.add_argument(
        "--west",
        type=float,
        default=None,
        help="AOI bounding box west longitude (min_lon)",
    )
    parser.add_argument(
        "--south",
        type=float,
        default=None,
        help="AOI bounding box south latitude (min_lat)",
    )
    parser.add_argument(
        "--east",
        type=float,
        default=None,
        help="AOI bounding box east longitude (max_lon)",
    )
    parser.add_argument(
        "--north",
        type=float,
        default=None,
        help="AOI bounding box north latitude (max_lat)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2024-01-01",
        help="Start acquisition date for STAC search (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2024-03-31",
        help="End acquisition date for STAC search (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--max-cloud",
        type=float,
        default=20.0,
        help="Maximum cloud cover percentage filter (0-100)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of candidate STAC scenes to retrieve",
    )
    # Multi-Temporal Change Detection Configuration
    parser.add_argument(
        "--temporal",
        action="store_true",
        default=False,
        help="Enable multi-temporal change detection (T0 vs T1)",
    )
    parser.add_argument(
        "--b03-t1",
        type=str,
        default=None,
        help="Path to T1 observation B03 (Green 10m) JP2 file (real temporal mode)",
    )
    parser.add_argument(
        "--b04-t1",
        type=str,
        default=None,
        help="Path to T1 observation B04 (Red 10m) JP2 file (real temporal mode)",
    )
    parser.add_argument(
        "--b08-t1",
        type=str,
        default=None,
        help="Path to T1 observation B08 (NIR 10m) JP2 file (real temporal mode)",
    )

    args = parser.parse_args()

    if args.mode == "synthetic":
        if args.temporal:
            out_dir = Path(args.output_dir or (ROOT / "outputs" / "satellite" / "temporal" / "synthetic")).resolve()
            run_synthetic_temporal_pipeline(output_dir=out_dir)
        else:
            out_dir = Path(args.output_dir or (ROOT / "outputs" / "satellite")).resolve()
            run_synthetic_pipeline(output_dir=out_dir)
    elif args.mode == "real":
        run_real_pipeline(
            b03_arg=args.b03,
            b04_arg=args.b04,
            b08_arg=args.b08,
            output_dir_arg=args.output_dir,
            lat_arg=args.lat,
            lon_arg=args.lon,
            buffer_arg=args.buffer,
            west_arg=args.west,
            south_arg=args.south,
            east_arg=args.east,
            north_arg=args.north,
            start_date=args.start_date,
            end_date=args.end_date,
            max_cloud=args.max_cloud,
            limit=args.limit,
            temporal_mode=args.temporal,
            b03_t1_arg=args.b03_t1,
            b04_t1_arg=args.b04_t1,
            b08_t1_arg=args.b08_t1,
        )

if __name__ == "__main__":
    main()
