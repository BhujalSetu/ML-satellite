"""
WATERSCOPE-AI — AOI & Location-Based Satellite Remote Sensing Workflow
Provides reusable components to:
1. Validate geographical coordinates and bounding boxes.
2. Convert point locations (lat, lon) into small, configurable AOI bounding boxes.
3. Query the Copernicus Data Space STAC API for optimal Sentinel-2 Level-2A scenes.
4. Extract required B03 (Green 10m), B04 (Red 10m), and B08 (NIR 10m) assets.
5. Check and report asset accessibility / CDSE authentication status honestly.
"""

from dataclasses import dataclass, asdict
from typing import Tuple, Optional, Dict, List, Any
from pathlib import Path
import json

from src.satellite.stac_client import CopernicusSTACClient, SentinelSceneMetadata, BandAsset

@dataclass
class AOIBoundingBox:
    """Represents a geographic bounding box in WGS 84 (EPSG:4326)."""
    west: float
    south: float
    east: float
    north: float

    def to_list(self) -> List[float]:
        """Returns [west, south, east, north] as required by STAC bbox."""
        return [self.west, self.south, self.east, self.north]

    def to_dict(self) -> Dict[str, float]:
        return {
            "west": self.west,
            "south": self.south,
            "east": self.east,
            "north": self.north,
        }

    def contains_point(self, lat: float, lon: float) -> bool:
        return (self.west <= lon <= self.east) and (self.south <= lat <= self.north)


def validate_coordinates(lat: float, lon: float) -> None:
    """
    Validates latitude and longitude values within standard Earth ranges.
    Raises ValueError if coordinates are out of bounds.
    """
    if not (-90.0 <= lat <= 90.0):
        raise ValueError(f"Latitude must be between -90.0 and +90.0 degrees, got {lat}")
    if not (-180.0 <= lon <= 180.0):
        raise ValueError(f"Longitude must be between -180.0 and +180.0 degrees, got {lon}")


def validate_bounding_box(west: float, south: float, east: float, north: float) -> None:
    """
    Validates bounding box coordinates and relative orientation.
    """
    validate_coordinates(south, west)
    validate_coordinates(north, east)

    if west >= east:
        raise ValueError(f"Bounding box west ({west}) must be strictly less than east ({east})")
    if south >= north:
        raise ValueError(f"Bounding box south ({south}) must be strictly less than north ({north})")


def create_point_aoi(
    lat: float,
    lon: float,
    buffer_deg: float = 0.05,
) -> AOIBoundingBox:
    """
    Creates a small rectangular AOI centered at (lat, lon) with a configurable degree buffer.
    Default buffer of 0.05 deg corresponds to ~5.5 km buffer in each direction (~11 km x 11 km AOI).
    """
    validate_coordinates(lat, lon)
    if buffer_deg <= 0.0 or buffer_deg > 5.0:
        raise ValueError(f"Buffer must be positive and <= 5.0 degrees, got {buffer_deg}")

    west = max(-180.0, lon - buffer_deg)
    east = min(180.0, lon + buffer_deg)
    south = max(-90.0, lat - buffer_deg)
    north = min(90.0, lat + buffer_deg)

    validate_bounding_box(west, south, east, north)
    return AOIBoundingBox(
        west=round(west, 5),
        south=round(south, 5),
        east=round(east, 5),
        north=round(north, 5),
    )


def build_aoi(
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    buffer_deg: float = 0.05,
    west: Optional[float] = None,
    south: Optional[float] = None,
    east: Optional[float] = None,
    north: Optional[float] = None,
) -> AOIBoundingBox:
    """
    Constructs an AOIBoundingBox either from explicit bbox coordinates or from a center point (lat, lon).
    Explicit bbox coordinates take precedence if all 4 are supplied.
    """
    bbox_coords = [west, south, east, north]
    has_bbox = all(c is not None for c in bbox_coords)
    has_partial_bbox = any(c is not None for c in bbox_coords) and not has_bbox

    if has_partial_bbox:
        raise ValueError(
            "Partial bounding box provided. All 4 coordinates (west, south, east, north) must be specified."
        )

    if has_bbox:
        validate_bounding_box(west, south, east, north)
        return AOIBoundingBox(
            west=round(float(west), 5),
            south=round(float(south), 5),
            east=round(float(east), 5),
            north=round(float(north), 5),
        )

    if lat is not None and lon is not None:
        return create_point_aoi(lat=float(lat), lon=float(lon), buffer_deg=float(buffer_deg))

    raise ValueError("Either lat/lon or a complete bounding box (west, south, east, north) must be supplied.")


def search_sentinel_scenes_for_aoi(
    aoi: AOIBoundingBox,
    date_range: Tuple[str, str],
    max_cloud_cover: float = 20.0,
    limit: int = 5,
    client: Optional[CopernicusSTACClient] = None,
) -> Dict[str, Any]:
    """
    Searches for Sentinel-2 Level-2A scenes matching the AOI and parameters via Copernicus STAC.
    Extracts key bands (B03, B04, B08) and identifies the optimal (least cloudy) candidate.
    """
    stac_client = client or CopernicusSTACClient()

    scenes = stac_client.search(
        bbox=aoi.to_list(),
        date_range=date_range,
        max_cloud_cover=max_cloud_cover,
        limit=limit,
    )

    if not scenes:
        return {
            "aoi": aoi.to_dict(),
            "date_range": list(date_range),
            "max_cloud_cover": max_cloud_cover,
            "total_scenes_found": 0,
            "scenes": [],
            "best_scene": None,
        }

    # Rank scenes by cloud cover percentage
    sorted_scenes = sorted(scenes, key=lambda s: s.cloud_cover_pct)
    best = sorted_scenes[0]

    # Verify key band presence on best scene
    b03 = best.get_band("B03")
    b04 = best.get_band("B04")
    b08 = best.get_band("B08")

    best_summary = {
        "scene_id": best.scene_id,
        "datetime": best.datetime,
        "cloud_cover_pct": best.cloud_cover_pct,
        "platform": best.platform,
        "thumbnail_url": best.thumbnail_url,
        "has_required_bands": all([b03 is not None, b04 is not None, b08 is not None]),
        "bands": {
            "B03_green": b03.to_dict() if b03 else None,
            "B04_red": b04.to_dict() if b04 else None,
            "B08_nir": b08.to_dict() if b08 else None,
        },
    }

    return {
        "aoi": aoi.to_dict(),
        "date_range": list(date_range),
        "max_cloud_cover": max_cloud_cover,
        "total_scenes_found": len(scenes),
        "scenes": [s.to_dict() for s in scenes],
        "best_scene": best_summary,
    }


def audit_scene_download_access(
    scene_metadata: SentinelSceneMetadata,
    client: Optional[CopernicusSTACClient] = None,
) -> Dict[str, Any]:
    """
    Checks accessibility of band assets without authentication.
    Accurately reports whether Copernicus OIDC credentials are required to download raw bands.
    """
    stac_client = client or CopernicusSTACClient()
    return stac_client.check_asset_accessibility(scene_metadata)
