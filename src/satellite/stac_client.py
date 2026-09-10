"""
WATERSCOPE-AI — Sentinel-2 Copernicus STAC Client
Connects to the official Copernicus Data Space Ecosystem (CDSE) STAC API:
https://stac.dataspace.copernicus.eu/v1/

Searches Level-2A (sentinel-2-l2a) scenes by bounding box (AOI), date range,
and cloud-cover filter. Parses scene metadata and identifies Red (B04),
Green (B03), and NIR (B08) band assets for downstream NDVI/NDWI processing.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Union, Any
from pathlib import Path
import json
import requests

DEFAULT_STAC_URL = "https://stac.dataspace.copernicus.eu/v1"
DEFAULT_COLLECTION = "sentinel-2-l2a"

# Standard headers required by CDSE Cloudflare / WAF
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/geo+json, application/json",
    "Content-Type": "application/json",
}

@dataclass
class BandAsset:
    """Represents a specific spectral band asset inside a Sentinel-2 scene."""
    band_id: str          # e.g. 'B04', 'B03', 'B08'
    resolution: str       # e.g. '10m', '20m', '60m'
    asset_key: str        # e.g. 'B04_10m'
    s3_href: str          # S3 URI (s3://eodata/...)
    https_href: Optional[str] = None  # Download HTTPS URL
    mime_type: Optional[str] = "image/jp2"
    data_type: Optional[str] = "uint16"
    file_size_bytes: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class SentinelSceneMetadata:
    """Structured representation of a Sentinel-2 Level-2A STAC Item."""
    scene_id: str
    collection: str
    datetime: str
    cloud_cover_pct: float
    bbox: List[float]
    platform: str
    thumbnail_url: Optional[str] = None
    bands: Dict[str, BandAsset] = field(default_factory=dict)
    properties: Dict[str, Any] = field(default_factory=dict)

    def get_band(self, band_name: str, prefer_resolution: str = "10m") -> Optional[BandAsset]:
        """
        Retrieves a band asset by name (e.g. 'B04', 'B03', 'B08'), preferring 10m resolution.
        """
        key = f"{band_name.upper()}_{prefer_resolution}"
        if key in self.bands:
            return self.bands[key]
        for b_key, asset in self.bands.items():
            if asset.band_id.upper() == band_name.upper():
                return asset
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "collection": self.collection,
            "datetime": self.datetime,
            "cloud_cover_pct": self.cloud_cover_pct,
            "bbox": self.bbox,
            "platform": self.platform,
            "thumbnail_url": self.thumbnail_url,
            "bands": {k: b.to_dict() for k, b in self.bands.items()},
            "identified_bands": {
                "red_B04": self.get_band("B04").to_dict() if self.get_band("B04") else None,
                "green_B03": self.get_band("B03").to_dict() if self.get_band("B03") else None,
                "nir_B08": self.get_band("B08").to_dict() if self.get_band("B08") else None,
            }
        }

class CopernicusSTACClient:
    """
    Client for querying the Copernicus Data Space Ecosystem (CDSE) STAC API.
    Supports AOI bounding box search, datetime interval filtering, cloud cover thresholds,
    and asset extraction.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_STAC_URL,
        collection: str = DEFAULT_COLLECTION,
        timeout: int = 30,
        session: Optional[requests.Session] = None
    ):
        self.base_url = base_url.rstrip("/")
        self.collection = collection
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)

    def build_search_payload(
        self,
        bbox: Union[Tuple[float, float, float, float], List[float]],
        date_range: Union[str, Tuple[str, str], List[str]],
        max_cloud_cover: float = 20.0,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """
        Builds the standard STAC search POST payload.

        Args:
            bbox: [min_lon, min_lat, max_lon, max_lat]
            date_range: ISO formatted range string (e.g. '2024-01-01T00:00:00Z/2024-03-31T23:59:59Z')
                        or tuple/list of two date strings ('2024-01-01', '2024-03-31').
            max_cloud_cover: Upper threshold for cloud cover percentage (0-100).
            limit: Maximum items to return.
        """
        if len(bbox) != 4:
            raise ValueError(f"bbox must be 4 coordinates [min_lon, min_lat, max_lon, max_lat], got {bbox}")

        if isinstance(date_range, (tuple, list)):
            if len(date_range) != 2:
                raise ValueError("date_range tuple/list must contain exactly 2 elements: (start, end)")
            start = date_range[0] if "T" in date_range[0] else f"{date_range[0]}T00:00:00Z"
            end = date_range[1] if "T" in date_range[1] else f"{date_range[1]}T23:59:59Z"
            datetime_str = f"{start}/{end}"
        else:
            datetime_str = date_range

        payload: Dict[str, Any] = {
            "collections": [self.collection],
            "bbox": [float(c) for c in bbox],
            "datetime": datetime_str,
            "limit": int(limit),
        }

        if max_cloud_cover is not None and max_cloud_cover < 100.0:
            payload["query"] = {
                "eo:cloud_cover": {"lte": float(max_cloud_cover)}
            }

        return payload

    def search(
        self,
        bbox: Union[Tuple[float, float, float, float], List[float]],
        date_range: Union[str, Tuple[str, str], List[str]],
        max_cloud_cover: float = 20.0,
        limit: int = 10,
    ) -> List[SentinelSceneMetadata]:
        """
        Executes a search against Copernicus STAC API and returns parsed scene metadata objects.
        """
        payload = self.build_search_payload(
            bbox=bbox,
            date_range=date_range,
            max_cloud_cover=max_cloud_cover,
            limit=limit,
        )

        search_endpoint = f"{self.base_url}/search"
        response = self.session.post(search_endpoint, json=payload, timeout=self.timeout)

        if response.status_code != 200:
            raise RuntimeError(
                f"Copernicus STAC search failed (HTTP {response.status_code}): {response.text[:300]}"
            )

        content_type = response.headers.get("Content-Type", "")
        if "json" not in content_type and response.text.startswith("<"):
            raise RuntimeError(
                f"Copernicus STAC returned unexpected non-JSON response: {response.text[:200]}"
            )

        data = response.json()
        features = data.get("features", [])

        scenes: List[SentinelSceneMetadata] = []
        for feat in features:
            scenes.append(self.parse_stac_feature(feat))

        return scenes

    def parse_stac_feature(self, feature: Dict[str, Any]) -> SentinelSceneMetadata:
        """
        Parses a single GeoJSON STAC Feature into a structured SentinelSceneMetadata.
        """
        scene_id = feature.get("id", "unknown")
        collection = feature.get("collection", self.collection)
        bbox = feature.get("bbox", [])
        props = feature.get("properties", {})

        dt = props.get("datetime") or props.get("start_datetime") or "unknown"
        cloud = float(props.get("eo:cloud_cover", props.get("cloudCover", 0.0)))
        platform = props.get("platform", "Sentinel-2")

        # Parse Assets
        assets_raw = feature.get("assets", {})
        bands: Dict[str, BandAsset] = {}
        thumbnail_url = None

        if "thumbnail" in assets_raw:
            thumbnail_url = assets_raw["thumbnail"].get("href")

        target_bands = ["B02", "B03", "B04", "B08", "B11", "B12", "TCI"]

        for asset_key, a_data in assets_raw.items():
            for tb in target_bands:
                if asset_key.startswith(tb):
                    parts = asset_key.split("_")
                    resolution = parts[1] if len(parts) > 1 else "10m"

                    s3_href = a_data.get("href", "")
                    https_href = None

                    alternate = a_data.get("alternate", {})
                    if "https" in alternate and isinstance(alternate["https"], dict):
                        https_href = alternate["https"].get("href")

                    bands[asset_key] = BandAsset(
                        band_id=tb,
                        resolution=resolution,
                        asset_key=asset_key,
                        s3_href=s3_href,
                        https_href=https_href,
                        mime_type=a_data.get("type", "image/jp2"),
                        data_type=a_data.get("data_type", "uint16"),
                        file_size_bytes=a_data.get("file:size"),
                    )

        return SentinelSceneMetadata(
            scene_id=scene_id,
            collection=collection,
            datetime=dt,
            cloud_cover_pct=round(cloud, 2),
            bbox=bbox,
            platform=platform,
            thumbnail_url=thumbnail_url,
            bands=bands,
            properties=props,
        )

    def check_asset_accessibility(self, scene: SentinelSceneMetadata) -> Dict[str, Any]:
        """
        Checks public HTTP accessibility of thumbnail and spectral band assets without authentication.
        Reports whether credentials (CDSE Bearer token / S3 credentials) are required.
        """
        results: Dict[str, Any] = {
            "scene_id": scene.scene_id,
            "thumbnail_accessible": False,
            "band_assets_accessible_without_auth": False,
            "authentication_required": True,
            "auth_type": "Copernicus Data Space OIDC / OAuth2 Bearer Token",
            "details": {}
        }

        # Check thumbnail with stream GET
        if scene.thumbnail_url:
            try:
                r_thumb = self.session.get(scene.thumbnail_url, stream=True, timeout=10)
                results["thumbnail_accessible"] = (r_thumb.status_code == 200)
                results["details"]["thumbnail_status"] = r_thumb.status_code
            except Exception as e:
                results["details"]["thumbnail_error"] = str(e)

        # Check B04 HTTPS alternate link with stream GET
        b04 = scene.get_band("B04")
        if b04 and b04.https_href:
            try:
                r_band = self.session.get(b04.https_href, stream=True, timeout=10)
                results["details"]["band_https_status"] = r_band.status_code
                if r_band.status_code == 200:
                    results["band_assets_accessible_without_auth"] = True
                    results["authentication_required"] = False
                elif r_band.status_code in (401, 403):
                    results["band_assets_accessible_without_auth"] = False
                    results["authentication_required"] = True
            except Exception as e:
                results["details"]["band_https_error"] = str(e)

        return results

    def find_temporal_pair(
        self,
        bbox: Union[Tuple[float, float, float, float], List[float]],
        t0_range: Union[str, Tuple[str, str], List[str]],
        t1_range: Union[str, Tuple[str, str], List[str]],
        max_cloud_cover: float = 20.0,
    ) -> Dict[str, Optional[SentinelSceneMetadata]]:
        """
        Discovers a pair of low-cloud scenes for temporal water body analysis:
        one from the initial period T0 (e.g. dry season) and one from subsequent period T1.
        """
        scenes_t0 = self.search(bbox=bbox, date_range=t0_range, max_cloud_cover=max_cloud_cover, limit=5)
        scenes_t1 = self.search(bbox=bbox, date_range=t1_range, max_cloud_cover=max_cloud_cover, limit=5)

        best_t0 = min(scenes_t0, key=lambda s: s.cloud_cover_pct) if scenes_t0 else None
        best_t1 = min(scenes_t1, key=lambda s: s.cloud_cover_pct) if scenes_t1 else None

        return {
            "t0_scene": best_t0,
            "t1_scene": best_t1,
        }
