"""
Bhujalsetu — Copernicus Sentinel Hub Process API Client
Integrates with the official Copernicus Data Space Ecosystem (CDSE) Sentinel Hub Process API:
https://sh.dataspace.copernicus.eu/process/v1 (or /api/v1/process)

Provides:
1. Server-side OAuth2 client_credentials token negotiation with in-memory caching.
2. Safe configuration management via CDSE_CLIENT_ID and CDSE_CLIENT_SECRET environment variables.
3. Evalscript generation requesting Sentinel-2 Level-2A B03 (Green), B04 (Red), and B08 (NIR) bands.
4. Dynamic dimension calculation to retrieve compact, georeferenced AOI GeoTIFFs rather than full tiles.
5. In-memory rasterio decoding into numpy arrays with spatial transforms and CRS.
6. Zero credential leakage across logs, representations, and errors.
"""

import os
import io
import math
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
import rasterio
import requests

DEFAULT_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
DEFAULT_PROCESS_URL = "https://sh.dataspace.copernicus.eu/process/v1"
DEFAULT_DATA_TYPE = "sentinel-2-l2a"
DEFAULT_CRS = "http://www.opengis.net/def/crs/EPSG/0/4326"

# Sentinel Hub Evalscript V3 requesting B03 (Green), B04 (Red), B08 (NIR) as Float32 reflectance
S2_L2A_EVALSCRIPT = """//VERSION=3
function setup() {
  return {
    input: [{
      bands: ["B03", "B04", "B08"],
      units: "REFLECTANCE"
    }],
    output: {
      id: "default",
      bands: 3,
      sampleType: "FLOAT32"
    }
  };
}

function evaluatePixel(sample) {
  return [sample.B03, sample.B04, sample.B08];
}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Typed Exceptions
# ─────────────────────────────────────────────────────────────────────────────

class SentinelHubError(Exception):
    """Base exception for all Copernicus Sentinel Hub client errors."""
    pass


class SentinelHubConfigError(SentinelHubError):
    """Raised when required credentials or configuration settings are missing."""
    pass


class SentinelHubAuthError(SentinelHubError):
    """Raised when OAuth2 authentication fails (e.g. invalid credentials, 401, 403)."""
    pass


class SentinelHubRequestError(SentinelHubError):
    """Raised when the Sentinel Hub Process API returns an HTTP error or malformed payload."""
    pass


class SentinelHubRasterError(SentinelHubError):
    """Raised when the returned raster data cannot be decoded or lacks spatial metadata."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SentinelHubConfig:
    """
    Configuration settings for Copernicus Sentinel Hub Process API.
    Reads CDSE_CLIENT_ID and CDSE_CLIENT_SECRET from environment variables by default.
    """
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    token_url: str = DEFAULT_TOKEN_URL
    process_url: str = DEFAULT_PROCESS_URL
    timeout_seconds: int = 45
    default_resolution_m: float = 10.0
    max_dimension: int = 1024
    min_dimension: int = 64

    def __post_init__(self):
        if self.client_id is None:
            self.client_id = os.environ.get("CDSE_CLIENT_ID")
        if self.client_secret is None:
            self.client_secret = os.environ.get("CDSE_CLIENT_SECRET")

    def is_configured(self) -> bool:
        """Returns True if both client_id and client_secret are non-empty strings."""
        return bool(self.client_id and self.client_id.strip() and self.client_secret and self.client_secret.strip())

    def validate(self) -> None:
        """Validates that credentials are fully configured."""
        if not self.is_configured():
            raise SentinelHubConfigError(
                "Copernicus Sentinel Hub credentials missing. "
                "Set CDSE_CLIENT_ID and CDSE_CLIENT_SECRET environment variables."
            )

    def __repr__(self) -> str:
        """Safe representation masking sensitive credentials."""
        has_id = bool(self.client_id and self.client_id.strip())
        has_secret = bool(self.client_secret and self.client_secret.strip())
        return (
            f"SentinelHubConfig(configured={has_id and has_secret}, "
            f"token_url='{self.token_url}', process_url='{self.process_url}', "
            f"timeout_seconds={self.timeout_seconds})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Sentinel Hub Client
# ─────────────────────────────────────────────────────────────────────────────

class SentinelHubClient:
    """
    Authenticated client for querying Copernicus Sentinel Hub Process API.
    Caches OAuth2 access tokens in memory and decodes georeferenced GeoTIFF rasters.
    """

    def __init__(
        self,
        config: Optional[SentinelHubConfig] = None,
        session: Optional[requests.Session] = None,
    ):
        self.config = config or SentinelHubConfig()
        self.session = session or requests.Session()
        self._cached_token: Optional[str] = None
        self._token_expiry_timestamp: float = 0.0

    def get_access_token(self, force_refresh: bool = False) -> str:
        """
        Retrieves a valid OAuth2 Bearer token, reusing the cached token if valid.
        Refreshes when within 60 seconds of expiration.
        """
        self.config.validate()

        now = time.time()
        # Reuse cached token if valid and not close to expiration
        if not force_refresh and self._cached_token and now < (self._token_expiry_timestamp - 60.0):
            return self._cached_token

        token_payload = {
            "grant_type": "client_credentials",
            "client_id": self.config.client_id,
            "client_secret": self.config.client_secret,
        }

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        }

        try:
            resp = self.session.post(
                self.config.token_url,
                data=token_payload,
                headers=headers,
                timeout=self.config.timeout_seconds,
            )
        except requests.RequestException as e:
            raise SentinelHubAuthError(f"OAuth2 authentication request failed: {type(e).__name__}") from None

        if resp.status_code != 200:
            status_desc = f"HTTP {resp.status_code}"
            # Never expose response body if it contains secrets; parse clean error message
            try:
                err_data = resp.json()
                err_msg = err_data.get("error_description") or err_data.get("error") or "Authentication rejected"
            except Exception:
                err_msg = "Authentication rejected"
            raise SentinelHubAuthError(f"OAuth2 authentication failed ({status_desc}): {err_msg}")

        try:
            data = resp.json()
            token = data.get("access_token")
            expires_in = int(data.get("expires_in", 300))
        except (ValueError, TypeError) as e:
            raise SentinelHubAuthError(f"Failed to parse OAuth2 token response: {e}") from None

        if not token:
            raise SentinelHubAuthError("OAuth2 response did not contain an access_token.")

        self._cached_token = token
        self._token_expiry_timestamp = now + expires_in
        return token

    def calculate_dimensions(
        self,
        bbox: Union[List[float], Tuple[float, float, float, float]],
        resolution_m: Optional[float] = None,
    ) -> Tuple[int, int]:
        """
        Calculates output width and height in pixels for an EPSG:4326 bounding box [west, south, east, north].
        Preserves aspect ratio and clamps dimensions to configured min/max bounds to maintain compact rasters.
        """
        west, south, east, north = bbox
        mid_lat = (south + north) / 2.0

        # Approximate meters per degree WGS-84
        m_per_deg_lat = 111320.0
        m_per_deg_lon = 111320.0 * math.cos(math.radians(mid_lat))

        width_m = max(1.0, abs(east - west) * m_per_deg_lon)
        height_m = max(1.0, abs(north - south) * m_per_deg_lat)

        res = resolution_m or self.config.default_resolution_m
        target_w = int(round(width_m / res))
        target_h = int(round(height_m / res))

        # Clamp to reasonable dimensions for AOI processing
        target_w = max(self.config.min_dimension, min(self.config.max_dimension, target_w))
        target_h = max(self.config.min_dimension, min(self.config.max_dimension, target_h))

        return target_w, target_h

    def build_process_payload(
        self,
        bbox: List[float],
        time_range: Tuple[str, str],
        width: int,
        height: int,
        max_cloud: float = 20.0,
    ) -> Dict[str, Any]:
        """
        Constructs the standard Copernicus Sentinel Hub Process API JSON payload.
        """
        return {
            "input": {
                "bounds": {
                    "bbox": bbox,
                    "properties": {
                        "crs": DEFAULT_CRS,
                    },
                },
                "data": [
                    {
                        "type": DEFAULT_DATA_TYPE,
                        "dataFilter": {
                            "timeRange": {
                                "from": time_range[0],
                                "to": time_range[1],
                            },
                            "maxCloudCoverage": float(max_cloud),
                        },
                    },
                ],
            },
            "output": {
                "width": int(width),
                "height": int(height),
                "responses": [
                    {
                        "identifier": "default",
                        "format": {
                            "type": "image/tiff",
                        },
                    },
                ],
            },
            "evalscript": S2_L2A_EVALSCRIPT,
        }

    def fetch_scene_raster(
        self,
        bbox: List[float],
        time_range: Tuple[str, str],
        width: Optional[int] = None,
        height: Optional[int] = None,
        max_cloud: float = 20.0,
    ) -> bytes:
        """
        Dispatches an authenticated POST request to the Sentinel Hub Process API
        and returns raw GeoTIFF bytes.
        """
        token = self.get_access_token()

        if width is None or height is None:
            width, height = self.calculate_dimensions(bbox)

        payload = self.build_process_payload(
            bbox=bbox,
            time_range=time_range,
            width=width,
            height=height,
            max_cloud=max_cloud,
        )

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "image/tiff",
        }

        try:
            resp = self.session.post(
                self.config.process_url,
                json=payload,
                headers=headers,
                timeout=self.config.timeout_seconds,
            )
        except requests.RequestException as e:
            raise SentinelHubRequestError(f"Sentinel Hub Process API request failed: {type(e).__name__}") from None

        if resp.status_code == 401 or resp.status_code == 403:
            # Token might have been revoked on the server; attempt one force refresh
            token = self.get_access_token(force_refresh=True)
            headers["Authorization"] = f"Bearer {token}"
            try:
                resp = self.session.post(
                    self.config.process_url,
                    json=payload,
                    headers=headers,
                    timeout=self.config.timeout_seconds,
                )
            except requests.RequestException as e:
                raise SentinelHubRequestError(f"Sentinel Hub Process API retry failed: {type(e).__name__}") from None

        if resp.status_code != 200:
            err_detail = "Unknown error"
            try:
                err_json = resp.json()
                err_detail = err_json.get("error", {}).get("message") or err_json.get("detail") or str(err_json)
            except Exception:
                err_detail = resp.text[:200] if resp.text else f"Status {resp.status_code}"
            raise SentinelHubRequestError(f"Sentinel Hub Process API returned HTTP {resp.status_code}: {err_detail}")

        content_type = resp.headers.get("Content-Type", "").lower()
        if "tiff" not in content_type and not resp.content.startswith(b"II*\x00") and not resp.content.startswith(b"MM\x00*"):
            raise SentinelHubRasterError(
                f"Sentinel Hub response is not a valid GeoTIFF (Content-Type: {content_type})."
            )

        return resp.content

    def decode_geotiff(
        self,
        tiff_bytes: bytes,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any, Any]:
        """
        Decodes a 3-band GeoTIFF buffer into individual band arrays (B03, B04, B08),
        spatial transform, and Coordinate Reference System (CRS).
        """
        try:
            with rasterio.open(io.BytesIO(tiff_bytes)) as src:
                count = src.count
                if count < 3:
                    raise SentinelHubRasterError(
                        f"Expected at least 3 raster bands (B03, B04, B08), got {count}."
                    )
                b03 = src.read(1)
                b04 = src.read(2)
                b08 = src.read(3)
                transform = src.transform
                crs = src.crs
        except Exception as e:
            if isinstance(e, SentinelHubRasterError):
                raise
            raise SentinelHubRasterError(f"Failed to decode GeoTIFF with rasterio: {e}") from None

        return b03, b04, b08, transform, crs

    def fetch_bands_for_scene(
        self,
        aoi_bbox: List[float],
        acquisition_datetime: str,
        max_cloud: float = 20.0,
        fixed_dimensions: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any, Any]:
        """
        Retrieves B03, B04, B08 bands for an AOI matching an exact STAC-discovered scene acquisition.

        Args:
            aoi_bbox: [west, south, east, north] in EPSG:4326.
            acquisition_datetime: ISO datetime string of the target scene (e.g. '2024-02-21T04:48:21Z').
            max_cloud: Maximum cloud coverage tolerance.
            fixed_dimensions: Optional (width, height) to enforce exact dimensions across temporal pairs.

        Returns:
            Tuple of (b03, b04, b08, affine_transform, crs)
        """
        if not acquisition_datetime:
            raise SentinelHubRequestError("acquisition_datetime is required to retrieve scene imagery.")

        # Construct single-day UTC time window around the target acquisition
        date_str = acquisition_datetime[:10]  # 'YYYY-MM-DD'
        time_range = (f"{date_str}T00:00:00Z", f"{date_str}T23:59:59Z")

        width, height = fixed_dimensions if fixed_dimensions else self.calculate_dimensions(aoi_bbox)

        tiff_bytes = self.fetch_scene_raster(
            bbox=aoi_bbox,
            time_range=time_range,
            width=width,
            height=height,
            max_cloud=max_cloud,
        )

        return self.decode_geotiff(tiff_bytes)


# ─────────────────────────────────────────────────────────────────────────────
# Application-level Helper / Factory
# ─────────────────────────────────────────────────────────────────────────────

_sentinelhub_singleton: Optional[SentinelHubClient] = None


def get_sentinelhub_client() -> Optional[SentinelHubClient]:
    """
    Returns an initialized SentinelHubClient if CDSE_CLIENT_ID and CDSE_CLIENT_SECRET
    are configured in the environment, otherwise returns None (triggering local fallback).
    """
    global _sentinelhub_singleton
    config = SentinelHubConfig()
    if config.is_configured():
        if _sentinelhub_singleton is None:
            _sentinelhub_singleton = SentinelHubClient(config)
        return _sentinelhub_singleton
    return None
