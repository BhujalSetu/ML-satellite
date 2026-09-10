"""
Unit Tests for Copernicus Sentinel Hub Process API Client and Route Integration
Verifies configuration validation, OAuth2 token negotiation and caching,
Process API payload generation, dimension calculation, in-memory GeoTIFF decoding,
error handling (HTTP 401/403, 500, network, malformed raster),
and FastAPI satellite route integration (remote mode vs local fallback).

IMPORTANT:
All tests mock Copernicus APIs and never make live external network requests.
"""

import io
import os
import time
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import rasterio
from rasterio.transform import from_origin
from fastapi.testclient import TestClient

from src.satellite.sentinelhub_client import (
    SentinelHubConfig,
    SentinelHubClient,
    SentinelHubError,
    SentinelHubConfigError,
    SentinelHubAuthError,
    SentinelHubRequestError,
    SentinelHubRasterError,
    get_sentinelhub_client,
    DEFAULT_CRS,
    S2_L2A_EVALSCRIPT,
)
from src.api.app import app


def _create_mock_3band_geotiff(
    width: int = 64,
    height: int = 64,
    crs: str = "EPSG:4326",
    west: float = 85.80,
    north: float = 20.50,
    res: float = 0.001,
) -> bytes:
    """Generates an in-memory 3-band Float32 GeoTIFF representing B03, B04, B08."""
    b03 = np.full((height, width), 0.15, dtype=np.float32)
    b04 = np.full((height, width), 0.10, dtype=np.float32)
    b08 = np.full((height, width), 0.35, dtype=np.float32)
    transform = from_origin(west, north, res, res)

    buf = io.BytesIO()
    with rasterio.open(
        buf,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype=np.float32,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(b03, 1)
        dst.write(b04, 2)
        dst.write(b08, 3)

    return buf.getvalue()


class TestSentinelHubConfig(unittest.TestCase):

    def test_config_missing_credentials(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = SentinelHubConfig()
            self.assertFalse(cfg.is_configured())
            with self.assertRaises(SentinelHubConfigError):
                cfg.validate()

    def test_config_present_credentials(self):
        with patch.dict(os.environ, {"CDSE_CLIENT_ID": "test-id", "CDSE_CLIENT_SECRET": "test-secret"}):
            cfg = SentinelHubConfig()
            self.assertTrue(cfg.is_configured())
            cfg.validate()  # Should not raise

    def test_config_safe_repr_no_secret_leak(self):
        cfg = SentinelHubConfig(client_id="my-secret-client-id", client_secret="super-confidential-secret")
        rep = repr(cfg)
        self.assertNotIn("my-secret-client-id", rep)
        self.assertNotIn("super-confidential-secret", rep)
        self.assertIn("configured=True", rep)


class TestSentinelHubClientOAuth(unittest.TestCase):

    def setUp(self):
        self.config = SentinelHubConfig(
            client_id="mock-client-id",
            client_secret="mock-client-secret",
        )

    def test_get_access_token_success_and_caching(self):
        client = SentinelHubClient(self.config)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "access_token": "valid-bearer-token-12345",
            "expires_in": 3600,
        }

        with patch.object(client.session, "post", return_value=mock_resp) as mock_post:
            token1 = client.get_access_token()
            self.assertEqual(token1, "valid-bearer-token-12345")
            self.assertEqual(mock_post.call_count, 1)

            # Second call should reuse cached token without new HTTP call
            token2 = client.get_access_token()
            self.assertEqual(token2, "valid-bearer-token-12345")
            self.assertEqual(mock_post.call_count, 1)

    def test_get_access_token_expired_refreshes(self):
        client = SentinelHubClient(self.config)

        mock_resp1 = MagicMock()
        mock_resp1.status_code = 200
        mock_resp1.json.return_value = {"access_token": "token-1", "expires_in": 3600}

        mock_resp2 = MagicMock()
        mock_resp2.status_code = 200
        mock_resp2.json.return_value = {"access_token": "token-2", "expires_in": 3600}

        with patch.object(client.session, "post", side_effect=[mock_resp1, mock_resp2]) as mock_post:
            t1 = client.get_access_token()
            self.assertEqual(t1, "token-1")

            # Simulate expiration
            client._token_expiry_timestamp = time.time() - 10

            t2 = client.get_access_token()
            self.assertEqual(t2, "token-2")
            self.assertEqual(mock_post.call_count, 2)

    def test_get_access_token_auth_failure_raises_clean_error(self):
        client = SentinelHubClient(self.config)

        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.json.return_value = {"error": "invalid_client", "error_description": "Bad credentials"}

        with patch.object(client.session, "post", return_value=mock_resp):
            with self.assertRaises(SentinelHubAuthError) as ctx:
                client.get_access_token()
            err_str = str(ctx.exception)
            self.assertIn("HTTP 401", err_str)
            # Ensure secrets are never exposed in error string
            self.assertNotIn("mock-client-secret", err_str)

    def test_get_access_token_network_error(self):
        client = SentinelHubClient(self.config)

        import requests
        with patch.object(client.session, "post", side_effect=requests.ConnectionError("Connection refused")):
            with self.assertRaises(SentinelHubAuthError) as ctx:
                client.get_access_token()
            self.assertIn("ConnectionError", str(ctx.exception))


class TestSentinelHubProcessAPI(unittest.TestCase):

    def setUp(self):
        self.config = SentinelHubConfig(
            client_id="mock-client-id",
            client_secret="mock-client-secret",
        )
        self.client = SentinelHubClient(self.config)
        self.client._cached_token = "cached-token-xyz"
        self.client._token_expiry_timestamp = time.time() + 3600

    def test_dimension_calculation(self):
        # 0.1 deg width and height around latitude 20
        bbox = [85.80, 20.40, 85.90, 20.50]
        w, h = self.client.calculate_dimensions(bbox, resolution_m=10.0)
        self.assertGreaterEqual(w, 64)
        self.assertLessEqual(w, 1024)
        self.assertGreaterEqual(h, 64)
        self.assertLessEqual(h, 1024)

    def test_build_process_payload(self):
        bbox = [85.80, 20.40, 85.90, 20.50]
        payload = self.client.build_process_payload(
            bbox=bbox,
            time_range=("2024-02-21T00:00:00Z", "2024-02-21T23:59:59Z"),
            width=256,
            height=256,
            max_cloud=15.0,
        )
        self.assertEqual(payload["input"]["bounds"]["bbox"], bbox)
        self.assertEqual(payload["input"]["bounds"]["properties"]["crs"], DEFAULT_CRS)
        self.assertEqual(payload["input"]["data"][0]["type"], "sentinel-2-l2a")
        self.assertEqual(payload["output"]["width"], 256)
        self.assertEqual(payload["output"]["height"], 256)
        self.assertEqual(payload["evalscript"], S2_L2A_EVALSCRIPT)

    def test_fetch_scene_raster_and_decode_geotiff(self):
        mock_geotiff = _create_mock_3band_geotiff(width=64, height=64)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "image/tiff"}
        mock_resp.content = mock_geotiff

        with patch.object(self.client.session, "post", return_value=mock_resp) as mock_post:
            b03, b04, b08, transform, crs = self.client.fetch_bands_for_scene(
                aoi_bbox=[85.80, 20.40, 85.90, 20.50],
                acquisition_datetime="2024-02-21T04:48:21Z",
                fixed_dimensions=(64, 64),
            )
            self.assertEqual(mock_post.call_count, 1)

            # Check authorization header in outgoing request
            call_kwargs = mock_post.call_args[1]
            auth_header = call_kwargs["headers"].get("Authorization")
            self.assertEqual(auth_header, "Bearer cached-token-xyz")

            # Check decoded array shapes and values
            self.assertEqual(b03.shape, (64, 64))
            self.assertEqual(b04.shape, (64, 64))
            self.assertEqual(b08.shape, (64, 64))
            self.assertAlmostEqual(float(b03[0, 0]), 0.15, places=4)
            self.assertAlmostEqual(float(b04[0, 0]), 0.10, places=4)
            self.assertAlmostEqual(float(b08[0, 0]), 0.35, places=4)
            self.assertIsNotNone(transform)
            self.assertIsNotNone(crs)

    def test_fetch_scene_raster_http_500_raises_request_error(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.json.return_value = {"error": {"message": "Internal Sentinel Hub Error"}}
        mock_resp.headers = {"Content-Type": "application/json"}

        with patch.object(self.client.session, "post", return_value=mock_resp):
            with self.assertRaises(SentinelHubRequestError) as ctx:
                self.client.fetch_bands_for_scene(
                    aoi_bbox=[85.80, 20.40, 85.90, 20.50],
                    acquisition_datetime="2024-02-21T04:48:21Z",
                )
            self.assertIn("HTTP 500", str(ctx.exception))

    def test_fetch_scene_raster_non_tiff_raises_raster_error(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "text/html"}
        mock_resp.content = b"<html>Service Busy</html>"

        with patch.object(self.client.session, "post", return_value=mock_resp):
            with self.assertRaises(SentinelHubRasterError) as ctx:
                self.client.fetch_bands_for_scene(
                    aoi_bbox=[85.80, 20.40, 85.90, 20.50],
                    acquisition_datetime="2024-02-21T04:48:21Z",
                )
            self.assertIn("not a valid GeoTIFF", str(ctx.exception))

    def test_decode_geotiff_insufficient_bands_raises(self):
        # Create a 1-band GeoTIFF
        buf = io.BytesIO()
        with rasterio.open(
            buf,
            "w",
            driver="GTiff",
            height=16,
            width=16,
            count=1,
            dtype=np.float32,
            crs="EPSG:4326",
            transform=from_origin(85.8, 20.5, 0.001, 0.001),
        ) as dst:
            dst.write(np.zeros((16, 16), dtype=np.float32), 1)

        with self.assertRaises(SentinelHubRasterError) as ctx:
            self.client.decode_geotiff(buf.getvalue())
        self.assertIn("Expected at least 3 raster bands", str(ctx.exception))


class TestSatelliteRoutesIntegration(unittest.TestCase):
    """
    Tests satellite routes in src/api/routes/satellite.py with:
    1. Authenticated Sentinel Hub Process API remote mode.
    2. Local JP2 development fallback mode.
    3. Missing credentials & missing local files 503 handling.
    4. Path leakage and credential leakage audits.
    """

    def setUp(self):
        self.client = TestClient(app)
        self.mock_discovery = {
            "total_scenes_found": 1,
            "best_scene": {
                "scene_id": "S2A_MSIL2A_20240221T044821_N0510_R076_T45QUC_20240221T075652",
                "datetime": "2024-02-21T04:48:21Z",
                "cloud_cover_pct": 0.01,
            },
            "scenes": [
                {
                    "scene_id": "S2A_MSIL2A_20240221T044821_N0510_R076_T45QUC_20240221T075652",
                    "datetime": "2024-02-21T04:48:21Z",
                    "cloud_cover_pct": 0.01,
                }
            ],
        }

    def test_satellite_analyze_remote_sentinelhub_flow(self):
        # Mock SentinelHubClient returning 3-band arrays
        b03 = np.full((64, 64), 0.15, dtype=np.float32)
        b04 = np.full((64, 64), 0.10, dtype=np.float32)
        b08 = np.full((64, 64), 0.35, dtype=np.float32)
        trans = from_origin(85.80, 20.50, 0.001, 0.001)

        mock_sh = MagicMock()
        mock_sh.fetch_bands_for_scene.return_value = (b03, b04, b08, trans, "EPSG:4326")

        with patch("src.api.routes.satellite.search_sentinel_scenes_for_aoi", return_value=self.mock_discovery), \
             patch("src.api.routes.satellite.get_sentinelhub_client", return_value=mock_sh):

            response = self.client.post(
                "/api/v1/satellite/analyze",
                json={
                    "latitude": 20.4625,
                    "longitude": 85.8828,
                    "buffer": 0.05,
                    "start_date": "2024-02-01",
                    "end_date": "2024-03-01",
                    "max_cloud": 15.0,
                },
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertTrue(data["success"])
            self.assertIn("indices", data)
            self.assertIn("ndvi_mean", data["indices"])
            self.assertIn("ndvi_min", data["indices"])
            self.assertIn("ndvi_max", data["indices"])
            self.assertIn("ndwi_mean", data["indices"])
            self.assertIn("ndwi_min", data["indices"])
            self.assertIn("ndwi_max", data["indices"])
            self.assertIsInstance(data["indices"]["ndvi_mean"], float)
            self.assertIsInstance(data["indices"]["ndwi_mean"], float)
            self.assertAlmostEqual(data["indices"]["ndvi_mean"], data["indices"]["ndvi"]["mean"])
            self.assertAlmostEqual(data["indices"]["ndwi_mean"], data["indices"]["ndwi"]["mean"])
            self.assertIn("outputs", data)
            self.assertTrue(data["outputs"]["ndvi_raster_url"].startswith("/files/satellite/"))
            self.assertTrue(data["outputs"]["ndwi_raster_url"].startswith("/files/satellite/"))
            self.assertTrue(data["outputs"]["water_mask_url"].startswith("/files/satellite/"))

            # Verify no secret or local drive leakage
            res_str = str(data)
            self.assertNotIn("C:\\", res_str)
            self.assertNotIn("client_secret", res_str)

    def test_satellite_analyze_local_fallback_when_unconfigured(self):
        # When get_sentinelhub_client returns None, falls back to resolve_local_scene_bands
        b03 = np.full((64, 64), 0.15, dtype=np.float32)
        b04 = np.full((64, 64), 0.10, dtype=np.float32)
        b08 = np.full((64, 64), 0.35, dtype=np.float32)
        trans = from_origin(85.80, 20.50, 0.001, 0.001)

        with patch("src.api.routes.satellite.search_sentinel_scenes_for_aoi", return_value=self.mock_discovery), \
             patch("src.api.routes.satellite.get_sentinelhub_client", return_value=None), \
             patch("src.api.routes.satellite.resolve_local_scene_bands", return_value={"B03": "p3", "B04": "p4", "B08": "p8"}), \
             patch("src.api.routes.satellite.load_band_arrays", return_value=(b03, b04, b08, trans, "EPSG:4326")):

            response = self.client.post(
                "/api/v1/satellite/analyze",
                json={
                    "latitude": 20.4625,
                    "longitude": 85.8828,
                    "buffer": 0.05,
                },
            )
            self.assertEqual(response.status_code, 200)

    def test_satellite_change_remote_sentinelhub_flow(self):
        b03 = np.full((64, 64), 0.15, dtype=np.float32)
        b04 = np.full((64, 64), 0.10, dtype=np.float32)
        b08 = np.full((64, 64), 0.35, dtype=np.float32)
        trans = from_origin(85.80, 20.50, 0.001, 0.001)

        mock_sh = MagicMock()
        mock_sh.calculate_dimensions.return_value = (64, 64)
        mock_sh.fetch_bands_for_scene.return_value = (b03, b04, b08, trans, "EPSG:4326")

        mock_before = {
            "total_scenes_found": 1,
            "best_scene": {
                "scene_id": "S2A_MSIL2A_20240221T044821_N0510_R076_T45QUC_20240221T075652",
                "datetime": "2024-02-21T04:48:21Z",
                "cloud_cover_pct": 0.01,
            },
            "scenes": [],
        }
        mock_after = {
            "total_scenes_found": 1,
            "best_scene": {
                "scene_id": "S2A_MSIL2A_20240302T044711_N0510_R076_T45QUC_20240302T090154",
                "datetime": "2024-03-02T04:47:11Z",
                "cloud_cover_pct": 0.01,
            },
            "scenes": [],
        }

        with patch("src.api.routes.satellite.search_sentinel_scenes_for_aoi", side_effect=[mock_before, mock_after]), \
             patch("src.api.routes.satellite.get_sentinelhub_client", return_value=mock_sh):

            response = self.client.post(
                "/api/v1/satellite/change",
                json={
                    "latitude": 20.4625,
                    "longitude": 85.8828,
                    "buffer": 0.05,
                    "before_date": "2024-02-21",
                    "after_date": "2024-03-02",
                },
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertTrue(data["success"])
            self.assertEqual(data["status"], "completed")
            self.assertIn("outputs", data)
            self.assertTrue(data["outputs"]["ndvi_change_raster_url"].startswith("/files/satellite/"))
            self.assertTrue(data["outputs"]["summary_json_url"].startswith("/files/satellite/"))

            # Path & credential leakage verification
            res_str = str(data)
            self.assertNotIn("C:\\", res_str)
            self.assertNotIn("client_secret", res_str)

    def test_satellite_analyze_truthful_503_when_unconfigured_and_no_local_bands(self):
        # When unconfigured AND local bands are absent -> HTTP 503
        with patch("src.api.routes.satellite.search_sentinel_scenes_for_aoi", return_value=self.mock_discovery), \
             patch("src.api.routes.satellite.get_sentinelhub_client", return_value=None), \
             patch("src.api.routes.satellite.resolve_local_scene_bands", return_value=None):

            response = self.client.post(
                "/api/v1/satellite/analyze",
                json={"latitude": 20.4625, "longitude": 85.8828},
            )
            self.assertEqual(response.status_code, 503)
            self.assertIn("inaccessible", response.json()["detail"].lower())
            self.assertIn("authentication", response.json()["detail"].lower())

    def test_satellite_change_truthful_503_when_unconfigured_and_no_local_bands(self):
        mock_before = {
            "total_scenes_found": 1,
            "best_scene": {
                "scene_id": "S2A_MSIL2A_20240221T044821_N0510_R076_T45QUC_20240221T075652",
                "datetime": "2024-02-21T04:48:21Z",
                "cloud_cover_pct": 0.01,
            },
            "scenes": [],
        }
        mock_after = {
            "total_scenes_found": 1,
            "best_scene": {
                "scene_id": "S2A_MSIL2A_20240302T044711_N0510_R076_T45QUC_20240302T090154",
                "datetime": "2024-03-02T04:47:11Z",
                "cloud_cover_pct": 0.01,
            },
            "scenes": [],
        }

        with patch("src.api.routes.satellite.search_sentinel_scenes_for_aoi", side_effect=[mock_before, mock_after]), \
             patch("src.api.routes.satellite.get_sentinelhub_client", return_value=None), \
             patch("src.api.routes.satellite.resolve_local_scene_bands", return_value=None):

            response = self.client.post(
                "/api/v1/satellite/change",
                json={
                    "latitude": 20.4625,
                    "longitude": 85.8828,
                    "buffer": 0.05,
                    "before_date": "2024-02-21",
                    "after_date": "2024-03-02",
                },
            )
            self.assertEqual(response.status_code, 503)
            self.assertIn("inaccessible", response.json()["detail"].lower())
            self.assertIn("authentication", response.json()["detail"].lower())


if __name__ == "__main__":
    unittest.main()
