"""
Unit Tests for WATERSCOPE-AI FastAPI Application Foundation and Pydantic Schemas
Verifies app initialization, GET /health endpoint, static file mount access,
and strict validation logic across AI & Satellite Pydantic v2 schemas.
"""

import unittest
from datetime import date
from pydantic import ValidationError
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.schemas import (
    HealthResponse,
    LocationCoordinates,
    AOIParameters,
    AIAnalyzeImageRequest,
    AIDetectionItem,
    AIDetectionSummary,
    AIAnalyzeImageOutputs,
    AIAnalyzeImageResponse,
    AIChangeDetectionRequest,
    AIChangeDetail,
    AIChangeDetectionOutputs,
    AIChangeDetectionResponse,
    SatelliteAnalyzeRequest,
    SceneMetadataSummary,
    RasterIndexStats,
    SatelliteIndices,
    SatelliteAnalyzeOutputs,
    SatelliteAnalyzeResponse,
    SatelliteChangeRequest,
    SceneObservationItem,
    NDVIChangePercentages,
    NDWIChangePercentages,
    SatelliteChangeOutputs,
    SatelliteChangeResponse,
)

class TestAPIApplication(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["service"], "WATERSCOPE-AI API")
        self.assertEqual(data["version"], "1.0.0")

    def test_static_files_mount_exists(self):
        # Requesting non-existent file should yield 404 from StaticFiles handler, not 500
        response = self.client.get("/files/non_existent_test_file.xyz")
        self.assertEqual(response.status_code, 404)


class TestAPISchemasValidation(unittest.TestCase):

    # ─────────────────────────────────────────────────────────────────────────
    # 1. AI Analyze Image Schema Tests
    # ─────────────────────────────────────────────────────────────────────────
    def test_ai_analyze_image_valid(self):
        req = AIAnalyzeImageRequest(latitude=20.4625, longitude=85.8828)
        self.assertAlmostEqual(req.latitude, 20.4625)
        self.assertAlmostEqual(req.longitude, 85.8828)

        resp = AIAnalyzeImageResponse(
            success=True,
            request_id="req-123",
            image_id="img-001",
            location=LocationCoordinates(latitude=20.4625, longitude=85.8828),
            model="yolo26n-baseline",
            detections=[
                AIDetectionItem(class_name="boat", confidence=0.8523, bbox=[10.0, 20.0, 100.0, 150.0])
            ],
            summary=AIDetectionSummary(objects_detected=1, ponds_detected=0),
            outputs=AIAnalyzeImageOutputs(annotated_image_url="/files/ai/annotated_img-001.jpg"),
            processing_time_ms=45.2,
        )
        self.assertTrue(resp.success)
        self.assertEqual(len(resp.detections), 1)
        self.assertEqual(resp.detections[0].class_name, "boat")
        self.assertEqual(resp.outputs.annotated_image_url, "/files/ai/annotated_img-001.jpg")

    def test_ai_analyze_image_invalid_coordinates(self):
        # Latitude out of range (> 90)
        with self.assertRaises(ValidationError):
            AIAnalyzeImageRequest(latitude=95.0, longitude=85.0)

        # Latitude out of range (< -90)
        with self.assertRaises(ValidationError):
            AIAnalyzeImageRequest(latitude=-90.1, longitude=85.0)

        # Longitude out of range (> 180)
        with self.assertRaises(ValidationError):
            AIAnalyzeImageRequest(latitude=20.0, longitude=185.0)

        # Longitude out of range (< -180)
        with self.assertRaises(ValidationError):
            AIAnalyzeImageRequest(latitude=20.0, longitude=-180.5)

    def test_ai_detection_invalid_confidence_and_bbox(self):
        # Confidence > 1.0
        with self.assertRaises(ValidationError):
            AIDetectionItem(class_name="boat", confidence=1.2, bbox=[0.0, 0.0, 10.0, 10.0])

        # Confidence < 0.0
        with self.assertRaises(ValidationError):
            AIDetectionItem(class_name="boat", confidence=-0.1, bbox=[0.0, 0.0, 10.0, 10.0])

        # Bounding box with 3 elements (must be exactly 4)
        with self.assertRaises(ValidationError):
            AIDetectionItem(class_name="boat", confidence=0.8, bbox=[0.0, 0.0, 10.0])

        # Bounding box with 5 elements
        with self.assertRaises(ValidationError):
            AIDetectionItem(class_name="boat", confidence=0.8, bbox=[0.0, 0.0, 10.0, 10.0, 12.0])

    # ─────────────────────────────────────────────────────────────────────────
    # 2. AI Change Detection Schema Tests
    # ─────────────────────────────────────────────────────────────────────────
    def test_ai_change_detection_valid_and_optional_fields(self):
        # Valid with all fields
        resp_full = AIChangeDetectionResponse(
            success=True,
            request_id="req-change-1",
            change_detected=True,
            change_type="New Farm Pond",
            confidence=0.89,
            change_percentage=12.5,
            changes=[AIChangeDetail(type="New Farm Pond", confidence=0.89)],
            outputs=AIChangeDetectionOutputs(
                change_map_url="/files/ai/change_map_001.png",
                annotated_before_url="/files/ai/t0_001.png",
                annotated_after_url="/files/ai/t1_001.png",
            ),
            processing_time_ms=120.5,
        )
        self.assertTrue(resp_full.change_detected)
        self.assertEqual(resp_full.change_percentage, 12.5)

        # Valid with genuine nulls / optional fields
        resp_nulls = AIChangeDetectionResponse(
            success=True,
            request_id="req-change-2",
            change_detected=False,
            change_type=None,
            confidence=None,
            change_percentage=None,
            changes=[],
            outputs=AIChangeDetectionOutputs(change_map_url=None),
            processing_time_ms=85.0,
        )
        self.assertFalse(resp_nulls.change_detected)
        self.assertIsNone(resp_nulls.change_type)
        self.assertIsNone(resp_nulls.confidence)
        self.assertIsNone(resp_nulls.change_percentage)

    def test_ai_change_detection_invalid_percentage_and_confidence(self):
        with self.assertRaises(ValidationError):
            AIChangeDetectionResponse(
                success=True,
                request_id="req-1",
                change_percentage=105.0,  # > 100
                outputs=AIChangeDetectionOutputs(),
                processing_time_ms=10.0,
            )

        with self.assertRaises(ValidationError):
            AIChangeDetail(type="Pond", confidence=1.5)  # > 1.0

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Satellite Analyze Schema Tests
    # ─────────────────────────────────────────────────────────────────────────
    def test_satellite_analyze_valid(self):
        req = SatelliteAnalyzeRequest(
            latitude=20.2961,
            longitude=85.8245,
            buffer=0.05,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 3, 31),
            max_cloud=15.0,
        )
        self.assertEqual(req.start_date, date(2024, 1, 1))

        resp = SatelliteAnalyzeResponse(
            success=True,
            request_id="sat-req-001",
            aoi=AOIParameters(latitude=20.2961, longitude=85.8245, buffer=0.05),
            scene=SceneMetadataSummary(
                scene_id="S2A_MSIL2A_20240221T044821_N0510_R076_T45QUC_20240221T075652",
                date="2024-02-21",
                cloud_cover=0.01,
            ),
            indices=SatelliteIndices(
                ndvi=RasterIndexStats(mean=0.2468, min=-0.4462, max=0.6649),
                ndwi=RasterIndexStats(mean=-0.2574, min=-0.6008, max=0.4673),
            ),
            outputs=SatelliteAnalyzeOutputs(
                ndvi_raster_url="/files/satellite/real/ndvi_real.tif",
                ndwi_raster_url="/files/satellite/real/ndwi_real.tif",
                water_mask_url="/files/satellite/real/water_mask_real.tif",
            ),
            processing_time_ms=5400.0,
        )
        self.assertTrue(resp.success)
        self.assertEqual(resp.indices.ndvi.mean, 0.2468)
        self.assertTrue(resp.outputs.ndvi_raster_url.startswith("/files/satellite/"))

    def test_satellite_analyze_invalid_cloud_cover(self):
        with self.assertRaises(ValidationError):
            SatelliteAnalyzeRequest(latitude=20.0, longitude=85.0, max_cloud=120.0)  # > 100

        with self.assertRaises(ValidationError):
            SceneMetadataSummary(scene_id="S2A_TEST", date="2024-02-21", cloud_cover=-5.0)  # < 0

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Satellite Change Schema Tests
    # ─────────────────────────────────────────────────────────────────────────
    def test_satellite_change_completed_and_unavailable_status(self):
        # Case A: Completed status with full dual observations
        resp_completed = SatelliteChangeResponse(
            success=True,
            request_id="sat-change-1",
            status="completed",
            aoi=AOIParameters(latitude=20.4625, longitude=85.8828, buffer=0.05),
            before=SceneObservationItem(date="2024-01-15", scene_id="S2A_SCENE_T0"),
            after=SceneObservationItem(date="2024-02-15", scene_id="S2A_SCENE_T1"),
            ndvi_change=NDVIChangePercentages(gain_percent=12.4, loss_percent=4.1, stable_percent=83.5),
            ndwi_change=NDWIChangePercentages(water_gain_percent=9.14, water_loss_percent=0.0, stable_percent=90.86),
            outputs=SatelliteChangeOutputs(
                ndvi_change_raster_url="/files/satellite/temporal/synthetic/ndvi_change.tif",
                ndwi_change_raster_url="/files/satellite/temporal/synthetic/ndwi_change.tif",
                water_change_raster_url="/files/satellite/temporal/synthetic/water_change.tif",
            ),
            processing_time_ms=850.0,
        )
        self.assertEqual(resp_completed.status, "completed")
        self.assertAlmostEqual(resp_completed.ndvi_change.gain_percent, 12.4)

        # Case B: Unavailable status when second observation is missing (Strict Project Truth)
        resp_unavailable = SatelliteChangeResponse(
            success=True,
            request_id="sat-change-2",
            status="unavailable",
            aoi=AOIParameters(latitude=20.4625, longitude=85.8828, buffer=0.05),
            before=SceneObservationItem(date="2024-02-21", scene_id="S2A_MSIL2A_20240221"),
            after=None,
            ndvi_change=None,
            ndwi_change=None,
            outputs=None,
            processing_time_ms=45.0,
            notice="Temporal comparison requires two compatible observations.",
        )
        self.assertEqual(resp_unavailable.status, "unavailable")
        self.assertIsNone(resp_unavailable.after)
        self.assertIsNone(resp_unavailable.ndvi_change)
        self.assertIn("requires two compatible observations", resp_unavailable.notice)

    def test_satellite_change_invalid_percentages(self):
        with self.assertRaises(ValidationError):
            NDVIChangePercentages(gain_percent=110.0, loss_percent=10.0, stable_percent=80.0)

        with self.assertRaises(ValidationError):
            NDWIChangePercentages(water_gain_percent=-2.0, water_loss_percent=0.0, stable_percent=100.0)


class TestAIAnalyzeImageEndpoint(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        # Create a small valid test image in-memory using OpenCV
        import cv2
        import numpy as np
        # 64x64 green square image
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[:] = (34, 139, 34)
        _, self.valid_image_bytes = cv2.imencode(".jpg", img)

    def test_analyze_image_endpoint_exists_and_success(self):
        # 1. Endpoint exists & valid upload returns 200 OK
        response = self.client.post(
            "/api/v1/ai/analyze-image",
            data={"latitude": 20.4625, "longitude": 85.8828},
            files={"file": ("test_frame.jpg", self.valid_image_bytes.tobytes(), "image/jpeg")},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        # 3. Response contains all required contract keys
        self.assertTrue(data["success"])
        self.assertIn("request_id", data)
        self.assertIn("image_id", data)
        self.assertIn("location", data)
        self.assertAlmostEqual(data["location"]["latitude"], 20.4625)
        self.assertAlmostEqual(data["location"]["longitude"], 85.8828)
        self.assertIn("model", data)
        self.assertIn("detections", data)
        self.assertIn("summary", data)
        self.assertIn("outputs", data)
        self.assertIn("processing_time_ms", data)
        self.assertGreaterEqual(data["processing_time_ms"], 0.0)

        # 4. Output annotated_image_url begins with /files/
        annotated_url = data["outputs"]["annotated_image_url"]
        self.assertIsNotNone(annotated_url)
        self.assertTrue(annotated_url.startswith("/files/ai/"))
        self.assertNotIn("C:", annotated_url)
        self.assertNotIn("\\", annotated_url)

        # Verify static files mount serves the saved image
        static_resp = self.client.get(annotated_url)
        self.assertEqual(static_resp.status_code, 200)

    def test_analyze_image_zero_detection_case(self):
        # 9. Test zero-detection case (pure blank image produces 0 COCO detections)
        response = self.client.post(
            "/api/v1/ai/analyze-image",
            data={"latitude": 20.4625, "longitude": 85.8828},
            files={"file": ("blank.jpg", self.valid_image_bytes.tobytes(), "image/jpeg")},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data["detections"]), data["summary"]["objects_detected"])
        self.assertEqual(data["summary"]["ponds_detected"], 0)

    def test_analyze_image_invalid_latitude_rejected(self):
        # 5. Invalid latitude rejected (HTTP 422)
        response = self.client.post(
            "/api/v1/ai/analyze-image",
            data={"latitude": 95.0, "longitude": 85.8828},
            files={"file": ("test.jpg", self.valid_image_bytes.tobytes(), "image/jpeg")},
        )
        self.assertEqual(response.status_code, 422)

    def test_analyze_image_invalid_longitude_rejected(self):
        # 6. Invalid longitude rejected (HTTP 422)
        response = self.client.post(
            "/api/v1/ai/analyze-image",
            data={"latitude": 20.0, "longitude": 195.0},
            files={"file": ("test.jpg", self.valid_image_bytes.tobytes(), "image/jpeg")},
        )
        self.assertEqual(response.status_code, 422)

    def test_analyze_image_missing_file_rejected(self):
        # 7. Missing file rejected (HTTP 422)
        response = self.client.post(
            "/api/v1/ai/analyze-image",
            data={"latitude": 20.4625, "longitude": 85.8828},
        )
        self.assertEqual(response.status_code, 422)

    def test_analyze_image_invalid_non_image_file_rejected(self):
        # 8. Invalid/non-image file rejected (HTTP 400)
        fake_bytes = b"This is plain text and definitely not a valid image format."
        response = self.client.post(
            "/api/v1/ai/analyze-image",
            data={"latitude": 20.4625, "longitude": 85.8828},
            files={"file": ("corrupt.txt", fake_bytes, "text/plain")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Could not decode image bytes", response.json()["detail"])


class TestAIChangeDetectionEndpoint(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        import cv2
        import numpy as np

        # Base 64x64 gray test image
        self.img0 = np.zeros((64, 64, 3), dtype=np.uint8)
        self.img0[:] = (60, 60, 60)
        _, enc0 = cv2.imencode(".jpg", self.img0)
        self.valid_bytes_0 = enc0.tobytes()

        # Altered 64x64 image with modified patch (brightening: demolished/filled)
        self.img1 = self.img0.copy()
        self.img1[15:35, 15:35] = (220, 220, 220)
        _, enc1 = cv2.imencode(".jpg", self.img1)
        self.valid_bytes_1 = enc1.tobytes()

        # Altered 64x64 image with darkening patch (constructed/excavated)
        self.img2 = self.img0.copy()
        self.img2[15:35, 15:35] = (10, 10, 10)
        _, enc2 = cv2.imencode(".jpg", self.img2)
        self.valid_bytes_2 = enc2.tobytes()

        # Incompatible dimension image (128x128)
        self.img_large = np.zeros((128, 128, 3), dtype=np.uint8)
        _, enc_large = cv2.imencode(".jpg", self.img_large)
        self.large_bytes = enc_large.tobytes()

    def test_change_detection_endpoint_exists_and_success_with_change(self):
        # 1. Endpoint exists & 2. Valid before/after produce HTTP 200
        response = self.client.post(
            "/api/v1/ai/change-detection",
            data={"latitude": 20.4625, "longitude": 85.8828},
            files={
                "before_image": ("t0.jpg", self.valid_bytes_0, "image/jpeg"),
                "after_image": ("t1.jpg", self.valid_bytes_1, "image/jpeg"),
            },
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        # 3. Response contains all required contract keys
        self.assertTrue(data["success"])
        self.assertIn("request_id", data)
        self.assertIn("change_detected", data)
        self.assertTrue(data["change_detected"])
        self.assertIn("change_type", data)
        self.assertEqual(data["change_type"], "Farm Pond Demolished")
        self.assertIn("confidence", data)
        self.assertIsNone(data["confidence"])  # Truthful: null confidence
        self.assertIn("change_percentage", data)
        self.assertGreater(data["change_percentage"], 0.0)
        self.assertIn("changes", data)
        self.assertGreaterEqual(len(data["changes"]), 1)
        self.assertEqual(data["changes"][0]["type"], "Farm Pond Demolished")
        self.assertIsNone(data["changes"][0]["confidence"])
        self.assertIn("outputs", data)
        self.assertIn("processing_time_ms", data)
        self.assertGreaterEqual(data["processing_time_ms"], 0.0)

        # 4. Output URLs are API-relative
        change_url = data["outputs"]["change_map_url"]
        self.assertIsNotNone(change_url)
        self.assertTrue(change_url.startswith("/files/ai/"))
        self.assertNotIn("C:", change_url)
        self.assertNotIn("\\", change_url)

        # Verify static file mount serves the generated JPEG
        static_resp = self.client.get(change_url)
        self.assertEqual(static_resp.status_code, 200)
        self.assertEqual(static_resp.headers.get("content-type"), "image/jpeg")

    def test_change_detection_identical_images_no_change(self):
        # Test case where images are identical (no change detected)
        response = self.client.post(
            "/api/v1/ai/change-detection",
            files={
                "before_image": ("t0.jpg", self.valid_bytes_0, "image/jpeg"),
                "after_image": ("t0_dup.jpg", self.valid_bytes_0, "image/jpeg"),
            },
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertFalse(data["change_detected"])
        self.assertIsNone(data["change_type"])
        self.assertIsNone(data["confidence"])
        self.assertEqual(data["change_percentage"], 0.0)
        self.assertEqual(data["changes"], [])
        self.assertTrue(data["outputs"]["change_map_url"].startswith("/files/ai/"))

    def test_change_detection_missing_before_image_rejected(self):
        # 5. Missing before image is rejected (HTTP 422 or 400)
        response = self.client.post(
            "/api/v1/ai/change-detection",
            files={
                "after_image": ("t1.jpg", self.valid_bytes_1, "image/jpeg"),
            },
        )
        self.assertIn(response.status_code, [400, 422])

    def test_change_detection_missing_after_image_rejected(self):
        # 6. Missing after image is rejected (HTTP 422 or 400)
        response = self.client.post(
            "/api/v1/ai/change-detection",
            files={
                "before_image": ("t0.jpg", self.valid_bytes_0, "image/jpeg"),
            },
        )
        self.assertIn(response.status_code, [400, 422])

    def test_change_detection_invalid_before_image_rejected(self):
        # 7. Invalid before image is rejected (HTTP 400)
        response = self.client.post(
            "/api/v1/ai/change-detection",
            files={
                "before_image": ("corrupt.jpg", b"not_an_image_content", "image/jpeg"),
                "after_image": ("t1.jpg", self.valid_bytes_1, "image/jpeg"),
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Could not decode image bytes", response.json()["detail"])

    def test_change_detection_invalid_after_image_rejected(self):
        # 8. Invalid after image is rejected (HTTP 400)
        response = self.client.post(
            "/api/v1/ai/change-detection",
            files={
                "before_image": ("t0.jpg", self.valid_bytes_0, "image/jpeg"),
                "after_image": ("corrupt.jpg", b"not_an_image_content", "image/jpeg"),
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Could not decode image bytes", response.json()["detail"])

    def test_change_detection_incompatible_dimensions_rejected(self):
        # 9. Incompatible image dimensions (64x64 vs 128x128) are rejected (HTTP 400)
        response = self.client.post(
            "/api/v1/ai/change-detection",
            files={
                "before_image": ("t0_64.jpg", self.valid_bytes_0, "image/jpeg"),
                "after_image": ("t1_128.jpg", self.large_bytes, "image/jpeg"),
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("dimensions mismatch", response.json()["detail"].lower())

    def test_change_detection_empty_file_rejected(self):
        # Empty byte file rejected (HTTP 400)
        response = self.client.post(
            "/api/v1/ai/change-detection",
            files={
                "before_image": ("empty.jpg", b"", "image/jpeg"),
                "after_image": ("t1.jpg", self.valid_bytes_1, "image/jpeg"),
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("empty", response.json()["detail"].lower())


class TestSatelliteAnalyzeEndpoint(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

        # Mock STAC discovery response
        self.mock_discovery = {
            "total_scenes_found": 1,
            "best_scene": {
                "scene_id": "S2A_MSIL2A_20240221T044821_N0510_R076_T45QUC_20240221T075652",
                "datetime": "2024-02-21T04:48:21Z",
                "cloud_cover_pct": 0.01,
                "platform": "Sentinel-2A",
                "has_required_bands": True,
                "bands": {
                    "B03_green": {"s3_href": "s3://eodata/...", "https_href": "https://zipper.dataspace.copernicus.eu/..."},
                    "B04_red": {"s3_href": "s3://eodata/...", "https_href": "https://zipper.dataspace.copernicus.eu/..."},
                    "B08_nir": {"s3_href": "s3://eodata/...", "https_href": "https://zipper.dataspace.copernicus.eu/..."},
                },
            },
            "scenes": [
                {
                    "scene_id": "S2A_MSIL2A_20240221T044821_N0510_R076_T45QUC_20240221T075652",
                    "datetime": "2024-02-21T04:48:21Z",
                    "cloud_cover_pct": 0.01,
                }
            ],
        }

    @staticmethod
    def _mock_band_arrays(paths):
        """Generates small 64x64 multispectral arrays for fast, deterministic, offline testing."""
        from rasterio.transform import from_origin
        import numpy as np
        h, w = 64, 64
        # Simulating vegetation and water
        green = np.full((h, w), 1200, dtype=np.float32)
        red = np.full((h, w), 500, dtype=np.float32)
        nir = np.full((h, w), 4000, dtype=np.float32)
        # Add a central water patch
        green[20:40, 20:40] = 500
        red[20:40, 20:40] = 200
        nir[20:40, 20:40] = 100
        trans = from_origin(85.8000, 20.4000, 0.0001, 0.0001)
        return green, red, nir, trans, "EPSG:4326"

    def test_satellite_route_exists(self):
        # 1. Satellite route exists
        response = self.client.post("/api/v1/satellite/analyze", json={})
        self.assertNotEqual(response.status_code, 404, "Endpoint must exist and not return 404 for empty payload")

    def test_satellite_analyze_success_and_response_contract(self):
        # 2. Valid request accepted & 3. Successful mocked satellite analysis returns HTTP 200
        from unittest.mock import patch
        with patch("src.api.routes.satellite.search_sentinel_scenes_for_aoi", return_value=self.mock_discovery), \
             patch("src.api.routes.satellite.resolve_local_scene_bands", return_value={"B03": "p3", "B04": "p4", "B08": "p8"}), \
             patch("src.api.routes.satellite.load_band_arrays", side_effect=self._mock_band_arrays):

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

            # 4. Response contains all required contract keys:
            # success, request_id, location, aoi, scene, indices, outputs, processing_time_ms
            self.assertTrue(data["success"])
            self.assertIn("request_id", data)
            self.assertIn("location", data)
            self.assertAlmostEqual(data["location"]["latitude"], 20.4625)
            self.assertAlmostEqual(data["location"]["longitude"], 85.8828)
            self.assertIn("aoi", data)
            self.assertEqual(data["aoi"]["buffer"], 0.05)
            self.assertIn("scene", data)
            self.assertEqual(data["scene"]["scene_id"], self.mock_discovery["best_scene"]["scene_id"])
            self.assertIn("indices", data)
            self.assertIn("ndvi", data["indices"])
            self.assertIn("ndwi", data["indices"])
            self.assertIn("outputs", data)
            self.assertIn("processing_time_ms", data)
            self.assertGreater(data["processing_time_ms"], 0.0)

            # 5. NDVI output URL starts with /files/satellite/
            ndvi_url = data["outputs"]["ndvi_raster_url"]
            self.assertTrue(ndvi_url.startswith("/files/satellite/"))

            # 6. NDWI output URL starts with /files/satellite/
            ndwi_url = data["outputs"]["ndwi_raster_url"]
            self.assertTrue(ndwi_url.startswith("/files/satellite/"))

            # 7. Water-mask output URL starts with /files/satellite/
            water_mask_url = data["outputs"]["water_mask_url"]
            self.assertIsNotNone(water_mask_url)
            self.assertTrue(water_mask_url.startswith("/files/satellite/"))

            # 8. No response URL contains C:\ or C:/
            for url in [ndvi_url, ndwi_url, water_mask_url]:
                self.assertNotIn("C:", url)
                self.assertNotIn("\\", url)

            # Verify generated file can actually be retrieved via static files mount
            static_resp = self.client.get(ndvi_url)
            self.assertEqual(static_resp.status_code, 200)
            self.assertEqual(static_resp.headers.get("content-type"), "image/tiff")

    def test_satellite_analyze_invalid_latitude_rejected(self):
        # 9. Invalid latitude rejected (HTTP 422)
        response = self.client.post(
            "/api/v1/satellite/analyze",
            json={"latitude": 95.0, "longitude": 85.8828},
        )
        self.assertEqual(response.status_code, 422)

    def test_satellite_analyze_invalid_longitude_rejected(self):
        # 10. Invalid longitude rejected (HTTP 422)
        response = self.client.post(
            "/api/v1/satellite/analyze",
            json={"latitude": 20.4625, "longitude": 185.0},
        )
        self.assertEqual(response.status_code, 422)

    def test_satellite_analyze_invalid_date_range_rejected(self):
        # 11. Invalid date range rejected (start_date > end_date -> HTTP 422)
        response = self.client.post(
            "/api/v1/satellite/analyze",
            json={
                "latitude": 20.4625,
                "longitude": 85.8828,
                "start_date": "2024-03-01",
                "end_date": "2024-02-01",
            },
        )
        self.assertEqual(response.status_code, 422)

    def test_satellite_analyze_invalid_max_cloud_rejected(self):
        # 12. Invalid max_cloud rejected (> 100 -> HTTP 422)
        response = self.client.post(
            "/api/v1/satellite/analyze",
            json={"latitude": 20.4625, "longitude": 85.8828, "max_cloud": 150.0},
        )
        self.assertEqual(response.status_code, 422)

    def test_satellite_analyze_no_matching_scene_cleanly_handled(self):
        # 13. No matching scene is handled cleanly (HTTP 404)
        from unittest.mock import patch
        empty_discovery = {"total_scenes_found": 0, "scenes": [], "best_scene": None}
        with patch("src.api.routes.satellite.search_sentinel_scenes_for_aoi", return_value=empty_discovery):
            response = self.client.post(
                "/api/v1/satellite/analyze",
                json={"latitude": 20.4625, "longitude": 85.8828},
            )
            self.assertEqual(response.status_code, 404)
            self.assertIn("No matching Sentinel-2 Level-2A scenes found", response.json()["detail"])

    def test_satellite_analyze_inaccessible_assets_handled_truthfully(self):
        # 14. Inaccessible satellite assets handled truthfully (HTTP 503 Service Unavailable)
        from unittest.mock import patch
        with patch("src.api.routes.satellite.search_sentinel_scenes_for_aoi", return_value=self.mock_discovery), \
             patch("src.api.routes.satellite.resolve_local_scene_bands", return_value=None):
            response = self.client.post(
                "/api/v1/satellite/analyze",
                json={"latitude": 20.4625, "longitude": 85.8828},
            )
            self.assertEqual(response.status_code, 503)
            self.assertIn("inaccessible", response.json()["detail"].lower())
            self.assertIn("authentication", response.json()["detail"].lower())


class TestSatelliteChangeEndpoint(unittest.TestCase):
    """
    Comprehensive test suite for POST /api/v1/satellite/change.
    Verifies routing, parameter validation, error handling (404, 503, 422),
    mocked temporal change pipeline execution, response contracts, and static file serving.
    """

    def setUp(self):
        self.client = TestClient(app)
        self.mock_before_discovery = {
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
        self.mock_after_discovery = {
            "total_scenes_found": 1,
            "best_scene": {
                "scene_id": "S2A_MSIL2A_20240302T044711_N0510_R076_T45QUC_20240302T090154",
                "datetime": "2024-03-02T04:47:11Z",
                "cloud_cover_pct": 0.05,
            },
            "scenes": [
                {
                    "scene_id": "S2A_MSIL2A_20240302T044711_N0510_R076_T45QUC_20240302T090154",
                    "datetime": "2024-03-02T04:47:11Z",
                    "cloud_cover_pct": 0.05,
                }
            ],
        }

    @staticmethod
    def _mock_band_arrays(paths):
        """Generates small 64x64 multispectral arrays for fast, deterministic, offline testing."""
        from rasterio.transform import from_origin
        import numpy as np
        h, w = 64, 64
        # Simulating vegetation and water
        green = np.full((h, w), 1200, dtype=np.float32)
        red = np.full((h, w), 500, dtype=np.float32)
        nir = np.full((h, w), 4000, dtype=np.float32)
        # Add a central water patch
        green[20:40, 20:40] = 500
        red[20:40, 20:40] = 200
        nir[20:40, 20:40] = 100
        trans = from_origin(85.8000, 20.4000, 0.0001, 0.0001)
        return green, red, nir, trans, "EPSG:4326"

    def test_satellite_change_route_exists(self):
        # 1. Route exists
        response = self.client.post("/api/v1/satellite/change", json={})
        self.assertNotEqual(response.status_code, 404, "Endpoint must exist and not return 404 for empty payload")

    def test_satellite_change_method_not_allowed(self):
        # 2. Correct POST method: GET should return 405 Method Not Allowed
        response = self.client.get("/api/v1/satellite/change")
        self.assertEqual(response.status_code, 405)

    def test_satellite_change_invalid_latitude_rejected(self):
        # 4. Invalid latitude rejected (HTTP 422)
        response = self.client.post(
            "/api/v1/satellite/change",
            json={"latitude": 95.0, "longitude": 85.8828},
        )
        self.assertEqual(response.status_code, 422)

    def test_satellite_change_invalid_longitude_rejected(self):
        # 5. Invalid longitude rejected (HTTP 422)
        response = self.client.post(
            "/api/v1/satellite/change",
            json={"latitude": 20.4625, "longitude": 190.0},
        )
        self.assertEqual(response.status_code, 422)

    def test_satellite_change_invalid_buffer_rejected(self):
        # 6. Invalid buffer rejected (HTTP 422)
        response = self.client.post(
            "/api/v1/satellite/change",
            json={"latitude": 20.4625, "longitude": 85.8828, "buffer": -0.05},
        )
        self.assertEqual(response.status_code, 422)

        response_large = self.client.post(
            "/api/v1/satellite/change",
            json={"latitude": 20.4625, "longitude": 85.8828, "buffer": 10.0},
        )
        self.assertEqual(response_large.status_code, 422)

    def test_satellite_change_equal_dates_rejected(self):
        # 7. before_date == after_date rejected (HTTP 422)
        response = self.client.post(
            "/api/v1/satellite/change",
            json={
                "latitude": 20.4625,
                "longitude": 85.8828,
                "before_date": "2024-02-21",
                "after_date": "2024-02-21",
            },
        )
        self.assertEqual(response.status_code, 422)

    def test_satellite_change_inverted_dates_rejected(self):
        # 8. before_date > after_date rejected (HTTP 422)
        response = self.client.post(
            "/api/v1/satellite/change",
            json={
                "latitude": 20.4625,
                "longitude": 85.8828,
                "before_date": "2024-03-02",
                "after_date": "2024-02-21",
            },
        )
        self.assertEqual(response.status_code, 422)

    def test_satellite_change_missing_latitude_rejected(self):
        # 9. Missing latitude rejected (HTTP 422)
        response = self.client.post(
            "/api/v1/satellite/change",
            json={"longitude": 85.8828},
        )
        self.assertEqual(response.status_code, 422)

    def test_satellite_change_missing_longitude_rejected(self):
        # 10. Missing longitude rejected (HTTP 422)
        response = self.client.post(
            "/api/v1/satellite/change",
            json={"latitude": 20.4625},
        )
        self.assertEqual(response.status_code, 422)

    def test_satellite_change_no_before_scene_returns_404(self):
        # 11. No before scene -> HTTP 404
        from unittest.mock import patch
        empty_discovery = {"total_scenes_found": 0, "scenes": [], "best_scene": None}
        with patch("src.api.routes.satellite.search_sentinel_scenes_for_aoi", return_value=empty_discovery):
            response = self.client.post(
                "/api/v1/satellite/change",
                json={"latitude": 20.4625, "longitude": 85.8828},
            )
            self.assertEqual(response.status_code, 404)
            self.assertIn("No suitable before Sentinel-2", response.json()["detail"])

    def test_satellite_change_no_after_scene_returns_404(self):
        # 12. No after scene -> HTTP 404
        from unittest.mock import patch
        empty_discovery = {"total_scenes_found": 0, "scenes": [], "best_scene": None}
        with patch("src.api.routes.satellite.search_sentinel_scenes_for_aoi", side_effect=[self.mock_before_discovery, empty_discovery]):
            response = self.client.post(
                "/api/v1/satellite/change",
                json={"latitude": 20.4625, "longitude": 85.8828},
            )
            self.assertEqual(response.status_code, 404)
            self.assertIn("No suitable after Sentinel-2", response.json()["detail"])

    def test_satellite_change_inaccessible_remote_assets_returns_503(self):
        # 13. Inaccessible remote assets -> HTTP 503
        from unittest.mock import patch
        with patch("src.api.routes.satellite.search_sentinel_scenes_for_aoi", side_effect=[self.mock_before_discovery, self.mock_after_discovery]), \
             patch("src.api.routes.satellite.resolve_local_scene_bands", return_value=None):
            response = self.client.post(
                "/api/v1/satellite/change",
                json={"latitude": 20.4625, "longitude": 85.8828},
            )
            self.assertEqual(response.status_code, 503)
            self.assertIn("inaccessible", response.json()["detail"].lower())
            self.assertIn("authentication", response.json()["detail"].lower())

    def test_satellite_change_successful_mocked_contract_and_static_serving(self):
        # 3. Valid request contract, 14. Successful mocked temporal analysis, 15-22. Response verification & static files
        from unittest.mock import patch
        with patch("src.api.routes.satellite.search_sentinel_scenes_for_aoi", side_effect=[self.mock_before_discovery, self.mock_after_discovery]), \
             patch("src.api.routes.satellite.resolve_local_scene_bands", return_value={"B03": "p3", "B04": "p4", "B08": "p8"}), \
             patch("src.api.routes.satellite.load_band_arrays", side_effect=self._mock_band_arrays):

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

            # 15. Successful response contains request_id
            self.assertTrue(data["success"])
            self.assertIn("request_id", data)
            self.assertEqual(data["status"], "completed")

            # Contains location and aoi
            self.assertIn("location", data)
            self.assertAlmostEqual(data["location"]["latitude"], 20.4625)
            self.assertAlmostEqual(data["location"]["longitude"], 85.8828)
            self.assertIn("aoi", data)
            self.assertEqual(data["aoi"]["buffer"], 0.05)

            # 16. Successful response contains actual before/after scene IDs
            self.assertIn("before", data)
            self.assertIn("after", data)
            self.assertEqual(data["before"]["scene_id"], self.mock_before_discovery["best_scene"]["scene_id"])
            self.assertEqual(data["after"]["scene_id"], self.mock_after_discovery["best_scene"]["scene_id"])

            # 17. Successful response contains actual dates
            self.assertEqual(data["before"]["date"], "2024-02-21")
            self.assertEqual(data["after"]["date"], "2024-03-02")

            # 18. Successful response contains NDVI change statistics
            self.assertIn("ndvi_change", data)
            self.assertIn("gain_percent", data["ndvi_change"])
            self.assertIn("loss_percent", data["ndvi_change"])
            self.assertIn("stable_percent", data["ndvi_change"])
            self.assertGreaterEqual(data["ndvi_change"]["gain_percent"], 0.0)
            self.assertLessEqual(data["ndvi_change"]["gain_percent"], 100.0)

            # 19. Successful response contains NDWI change statistics
            self.assertIn("ndwi_change", data)
            self.assertIn("water_gain_percent", data["ndwi_change"])
            self.assertIn("water_loss_percent", data["ndwi_change"])
            self.assertIn("stable_percent", data["ndwi_change"])
            self.assertGreaterEqual(data["ndwi_change"]["water_gain_percent"], 0.0)
            self.assertLessEqual(data["ndwi_change"]["water_gain_percent"], 100.0)

            # 20. Output URLs start with /files/satellite/
            self.assertIn("outputs", data)
            ndvi_change_url = data["outputs"]["ndvi_change_raster_url"]
            ndwi_change_url = data["outputs"]["ndwi_change_raster_url"]
            water_change_url = data["outputs"]["water_change_raster_url"]

            self.assertTrue(ndvi_change_url.startswith("/files/satellite/"))
            self.assertTrue(ndwi_change_url.startswith("/files/satellite/"))
            self.assertTrue(water_change_url.startswith("/files/satellite/"))

            # 21. Output URLs do not expose filesystem paths (no C:\, C:/, file:///)
            for url in [ndvi_change_url, ndwi_change_url, water_change_url]:
                self.assertNotIn("C:", url)
                self.assertNotIn("\\", url)
                self.assertNotIn("file:", url)

            # 22. Static output files can be served
            static_ndvi = self.client.get(ndvi_change_url)
            self.assertEqual(static_ndvi.status_code, 200)
            self.assertEqual(static_ndvi.headers.get("content-type"), "image/tiff")

            static_ndwi = self.client.get(ndwi_change_url)
            self.assertEqual(static_ndwi.status_code, 200)

            static_water = self.client.get(water_change_url)
            self.assertEqual(static_water.status_code, 200)

            # Execution timing
            self.assertIn("processing_time_ms", data)
            self.assertGreater(data["processing_time_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
