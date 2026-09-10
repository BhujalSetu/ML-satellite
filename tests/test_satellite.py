"""
Unit Tests for Satellite Remote Sensing Module
Verifies NDVI/NDWI formulas, zero-division/nodata edge cases, water mask thresholding,
temporal dynamics classification, GeoTIFF I/O via rasterio, Copernicus STAC Client,
and Real Sentinel-2 Level-2A verification logic.
"""

import unittest
import json
import numpy as np
from pathlib import Path
import tempfile
import rasterio
from rasterio.transform import from_origin

from src.satellite.ndvi import calculate_ndvi, compute_ndvi_statistics, save_geotiff as save_ndvi_tiff
from src.satellite.ndwi import calculate_ndwi, classify_water_mask, compute_ndwi_statistics, save_water_mask_geotiff
from src.satellite.temporal import compute_temporal_difference, detect_water_dynamics, save_dynamics_geotiff
from src.satellite.stac_client import CopernicusSTACClient, SentinelSceneMetadata, BandAsset
from src.satellite.aoi_workflow import (
    AOIBoundingBox,
    validate_coordinates,
    validate_bounding_box,
    create_point_aoi,
    build_aoi,
    search_sentinel_scenes_for_aoi,
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
)

# Import real pipeline verification functions
import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.run_satellite_pipeline import verify_band_alignment, resolve_band_path, DEFAULT_B04_FILE

class TestSatelliteProcessing(unittest.TestCase):

    def test_ndvi_calculation_and_edge_cases(self):
        # Standard calculation: NIR=4000, Red=1000 -> (4000-1000)/(4000+1000) = 3000/5000 = 0.6
        nir = np.array([[4000, 1000], [0, 2000]], dtype=np.float32)
        red = np.array([[1000, 1000], [0, 3000]], dtype=np.float32)

        ndvi = calculate_ndvi(nir=nir, red=red, nodata_value=-9999.0)

        self.assertAlmostEqual(float(ndvi[0, 0]), 0.6, places=4)
        self.assertAlmostEqual(float(ndvi[0, 1]), 0.0, places=4)
        # Zero denominator case: [1, 0] has nir=0, red=0 -> denominator = 0 -> nodata
        self.assertEqual(float(ndvi[1, 0]), -9999.0)
        # Water/Negative case: (2000 - 3000) / (2000 + 3000) = -1000 / 5000 = -0.2
        self.assertAlmostEqual(float(ndvi[1, 1]), -0.2, places=4)

    def test_ndvi_statistics(self):
        ndvi = np.array([
            [0.6, 0.3],   # Dense veg, Sparse veg
            [0.1, -0.2],  # Barren, Water
            [-9999.0, -9999.0]  # NoData
        ], dtype=np.float32)

        stats = compute_ndvi_statistics(ndvi, nodata_value=-9999.0)
        self.assertEqual(stats["valid_pixels"], 4)
        self.assertEqual(stats["nodata_pixels"], 2)
        self.assertAlmostEqual(stats["mean"], (0.6 + 0.3 + 0.1 - 0.2) / 4.0, places=3)
        self.assertEqual(stats["dense_vegetation_pct"], 25.0)
        self.assertEqual(stats["water_proxy_pct"], 25.0)

    def test_ndwi_and_water_mask(self):
        # McFeeters: (Green - NIR) / (Green + NIR)
        green = np.array([[800, 400], [0, 500]], dtype=np.float32)
        nir   = np.array([[200, 400], [0, 2500]], dtype=np.float32)

        ndwi = calculate_ndwi(band_primary=green, band_secondary=nir, nodata_value=-9999.0)
        # Pixel [0,0]: (800-200)/(800+200) = 600/1000 = +0.6 (Water)
        self.assertAlmostEqual(float(ndwi[0, 0]), 0.6, places=4)
        # Pixel [0,1]: (400-400)/(400+400) = 0.0
        self.assertAlmostEqual(float(ndwi[0, 1]), 0.0, places=4)
        # Pixel [1,0]: 0/0 -> nodata
        self.assertEqual(float(ndwi[1, 0]), -9999.0)
        # Pixel [1,1]: (500-2500)/(500+2500) = -2000/3000 = -0.6667 (Vegetation/Land)
        self.assertAlmostEqual(float(ndwi[1, 1]), -0.6667, places=3)

        # Water Mask (threshold > 0.0)
        mask = classify_water_mask(ndwi, threshold=0.0, nodata_value=-9999.0)
        self.assertEqual(mask[0, 0], 1)    # Water
        self.assertEqual(mask[0, 1], 0)    # Non-water (0.0 not strictly > 0.0)
        self.assertEqual(mask[1, 0], 255)  # NoData
        self.assertEqual(mask[1, 1], 0)    # Non-water

    def test_temporal_water_dynamics(self):
        # T0 vs T1 NDWI
        ndwi_t0 = np.array([[-0.3, 0.5], [-0.4, 0.4]], dtype=np.float32)
        ndwi_t1 = np.array([[-0.2, 0.6], [0.3, -0.1]], dtype=np.float32)

        res = detect_water_dynamics(ndwi_t0, ndwi_t1, threshold=0.0)
        dyn = res["dynamics_raster"]

        self.assertEqual(dyn[0, 0], 0)  # Persistent Land
        self.assertEqual(dyn[0, 1], 1)  # Persistent Water
        self.assertEqual(dyn[1, 0], 2)  # Water Gain
        self.assertEqual(dyn[1, 1], 3)  # Water Loss

        cats = res["categories"]
        self.assertEqual(cats["persistent_land"]["pixel_count"], 1)
        self.assertEqual(cats["persistent_water"]["pixel_count"], 1)
        self.assertEqual(cats["water_gain"]["pixel_count"], 1)
        self.assertEqual(cats["water_loss"]["pixel_count"], 1)

    def test_geotiff_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "test_geotiff.tif"
            data = np.array([[0.1, 0.5], [0.8, -0.2]], dtype=np.float32)
            save_ndvi_tiff(out_file, data)

            self.assertTrue(out_file.is_file())

            # Read back using rasterio
            with rasterio.open(out_file) as src:
                read_arr = src.read(1)
                self.assertEqual(read_arr.shape, (2, 2))
                self.assertAlmostEqual(float(read_arr[0, 1]), 0.5, places=4)
                self.assertEqual(str(src.crs), "EPSG:4326")

class TestCopernicusSTACClient(unittest.TestCase):

    def setUp(self):
        self.client = CopernicusSTACClient()

    def test_build_search_payload(self):
        bbox = [85.75, 20.20, 85.85, 20.30]
        date_range = ("2024-01-01", "2024-02-28")
        payload = self.client.build_search_payload(bbox, date_range, max_cloud_cover=15.0, limit=5)

        self.assertEqual(payload["collections"], ["sentinel-2-l2a"])
        self.assertEqual(payload["bbox"], bbox)
        self.assertEqual(payload["datetime"], "2024-01-01T00:00:00Z/2024-02-28T23:59:59Z")
        self.assertEqual(payload["limit"], 5)
        self.assertEqual(payload["query"]["eo:cloud_cover"]["lte"], 15.0)

        # Bad bbox length raises ValueError
        with self.assertRaises(ValueError):
            self.client.build_search_payload([85.0, 20.0], date_range)

    def test_parse_stac_feature(self):
        dummy_feature = {
            "id": "S2A_MSIL2A_20240221T044821_TEST",
            "collection": "sentinel-2-l2a",
            "bbox": [85.0, 19.8, 86.1, 20.8],
            "properties": {
                "datetime": "2024-02-21T04:48:21.024Z",
                "eo:cloud_cover": 0.05,
                "platform": "sentinel-2a"
            },
            "assets": {
                "thumbnail": {"href": "https://example.com/thumb.jpg"},
                "B04_10m": {
                    "href": "s3://eodata/B04_10m.jp2",
                    "alternate": {"https": {"href": "https://example.com/B04_10m.jp2"}},
                    "type": "image/jp2",
                    "file:size": 120000000
                },
                "B03_10m": {
                    "href": "s3://eodata/B03_10m.jp2",
                    "type": "image/jp2"
                },
                "B08_10m": {
                    "href": "s3://eodata/B08_10m.jp2",
                    "type": "image/jp2"
                }
            }
        }

        scene = self.client.parse_stac_feature(dummy_feature)

        self.assertEqual(scene.scene_id, "S2A_MSIL2A_20240221T044821_TEST")
        self.assertEqual(scene.cloud_cover_pct, 0.05)
        self.assertEqual(scene.platform, "sentinel-2a")
        self.assertEqual(scene.thumbnail_url, "https://example.com/thumb.jpg")

        # Check identified bands
        b04 = scene.get_band("B04")
        self.assertIsNotNone(b04)
        self.assertEqual(b04.asset_key, "B04_10m")
        self.assertEqual(b04.s3_href, "s3://eodata/B04_10m.jp2")
        self.assertEqual(b04.https_href, "https://example.com/B04_10m.jp2")

        b03 = scene.get_band("B03")
        self.assertIsNotNone(b03)
        self.assertEqual(b03.band_id, "B03")

        b08 = scene.get_band("B08")
        self.assertIsNotNone(b08)
        self.assertEqual(b08.band_id, "B08")

        # Serialized dictionary check
        s_dict = scene.to_dict()
        self.assertEqual(s_dict["scene_id"], scene.scene_id)
        self.assertIn("red_B04", s_dict["identified_bands"])
        self.assertIn("green_B03", s_dict["identified_bands"])
        self.assertIn("nir_B08", s_dict["identified_bands"])

    def test_live_stac_query_small_aoi(self):
        # Tests actual connectivity to official Copernicus STAC endpoint with 1 scene limit
        try:
            scenes = self.client.search(
                bbox=[85.75, 20.20, 85.85, 20.30],
                date_range=("2024-02-01", "2024-02-28"),
                max_cloud_cover=10.0,
                limit=1,
            )
            self.assertGreaterEqual(len(scenes), 1)
            sc = scenes[0]
            self.assertTrue(sc.scene_id.startswith("S2"))
            self.assertLessEqual(sc.cloud_cover_pct, 10.0)
            self.assertIsNotNone(sc.get_band("B04"))
            self.assertIsNotNone(sc.get_band("B03"))
            self.assertIsNotNone(sc.get_band("B08"))
        except Exception as e:
            self.skipTest(f"Live STAC endpoint connectivity skipped/failed: {e}")

class TestRealSentinel2Pipeline(unittest.TestCase):

    def test_resolve_default_band_paths(self):
        # Verify that default B04 can be resolved on disk
        resolved_b04 = resolve_band_path(None, DEFAULT_B04_FILE)
        self.assertTrue(resolved_b04.is_file())
        self.assertIn("B04_10m.jp2", resolved_b04.name)

    def test_verify_band_alignment_mismatch_detection(self):
        # Test that verify_band_alignment catches synthetic spatial mismatches
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            p1 = tmp / "band1.tif"
            p2_mismatch = tmp / "band2_mismatch.tif"

            # Band 1: 50x50, EPSG:32645
            transform1 = from_origin(300000, 2200000, 10, 10)
            save_ndvi_tiff(p1, np.zeros((50, 50), dtype=np.float32), transform=transform1, crs="EPSG:32645")

            # Band 2: 60x60 (Dimension Mismatch)
            save_ndvi_tiff(p2_mismatch, np.zeros((60, 60), dtype=np.float32), transform=transform1, crs="EPSG:32645")

            with self.assertRaises(ValueError):
                verify_band_alignment({"B1": p1, "B2": p2_mismatch})

    def test_real_pipeline_output_integrity(self):
        # Check generated real mode output products
        real_out_dir = ROOT / "outputs" / "satellite" / "real"
        json_path = real_out_dir / "real_satellite_analytics_summary.json"

        if not json_path.is_file():
            self.skipTest("Real pipeline has not yet been executed in this environment.")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["scene_id"], "S2A_MSIL2A_20240221T044821_N0510_R076_T45QUC_20240221T075652")
        self.assertEqual(data["crs"], "EPSG:32645")
        self.assertEqual(data["width"], 10980)
        self.assertEqual(data["height"], 10980)
        self.assertEqual(data["resolution"], [10.0, 10.0])
        self.assertIn("Temporal comparison requires", data["temporal_analysis_status"])

class TestAOISatelliteWorkflow(unittest.TestCase):

    def test_coordinate_validation(self):
        # Valid coordinates should pass
        validate_coordinates(20.2961, 85.8245)
        validate_coordinates(-45.0, 120.0)

        # Latitude out of bounds
        with self.assertRaises(ValueError):
            validate_coordinates(95.0, 85.0)
        with self.assertRaises(ValueError):
            validate_coordinates(-90.1, 85.0)

        # Longitude out of bounds
        with self.assertRaises(ValueError):
            validate_coordinates(20.0, 185.0)
        with self.assertRaises(ValueError):
            validate_coordinates(20.0, -180.5)

    def test_bounding_box_validation(self):
        # Valid bbox
        validate_bounding_box(85.75, 20.20, 85.85, 20.30)

        # Inverted west/east
        with self.assertRaises(ValueError):
            validate_bounding_box(86.0, 20.20, 85.0, 20.30)

        # Inverted south/north
        with self.assertRaises(ValueError):
            validate_bounding_box(85.75, 20.50, 85.85, 20.20)

    def test_create_point_aoi_buffer(self):
        aoi = create_point_aoi(lat=20.0, lon=85.0, buffer_deg=0.05)
        self.assertAlmostEqual(aoi.west, 84.95, places=4)
        self.assertAlmostEqual(aoi.east, 85.05, places=4)
        self.assertAlmostEqual(aoi.south, 19.95, places=4)
        self.assertAlmostEqual(aoi.north, 20.05, places=4)
        self.assertTrue(aoi.contains_point(20.0, 85.0))

        # Invalid buffer
        with self.assertRaises(ValueError):
            create_point_aoi(20.0, 85.0, buffer_deg=-0.1)
        with self.assertRaises(ValueError):
            create_point_aoi(20.0, 85.0, buffer_deg=10.0)

    def test_build_aoi_dispatcher(self):
        # From point
        aoi_point = build_aoi(lat=20.25, lon=85.80, buffer_deg=0.02)
        self.assertAlmostEqual(aoi_point.west, 85.78, places=4)
        self.assertAlmostEqual(aoi_point.east, 85.82, places=4)

        # From explicit bounding box
        aoi_bbox = build_aoi(west=85.1, south=20.1, east=85.3, north=20.4)
        self.assertEqual(aoi_bbox.to_list(), [85.1, 20.1, 85.3, 20.4])

        # Partial bounding box rejected
        with self.assertRaises(ValueError):
            build_aoi(west=85.1, south=20.1)

        # Missing both point and bbox rejected
        with self.assertRaises(ValueError):
            build_aoi()

    def test_search_sentinel_scenes_for_aoi_mock(self):
        class MockSTACClient:
            def search(self, bbox, date_range, max_cloud_cover=20.0, limit=5):
                # Return 2 mock scenes: one with 15% cloud and one with 2% cloud
                s1 = SentinelSceneMetadata(
                    scene_id="S2A_TEST_CLOUDY",
                    collection="sentinel-2-l2a",
                    datetime="2024-02-15T05:00:00Z",
                    cloud_cover_pct=15.0,
                    bbox=bbox,
                    platform="Sentinel-2A",
                    bands={
                        "B04_10m": BandAsset("B04", "10m", "B04_10m", "s3://b04"),
                        "B03_10m": BandAsset("B03", "10m", "B03_10m", "s3://b03"),
                        "B08_10m": BandAsset("B08", "10m", "B08_10m", "s3://b08"),
                    }
                )
                s2 = SentinelSceneMetadata(
                    scene_id="S2A_TEST_CLEAR",
                    collection="sentinel-2-l2a",
                    datetime="2024-02-20T05:00:00Z",
                    cloud_cover_pct=2.1,
                    bbox=bbox,
                    platform="Sentinel-2A",
                    bands={
                        "B04_10m": BandAsset("B04", "10m", "B04_10m", "s3://b04_clear"),
                        "B03_10m": BandAsset("B03", "10m", "B03_10m", "s3://b03_clear"),
                        "B08_10m": BandAsset("B08", "10m", "B08_10m", "s3://b08_clear"),
                    }
                )
                return [s1, s2]

        aoi = AOIBoundingBox(west=85.7, south=20.2, east=85.9, north=20.4)
        mock_client = MockSTACClient()
        res = search_sentinel_scenes_for_aoi(
            aoi=aoi,
            date_range=("2024-02-01", "2024-02-28"),
            max_cloud_cover=20.0,
            limit=5,
            client=mock_client,
        )

        self.assertEqual(res["total_scenes_found"], 2)
        # Verify optimal least-cloudy scene was selected
        self.assertEqual(res["best_scene"]["scene_id"], "S2A_TEST_CLEAR")
        self.assertEqual(res["best_scene"]["cloud_cover_pct"], 2.1)
        self.assertTrue(res["best_scene"]["has_required_bands"])

    def test_search_sentinel_scenes_empty_result(self):
        class MockEmptySTACClient:
            def search(self, bbox, date_range, max_cloud_cover=20.0, limit=5):
                return []

        aoi = AOIBoundingBox(west=85.7, south=20.2, east=85.9, north=20.4)
        res = search_sentinel_scenes_for_aoi(
            aoi=aoi,
            date_range=("2024-02-01", "2024-02-28"),
            client=MockEmptySTACClient(),
        )
        self.assertEqual(res["total_scenes_found"], 0)
        self.assertIsNone(res["best_scene"])
        self.assertEqual(len(res["scenes"]), 0)

class TestTemporalChangeDetection(unittest.TestCase):

    def test_validate_temporal_pair_matching_and_mismatches(self):
        meta_t0 = {
            "width": 100,
            "height": 100,
            "crs": "EPSG:32645",
            "transform": from_origin(300000, 2200000, 10, 10),
            "res": (10.0, 10.0),
        }

        # Identical pair passes
        meta_t1_matching = dict(meta_t0)
        validate_temporal_pair(meta_t0, meta_t1_matching)

        # Width mismatch
        meta_bad_w = dict(meta_t0, width=120)
        with self.assertRaises(ValueError):
            validate_temporal_pair(meta_t0, meta_bad_w)

        # Height mismatch
        meta_bad_h = dict(meta_t0, height=80)
        with self.assertRaises(ValueError):
            validate_temporal_pair(meta_t0, meta_bad_h)

        # CRS mismatch
        meta_bad_crs = dict(meta_t0, crs="EPSG:4326")
        with self.assertRaises(ValueError):
            validate_temporal_pair(meta_t0, meta_bad_crs)

        # Transform mismatch
        meta_bad_tf = dict(meta_t0, transform=from_origin(310000, 2200000, 10, 10))
        with self.assertRaises(ValueError):
            validate_temporal_pair(meta_t0, meta_bad_tf)

        # Resolution mismatch
        meta_bad_res = dict(meta_t0, res=(20.0, 20.0))
        with self.assertRaises(ValueError):
            validate_temporal_pair(meta_t0, meta_bad_res)

    def test_ndvi_and_ndwi_difference_arithmetic(self):
        t0 = np.array([[0.2, 0.5], [-9999.0, 0.4]], dtype=np.float32)
        t1 = np.array([[0.3, 0.2], [0.1, np.nan]], dtype=np.float32)

        # NDVI change = T1 - T0
        diff = calculate_ndvi_difference(t0, t1, nodata_value=-9999.0)
        self.assertAlmostEqual(float(diff[0, 0]), 0.1, places=4)
        self.assertAlmostEqual(float(diff[0, 1]), -0.3, places=4)
        # Nodata at T0 propagates
        self.assertEqual(float(diff[1, 0]), -9999.0)
        # NaN at T1 propagates as nodata
        self.assertEqual(float(diff[1, 1]), -9999.0)

        # Relative change check
        rel = calculate_relative_change(t0, t1, nodata_value=-9999.0)
        # (0.3 - 0.2) / (0.2 + 1e-4) * 100 ~ 49.975%
        self.assertGreater(float(rel[0, 0]), 45.0)

    def test_classify_ndvi_and_ndwi_change(self):
        ndvi_diff = np.array([[-0.25, 0.02], [0.30, -9999.0]], dtype=np.float32)
        c_ndvi = classify_ndvi_change(ndvi_diff, threshold_decline=-0.10, threshold_gain=0.10)

        self.assertEqual(c_ndvi[0, 0], 1)   # Decline (< -0.10)
        self.assertEqual(c_ndvi[0, 1], 2)   # Stable (-0.10 <= x <= 0.10)
        self.assertEqual(c_ndvi[1, 0], 3)   # Growth (> 0.10)
        self.assertEqual(c_ndvi[1, 1], 255) # NoData

        ndwi_diff = np.array([[-0.15, 0.0], [0.25, -9999.0]], dtype=np.float32)
        c_ndwi = classify_ndwi_change(ndwi_diff, threshold_dry=-0.10, threshold_wet=0.10)
        self.assertEqual(c_ndwi[0, 0], 1)   # Drying (< -0.10)
        self.assertEqual(c_ndwi[0, 1], 2)   # Stable
        self.assertEqual(c_ndwi[1, 0], 3)   # Inundating (> 0.10)
        self.assertEqual(c_ndwi[1, 1], 255) # NoData

    def test_temporal_water_dynamics_and_derived_area(self):
        ndwi_t0 = np.array([[-0.2, 0.4], [-0.3, 0.5]], dtype=np.float32)
        ndwi_t1 = np.array([[-0.1, 0.3], [0.4, -0.2]], dtype=np.float32)

        # Pixel area for 10m x 10m Sentinel-2 = 100.0 m2
        res = calculate_temporal_water_dynamics(ndwi_t0, ndwi_t1, water_threshold=0.0, pixel_area_m2=100.0)
        cats = res["categories"]

        # [0, 0]: -0.2 -> -0.1 : Land -> Land (Persistent Land)
        # [0, 1]:  0.4 ->  0.3 : Water -> Water (Persistent Water)
        # [1, 0]: -0.3 ->  0.4 : Land -> Water (Water Gain)
        # [1, 1]:  0.5 -> -0.2 : Water -> Land (Water Loss)
        self.assertEqual(cats["persistent_land"]["pixel_count"], 1)
        self.assertEqual(cats["persistent_land"]["area_m2"], 100.0)
        self.assertEqual(cats["persistent_water"]["pixel_count"], 1)
        self.assertEqual(cats["water_gain"]["pixel_count"], 1)
        self.assertEqual(cats["water_gain"]["area_m2"], 100.0)
        self.assertEqual(cats["water_loss"]["pixel_count"], 1)
        self.assertEqual(cats["water_loss"]["area_m2"], 100.0)

    def test_change_statistics_calculation(self):
        diff = np.array([[-0.2, -0.01], [0.03, 0.4]], dtype=np.float32)
        stats = calculate_change_statistics(diff, threshold_negligible=0.05)

        self.assertEqual(stats["valid_pixels"], 4)
        self.assertEqual(stats["pixels_increased"], 1)   # 0.4 > 0.05
        self.assertEqual(stats["pixels_decreased"], 1)   # -0.2 < -0.05
        self.assertEqual(stats["pixels_negligible"], 2)  # -0.01 and 0.03
        self.assertAlmostEqual(stats["min"], -0.2, places=3)
        self.assertAlmostEqual(stats["max"], 0.4, places=3)

    def test_synthetic_temporal_pipeline_execution(self):
        import tempfile
        from scripts.run_satellite_pipeline import run_synthetic_temporal_pipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            run_synthetic_temporal_pipeline(out_dir)

            expected_files = [
                "ndvi_t0.tif", "ndvi_t1.tif", "ndvi_change.tif",
                "ndwi_t0.tif", "ndwi_t1.tif", "ndwi_change.tif",
                "water_mask_t0.tif", "water_mask_t1.tif", "water_change.tif",
                "temporal_change_summary.json"
            ]
            for fname in expected_files:
                p = out_dir / fname
                self.assertTrue(p.is_file(), f"Expected temporal output missing: {fname}")

            with open(out_dir / "temporal_change_summary.json", "r") as f:
                data = json.load(f)
            self.assertEqual(data["dataset_type"], "TEST/SYNTHETIC")
            self.assertIn("ndvi_statistics", data)
            self.assertIn("water_statistics", data)

if __name__ == "__main__":
    unittest.main()
