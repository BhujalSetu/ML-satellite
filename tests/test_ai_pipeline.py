"""
Unit Tests for WATERSCOPE-AI AI Pipeline
Verifies device selection, schemas, image preprocessing, YOLO inference, and visualizations.
"""

import unittest
import numpy as np
from pathlib import Path
import tempfile
import cv2

from src.ai.config import AIConfig, FPCD_OBJECT_CLASSES, FPCD_CHANGE_CLASSES
from src.ai.schemas import (
    BoundingBox,
    DetectionResult,
    InferenceMetadata,
    InferenceOutput,
    ChangeCategoryStat,
)
from src.ai.preprocessing import (
    load_image,
    validate_image,
    resize_image_aspect_ratio,
    normalize_image,
)
from src.ai.inference import WatershedYOLO
from src.ai.visualize import (
    draw_detections,
    draw_change_mask,
    create_temporal_comparison_panel,
)

class TestAIPipeline(unittest.TestCase):

    def test_config_device_resolution(self):
        # Explicit CPU
        cfg_cpu = AIConfig(device="cpu")
        self.assertEqual(cfg_cpu.resolve_device(), "cpu")
        self.assertEqual(cfg_cpu.get_device_name(), "CPU")

        # Auto selection
        cfg_auto = AIConfig(device="auto")
        resolved = cfg_auto.resolve_device()
        self.assertIn(resolved, ["cpu", "cuda"])

    def test_schemas_to_dict(self):
        bbox = BoundingBox(x1=10.0, y1=20.0, x2=110.0, y2=120.0)
        self.assertEqual(bbox.width, 100.0)
        self.assertEqual(bbox.height, 100.0)
        self.assertEqual(bbox.area, 10000.0)

        det = DetectionResult(class_id=1, class_name="Wet Farm Pond - Unlined", confidence=0.88, bbox=bbox)
        d_dict = det.to_dict()
        self.assertEqual(d_dict["class_id"], 1)
        self.assertEqual(d_dict["class_name"], "Wet Farm Pond - Unlined")
        self.assertIn("bbox", d_dict)

        meta = InferenceMetadata(
            model_name="yolo26n.pt",
            model_domain="generic_coco",
            is_pretrained_baseline=True,
            device="cpu",
            image_path="test.jpg",
            image_shape=(768, 1024, 3),
            preprocess_time_ms=10.0,
            inference_time_ms=50.0,
            postprocess_time_ms=5.0,
            total_time_ms=65.0,
        )
        out = InferenceOutput(metadata=meta, detections=[det], num_detections=1)
        out_dict = out.to_dict()
        self.assertEqual(out_dict["num_detections"], 1)
        self.assertEqual(out_dict["metadata"]["model_name"], "yolo26n.pt")
        self.assertTrue(out_dict["metadata"]["is_pretrained_baseline"])

    def test_preprocessing(self):
        # 1. Validation
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        self.assertTrue(validate_image(img))

        with self.assertRaises(ValueError):
            validate_image(np.array([]))

        # 2. Resizing with aspect ratio preservation
        padded, scale, (pad_w, pad_h) = resize_image_aspect_ratio(img, target_size=640)
        self.assertEqual(padded.shape, (640, 640, 3))
        self.assertGreater(scale, 0)

        # 3. Normalization
        norm = normalize_image(img)
        self.assertEqual(norm.dtype, np.float32)
        self.assertAlmostEqual(norm.max(), 128.0 / 255.0, places=4)

    def test_yolo_inference(self):
        model_path = Path(__file__).resolve().parent.parent / "yolo26n.pt"
        if not model_path.is_file():
            self.skipTest("yolo26n.pt not found on disk")

        config = AIConfig(model_path=str(model_path), device="cpu", confidence_threshold=0.2)
        engine = WatershedYOLO(config=config)
        self.assertEqual(engine.device, "cpu")

        # Test inference on a generated dummy image
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        output = engine.predict(dummy_img, save_annotated=False)

        self.assertIsInstance(output, InferenceOutput)
        self.assertEqual(output.metadata.device, "cpu")
        self.assertTrue(output.metadata.is_pretrained_baseline)
        self.assertIsNotNone(output.metadata.warning)

    def test_visualizations(self):
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        det = DetectionResult(
            class_id=0,
            class_name="Wet Farm Pond - Lined",
            confidence=0.92,
            bbox=BoundingBox(50, 50, 150, 150),
        )
        drawn = draw_detections(img, [det])
        self.assertEqual(drawn.shape, img.shape)

        # Mask visualization
        mask = np.zeros((400, 400), dtype=np.uint8)
        mask[100:200, 100:200] = 1  # Constructed
        colored = draw_change_mask(mask, bg_image=img)
        self.assertEqual(colored.shape, (400, 400, 3))

        # 3-panel dashboard
        t0 = img.copy()
        t1 = img.copy()
        stat = ChangeCategoryStat(class_id=1, class_name="Farm Pond Constructed", pixel_count=10000, percentage=6.25)
        panel = create_temporal_comparison_panel(t0, t1, mask, stats=[stat])
        self.assertGreater(panel.shape[1], img.shape[1] * 2)

if __name__ == "__main__":
    unittest.main()
