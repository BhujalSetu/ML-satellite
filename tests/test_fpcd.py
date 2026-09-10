"""
Unit Tests for FPCD Dataset and Change Detection
Verifies dataset structure, truthful annotation reporting, mask class bounds (0..4),
pair matching logic, and change detection metrics.
"""

import unittest
import numpy as np
from pathlib import Path

from src.ai.fpcd_dataset import FPCDDataset
from src.ai.change_detection import ChangeDetector
from src.ai.config import FPCD_CHANGE_CLASSES

class TestFPCDDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset = FPCDDataset()
        cls.detector = ChangeDetector(resolution_meters_per_pixel=1.0)

    def test_dataset_discovery_and_audit(self):
        audit = self.dataset.check_dataset_integrity()
        self.assertIn("root_dir", audit)
        self.assertGreaterEqual(audit["t0_images_found"], 690)
        self.assertGreaterEqual(audit["t1_images_found"], 690)
        self.assertGreaterEqual(audit["multi_class_masks_found"], 690)
        self.assertEqual(audit["complete_temporal_pairs"], 693)

        # Truthfulness check on annotations
        self.assertTrue(audit["test_coco_annotations_present"], "Test COCO annotations should exist")
        self.assertFalse(audit["train_coco_annotations_present"], "Train COCO annotations must NOT be fabricated")

    def test_pair_matching_consistency(self):
        pairs = self.dataset.get_matched_pairs()
        self.assertEqual(len(pairs), 693)

        # Inspect first and last pairs
        first = pairs[0]
        self.assertIn("t0_path", first)
        self.assertIn("t1_path", first)
        self.assertIn("mask_path", first)
        self.assertTrue(Path(first["t0_path"]).is_file())
        self.assertTrue(Path(first["t1_path"]).is_file())
        self.assertTrue(Path(first["mask_path"]).is_file())

    def test_mask_class_values(self):
        # Load sample pair and verify classes are strictly within {0, 1, 2, 3, 4}
        t0, t1, mask, info = self.dataset.load_pair(0)
        self.assertEqual(mask.ndim, 2)
        self.assertTrue(self.dataset.validate_mask_classes(mask))

        unique_vals = set(np.unique(mask).tolist())
        self.assertTrue(unique_vals.issubset({0, 1, 2, 3, 4}))

    def test_change_detection_statistics(self):
        # Synthetic mask with all 5 classes
        h, w = 100, 100
        total_px = h * w
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[0:10, 0:10] = 1   # 100 px Constructed
        mask[10:20, 0:10] = 2  # 100 px Demolished
        mask[20:30, 0:10] = 3  # 100 px Dried
        mask[30:40, 0:10] = 4  # 100 px Wetted
        # Remaining 9600 px are Background (0)

        stats = self.detector.compute_change_statistics(mask, resolution_m=1.0)
        self.assertEqual(len(stats), 5)

        stats_by_id = {s.class_id: s for s in stats}
        self.assertEqual(stats_by_id[0].pixel_count, 9600)
        self.assertEqual(stats_by_id[0].percentage, 96.0)

        for c_id in [1, 2, 3, 4]:
            self.assertEqual(stats_by_id[c_id].pixel_count, 100)
            self.assertEqual(stats_by_id[c_id].percentage, 1.0)
            self.assertEqual(stats_by_id[c_id].estimated_area_m2, 100.0)
            self.assertEqual(stats_by_id[c_id].estimated_area_hectares, 0.01)

if __name__ == "__main__":
    unittest.main()
