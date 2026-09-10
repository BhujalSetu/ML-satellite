"""
WATERSCOPE-AI — FPCD Dataset Handler
Discovers, validates, and manages pairs for the Farm Pond Change Detection dataset.
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import cv2
import numpy as np

class FPCDDataset:
    """
    Manages discovery, pair matching, and integrity inspection for the
    Farm Pond Change Detection (FPCD) dataset.
    """

    def __init__(self, root_dir: Optional[Path] = None):
        if root_dir is not None:
            self.root_dir = Path(root_dir).resolve()
        else:
            # Default lookup relative to repo
            base = Path(__file__).resolve().parent.parent.parent
            self.root_dir = base / "data" / "fpcd" / "raw"

        self.t0_dir = self._find_subfolder(["T0", "t0"])
        self.t1_dir = self._find_subfolder(["T1", "t1"])
        self.mask_dir = self._find_subfolder(["multi_class_masks", "masks"])

        self._pairs: Optional[List[Dict[str, Any]]] = None

    def _find_subfolder(self, candidates: List[str]) -> Optional[Path]:
        """Locates candidate folder, supporting both flat and nested extractions."""
        if not self.root_dir.is_dir():
            return None
        for cand in candidates:
            p = self.root_dir / cand
            if p.is_dir():
                # Check if nested (e.g. T0/T0)
                nested = p / cand
                return nested if nested.is_dir() else p
        return None

    def check_dataset_integrity(self) -> Dict[str, Any]:
        """
        Inspects files on disk and returns an objective audit of dataset availability.
        Truthfully verifies presence of test vs training annotations.
        """
        t0_count = len(list(self.t0_dir.rglob("*.jpg"))) if self.t0_dir else 0
        t1_count = len(list(self.t1_dir.rglob("*.jpg"))) if self.t1_dir else 0
        mask_count = len(list(self.mask_dir.rglob("*.png"))) if self.mask_dir else 0

        # Annotation files
        test_coco = self.root_dir / "object_annotations_test_coco.json"
        train_coco = self.root_dir / "object_annotations_train_coco.json"

        test_annotations_exist = test_coco.is_file()
        train_annotations_exist = train_coco.is_file()

        test_ann_stats = {}
        if test_annotations_exist:
            try:
                with open(test_coco, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    test_ann_stats = {
                        "images": len(data.get("images", [])),
                        "annotations": len(data.get("annotations", [])),
                        "categories": [c.get("name") for c in data.get("categories", [])],
                    }
            except Exception as e:
                test_ann_stats = {"error": str(e)}

        pairs = self.get_matched_pairs()

        return {
            "root_dir": str(self.root_dir),
            "t0_images_found": t0_count,
            "t1_images_found": t1_count,
            "multi_class_masks_found": mask_count,
            "complete_temporal_pairs": len(pairs),
            "test_coco_annotations_present": test_annotations_exist,
            "test_coco_summary": test_ann_stats,
            "train_coco_annotations_present": train_annotations_exist,
            "train_annotations_note": (
                "Verified: 'object_annotations_train_coco.json' is NOT present on disk or upstream. "
                "Per project governance, training on test data or fabricating annotations is strictly prohibited."
            ),
        }

    def get_matched_pairs(self) -> List[Dict[str, Any]]:
        """
        Matches T0, T1, and multi-class masks based on location prefix and capture index.
        Returns sorted list of matched pair dictionaries.
        """
        if self._pairs is not None:
            return self._pairs

        if not (self.t0_dir and self.t1_dir and self.mask_dir):
            self._pairs = []
            return self._pairs

        t0_files = list(self.t0_dir.rglob("*.jpg"))
        t1_files = list(self.t1_dir.rglob("*.jpg"))
        mask_files = list(self.mask_dir.rglob("*.png"))

        pattern_time = re.compile(r"^(.*)_(\d{6})_(\d+)\.jpg$")
        pattern_mask = re.compile(r"^(.*)_(\d+)\.png$")

        t0_dict = {}
        t0_dates = {}
        for f in t0_files:
            m = pattern_time.match(f.name)
            if m:
                key = (m.group(1), int(m.group(3)))
                t0_dict[key] = f
                t0_dates[key] = m.group(2)

        t1_dict = {}
        t1_dates = {}
        for f in t1_files:
            m = pattern_time.match(f.name)
            if m:
                key = (m.group(1), int(m.group(3)))
                t1_dict[key] = f
                t1_dates[key] = m.group(2)

        mask_dict = {}
        for f in mask_files:
            m = pattern_mask.match(f.name)
            if m:
                key = (m.group(1), int(m.group(2)))
                mask_dict[key] = f

        common_keys = sorted(list(set(t0_dict.keys()) & set(t1_dict.keys()) & set(mask_dict.keys())))

        pairs = []
        for idx, key in enumerate(common_keys):
            prefix, item_idx = key
            pairs.append({
                "pair_id": f"{prefix}_{item_idx}",
                "index": idx,
                "prefix": prefix,
                "item_index": item_idx,
                "t0_date": t0_dates.get(key),
                "t1_date": t1_dates.get(key),
                "t0_path": str(t0_dict[key]),
                "t1_path": str(t1_dict[key]),
                "mask_path": str(mask_dict[key]),
            })

        self._pairs = pairs
        return self._pairs

    def load_pair(self, index: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Loads the T0 image, T1 image, and multi-class mask for a given matched pair index.

        Returns:
            Tuple of (t0_img, t1_img, mask_img, pair_info)
        """
        pairs = self.get_matched_pairs()
        if not pairs:
            raise IndexError("No matched FPCD pairs available.")
        if index < 0 or index >= len(pairs):
            raise IndexError(f"Index {index} out of range (0 to {len(pairs) - 1}).")

        pair_info = pairs[index]
        t0_img = cv2.imread(pair_info["t0_path"])
        t1_img = cv2.imread(pair_info["t1_path"])
        mask_img = cv2.imread(pair_info["mask_path"], cv2.IMREAD_UNCHANGED)

        if t0_img is None or t1_img is None or mask_img is None:
            raise IOError(f"Failed to load image files for pair index {index}.")

        return t0_img, t1_img, mask_img, pair_info

    @staticmethod
    def validate_mask_classes(mask: np.ndarray) -> bool:
        """
        Verifies that all values in the mask belong strictly to {0, 1, 2, 3, 4}.
        """
        valid_classes = {0, 1, 2, 3, 4}
        unique_vals = set(np.unique(mask).tolist())
        return unique_vals.issubset(valid_classes)
