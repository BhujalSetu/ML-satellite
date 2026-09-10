"""
WATERSCOPE-AI — AI & Computer Vision Package
Module for object detection, farm pond change detection, and visual analytics.
"""

from .config import (
    AIConfig,
    FPCD_OBJECT_CLASSES,
    FPCD_CHANGE_CLASSES,
    CHANGE_CLASS_COLORS,
    OBJECT_CLASS_COLORS,
)
from .schemas import (
    BoundingBox,
    DetectionResult,
    InferenceMetadata,
    InferenceOutput,
    ChangeCategoryStat,
    ChangeDetectionOutput,
)
from .preprocessing import (
    load_image,
    validate_image,
    resize_image_aspect_ratio,
    normalize_image,
)
from .inference import WatershedYOLO
from .visualize import (
    draw_detections,
    draw_change_mask,
    create_temporal_comparison_panel,
)
from .change_detection import ChangeDetector
from .fpcd_dataset import FPCDDataset

__all__ = [
    "AIConfig",
    "FPCD_OBJECT_CLASSES",
    "FPCD_CHANGE_CLASSES",
    "CHANGE_CLASS_COLORS",
    "OBJECT_CLASS_COLORS",
    "BoundingBox",
    "DetectionResult",
    "InferenceMetadata",
    "InferenceOutput",
    "ChangeCategoryStat",
    "ChangeDetectionOutput",
    "load_image",
    "validate_image",
    "resize_image_aspect_ratio",
    "normalize_image",
    "WatershedYOLO",
    "draw_detections",
    "draw_change_mask",
    "create_temporal_comparison_panel",
    "ChangeDetector",
    "FPCDDataset",
]
