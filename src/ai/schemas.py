"""
WATERSCOPE-AI — AI Pipeline Schemas
Defines structured data contracts for object detection, metadata, and change analysis.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Dict, Any

@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['width'] = round(self.width, 2)
        d['height'] = round(self.height, 2)
        d['area'] = round(self.area, 2)
        return d

@dataclass
class DetectionResult:
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox

    def to_dict(self) -> Dict[str, Any]:
        return {
            'class_id': self.class_id,
            'class_name': self.class_name,
            'confidence': round(self.confidence, 4),
            'bbox': self.bbox.to_dict(),
        }

@dataclass
class InferenceMetadata:
    model_name: str
    model_domain: str
    is_pretrained_baseline: bool
    device: str
    image_path: str
    image_shape: Tuple[int, int, int]
    preprocess_time_ms: float
    inference_time_ms: float
    postprocess_time_ms: float
    total_time_ms: float
    warning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class InferenceOutput:
    metadata: InferenceMetadata
    detections: List[DetectionResult] = field(default_factory=list)
    num_detections: int = 0
    annotated_image_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'metadata': self.metadata.to_dict(),
            'num_detections': self.num_detections,
            'detections': [d.to_dict() for d in self.detections],
            'annotated_image_path': self.annotated_image_path,
        }

@dataclass
class ChangeCategoryStat:
    class_id: int
    class_name: str
    pixel_count: int
    percentage: float
    estimated_area_m2: Optional[float] = None
    estimated_area_hectares: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ChangeDetectionOutput:
    t0_path: str
    t1_path: str
    mask_path: str
    image_dimensions: Tuple[int, int]
    total_pixels: int
    category_statistics: List[ChangeCategoryStat] = field(default_factory=list)
    visualization_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            't0_path': self.t0_path,
            't1_path': self.t1_path,
            'mask_path': self.mask_path,
            'image_dimensions': self.image_dimensions,
            'total_pixels': self.total_pixels,
            'category_statistics': [s.to_dict() for s in self.category_statistics],
            'visualization_path': self.visualization_path,
        }
