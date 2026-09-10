"""
WATERSCOPE-AI — YOLO Inference Module
Handles YOLO26 model loading, CUDA/CPU resolution, inference, and structured output formatting.
"""

import time
import os
from pathlib import Path
from typing import Union, Optional, List
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .config import AIConfig, FPCD_OBJECT_CLASSES
from .schemas import (
    BoundingBox,
    DetectionResult,
    InferenceMetadata,
    InferenceOutput,
)
from .preprocessing import load_image, validate_image
from .visualize import draw_detections

class WatershedYOLO:
    """
    Watershed Object Detection Wrapper.
    Supports YOLO26 pretrained baseline inference with transparent domain attribution,
    GPU/CUDA auto-selection, and structured output serialization.
    """

    def __init__(self, config: Optional[AIConfig] = None):
        self.config = config or AIConfig()
        self.device = self.config.resolve_device()
        self.model: Optional[YOLO] = None
        self.model_loaded_path: Optional[str] = None
        self.load_model()

    def load_model(self, model_path: Optional[Union[str, Path]] = None) -> None:
        """
        Loads the YOLO weights from the configured or specified path.
        """
        target_path = Path(model_path or self.config.model_path).resolve()
        if not target_path.is_file():
            # Check fallback in current working dir or repo root
            root_cand = Path.cwd() / target_path.name
            if root_cand.is_file():
                target_path = root_cand
            else:
                raise FileNotFoundError(f"YOLO model weights file not found at: {target_path}")

        self.model = YOLO(str(target_path))
        # Ensure model is allocated to the resolved device
        try:
            self.model.to(self.device)
        except Exception:
            # Fallback if device move fails (e.g. driver limitation)
            self.device = "cpu"
            self.model.to("cpu")

        self.model_loaded_path = str(target_path)

    def predict(
        self,
        image_input: Union[str, Path, np.ndarray],
        save_annotated: bool = False,
        output_path: Optional[Union[str, Path]] = None,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
    ) -> InferenceOutput:
        """
        Runs object detection inference on the provided image.

        Args:
            image_input: Image path or numpy array (BGR).
            save_annotated: Whether to save the annotated output image.
            output_path: Target path for the annotated image (if save_annotated is True).
            conf: Optional confidence threshold override.
            iou: Optional IoU threshold override.

        Returns:
            InferenceOutput: Structured detection output with metadata.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        img = load_image(image_input)
        validate_image(img)
        h, w, c = img.shape
        img_path_str = str(image_input) if isinstance(image_input, (str, Path)) else "<in-memory-array>"

        conf_thresh = conf if conf is not None else self.config.confidence_threshold
        iou_thresh = iou if iou is not None else self.config.iou_threshold

        # Run inference and time execution
        t0 = time.perf_counter()
        results = self.model(
            source=img,
            device=self.device,
            conf=conf_thresh,
            iou=iou_thresh,
            imgsz=self.config.imgsz,
            save=False,
            verbose=False,
        )
        total_time_ms = (time.perf_counter() - t0) * 1000

        result = results[0]
        speed = result.speed or {}
        prep_ms = float(speed.get("preprocess", 0.0))
        infer_ms = float(speed.get("inference", 0.0))
        post_ms = float(speed.get("postprocess", 0.0))

        # Parse detected boxes
        detections: List[DetectionResult] = []
        names = result.names or {}

        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                cls_name = names.get(cls_id, f"class_{cls_id}")
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]

                detections.append(
                    DetectionResult(
                        class_id=cls_id,
                        class_name=cls_name,
                        confidence=confidence,
                        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    )
                )

        # Baseline transparency warning
        warning_msg = None
        if self.config.is_pretrained_baseline:
            warning_msg = (
                "Notice: Pretrained generic COCO model weights (yolo26n.pt) were used. "
                "No watershed-specific fine-tuning has been performed yet. "
                "Detections reflect COCO classes, not dedicated farm-pond categories."
            )

        metadata = InferenceMetadata(
            model_name=Path(self.model_loaded_path).name if self.model_loaded_path else "unknown",
            model_domain=self.config.model_domain,
            is_pretrained_baseline=self.config.is_pretrained_baseline,
            device=self.device,
            image_path=img_path_str,
            image_shape=(h, w, c),
            preprocess_time_ms=round(prep_ms, 2),
            inference_time_ms=round(infer_ms, 2),
            postprocess_time_ms=round(post_ms, 2),
            total_time_ms=round(total_time_ms, 2),
            warning=warning_msg,
        )

        saved_annotated_str = None
        if save_annotated:
            if output_path is None:
                default_dir = Path("outputs")
                default_dir.mkdir(parents=True, exist_ok=True)
                stem = Path(img_path_str).stem if isinstance(image_input, (str, Path)) else "inference_result"
                output_path = default_dir / f"{stem}_annotated.jpg"

            out_p = Path(output_path).resolve()
            out_p.parent.mkdir(parents=True, exist_ok=True)

            # Ultralytics plot provides clean annotations with labels
            annotated = result.plot()
            cv2.imwrite(str(out_p), annotated)
            saved_annotated_str = str(out_p)

        return InferenceOutput(
            metadata=metadata,
            detections=detections,
            num_detections=len(detections),
            annotated_image_path=saved_annotated_str,
        )
