"""
WATERSCOPE-AI — AI Route Handlers
Implements AI object detection and bi-temporal change detection endpoints.
"""

import time
import uuid
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from pydantic import ValidationError

from src.ai.inference import WatershedYOLO
from src.ai.preprocessing import validate_image
from src.ai.change_detection import ChangeDetector
from src.ai.visualize import create_temporal_comparison_panel
from src.api.schemas import (
    AIAnalyzeImageRequest,
    AIAnalyzeImageResponse,
    AIDetectionItem,
    AIDetectionSummary,
    AIAnalyzeImageOutputs,
    LocationCoordinates,
    AIChangeDetectionRequest,
    AIChangeDetectionResponse,
    AIChangeDetail,
    AIChangeDetectionOutputs,
)

router = APIRouter(prefix="/api/v1/ai", tags=["AI"])

# Safe resolution of outputs/ai directory relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
AI_OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "ai"
AI_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Application-level lazy/singleton instantiation for WatershedYOLO
_yolo_instance: Optional[WatershedYOLO] = None

def get_yolo_model() -> WatershedYOLO:
    global _yolo_instance
    if _yolo_instance is None:
        _yolo_instance = WatershedYOLO()
    return _yolo_instance


@router.post(
    "/analyze-image",
    response_model=AIAnalyzeImageResponse,
    status_code=status.HTTP_200_OK,
    summary="Detect objects and potential watershed features in an image",
)
async def analyze_image(
    file: UploadFile = File(..., description="Uploaded image file (JPEG, PNG, WebP)"),
    latitude: float = Form(..., description="Observation latitude (-90.0 to 90.0)"),
    longitude: float = Form(..., description="Observation longitude (-180.0 to 180.0)"),
):
    """
    Analyzes an uploaded image using the configured YOLO model:
    1. Validates geographic coordinates.
    2. Reads and validates the image bytes safely in memory.
    3. Runs object detection using the shared WatershedYOLO model instance.
    4. Saves the annotated result image to outputs/ai/<image_id>_annotated.jpg.
    5. Returns detection items and API-relative URL path (/files/ai/...).
    """
    t_start = time.perf_counter()

    # 1. Validate coordinates using Pydantic request schema
    try:
        req_coords = AIAnalyzeImageRequest(latitude=latitude, longitude=longitude)
    except ValidationError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=ve.errors(),
        )

    # 2. Check file presence and read content
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a valid filename.",
        )

    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {str(e)}",
        )

    if not content or len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded image file is empty.",
        )

    # Decode image from buffer using OpenCV
    np_arr = np.frombuffer(content, np.uint8)
    image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image_bgr is None or image_bgr.size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image format. Could not decode image bytes.",
        )

    try:
        validate_image(image_bgr)
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image validation failed: {str(ve)}",
        )

    # 3. Generate unique request and image identifiers
    request_id = str(uuid.uuid4())
    image_id = f"img_{uuid.uuid4().hex[:12]}"

    # 4. Execute inference with WatershedYOLO
    try:
        yolo = get_yolo_model()
        annotated_filename = f"{image_id}_annotated.jpg"
        annotated_disk_path = AI_OUTPUTS_DIR / annotated_filename

        infer_output = yolo.predict(
            image_input=image_bgr,
            save_annotated=True,
            output_path=annotated_disk_path,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline execution error: {str(e)}",
        )

    # 5. Format detection results according to API contract
    detection_items = [
        AIDetectionItem(
            class_name=d.class_name,
            confidence=round(d.confidence, 4),
            bbox=[
                round(d.bbox.x1, 2),
                round(d.bbox.y1, 2),
                round(d.bbox.x2, 2),
                round(d.bbox.y2, 2),
            ],
        )
        for d in infer_output.detections
    ]

    # Model attribution and pond count governance
    # Pretrained baseline is generic COCO; count only genuine pond class if present
    ponds_count = sum(1 for d in detection_items if d.class_name.lower() in ["pond", "farm_pond"])
    total_objects = len(detection_items)

    elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)

    # API-relative URL path (never exposes C:\Users\...)
    annotated_url = f"/files/ai/{annotated_filename}"

    return AIAnalyzeImageResponse(
        success=True,
        request_id=request_id,
        image_id=image_id,
        location=LocationCoordinates(
            latitude=req_coords.latitude,
            longitude=req_coords.longitude,
        ),
        model=infer_output.metadata.model_name,
        detections=detection_items,
        summary=AIDetectionSummary(
            objects_detected=total_objects,
            ponds_detected=ponds_count,
        ),
        outputs=AIAnalyzeImageOutputs(
            annotated_image_url=annotated_url,
        ),
        processing_time_ms=elapsed_ms,
    )


@router.post(
    "/change-detection",
    response_model=AIChangeDetectionResponse,
    status_code=status.HTTP_200_OK,
    summary="Bi-temporal farm pond change detection between before and after images",
)
async def change_detection(
    before_image: UploadFile = File(..., description="T0 / before image file (JPEG, PNG, WebP)"),
    after_image: UploadFile = File(..., description="T1 / after image file (JPEG, PNG, WebP)"),
    latitude: Optional[float] = Form(None, description="Optional observation latitude (-90.0 to 90.0)"),
    longitude: Optional[float] = Form(None, description="Optional observation longitude (-180.0 to 180.0)"),
    mask_file: Optional[UploadFile] = File(None, description="Optional ground-truth or indexed mask (classes 0..4)"),
):
    """
    Executes bi-temporal change detection between before (T0) and after (T1) images:
    1. Validates optional geographic coordinates using AIChangeDetectionRequest.
    2. Reads and validates before and after images in-memory safely.
    3. Verifies image dimension compatibility (raises HTTP 400 on dimension mismatch).
    4. Evaluates or accepts multi-class change mask (classes 0-4 per FPCD convention).
    5. Calculates change statistics via ChangeDetector.compute_change_statistics().
    6. Renders composite 3-panel comparison dashboard to outputs/ai/change_<request_id>.jpg.
    7. Returns API-relative URL path (/files/ai/...) with genuine calculated change metrics.
    """
    t_start = time.perf_counter()

    # 1. Validate optional coordinates using Pydantic request schema
    if latitude is not None or longitude is not None:
        try:
            AIChangeDetectionRequest(latitude=latitude, longitude=longitude)
        except ValidationError as ve:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=ve.errors(),
            )

    # 2. Check file presence and filenames
    if not before_image.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded before_image must have a valid filename.",
        )
    if not after_image.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded after_image must have a valid filename.",
        )

    # 3. Read image bytes safely
    try:
        content_before = await before_image.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read before_image: {str(e)}",
        )
    try:
        content_after = await after_image.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read after_image: {str(e)}",
        )

    if not content_before or len(content_before) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded before_image is empty.",
        )
    if not content_after or len(content_after) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded after_image is empty.",
        )

    # 4. Decode images via OpenCV
    np_before = np.frombuffer(content_before, np.uint8)
    img_before = cv2.imdecode(np_before, cv2.IMREAD_COLOR)
    if img_before is None or img_before.size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid before_image format. Could not decode image bytes.",
        )

    np_after = np.frombuffer(content_after, np.uint8)
    img_after = cv2.imdecode(np_after, cv2.IMREAD_COLOR)
    if img_after is None or img_after.size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid after_image format. Could not decode image bytes.",
        )

    # 5. Validate image contents
    try:
        validate_image(img_before)
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"before_image validation failed: {str(ve)}",
        )
    try:
        validate_image(img_after)
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"after_image validation failed: {str(ve)}",
        )

    # 6. Verify dimensional compatibility
    h0, w0 = img_before.shape[:2]
    h1, w1 = img_after.shape[:2]
    if (h0, w0) != (h1, w1):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image dimensions mismatch: before_image has dimensions ({h0}, {w0}) but after_image has dimensions ({h1}, {w1}). Bi-temporal change detection requires matching dimensions.",
        )

    # 7. Derive or load multi-class change mask
    mask: Optional[np.ndarray] = None
    if mask_file is not None and mask_file.filename:
        try:
            mask_bytes = await mask_file.read()
            if mask_bytes and len(mask_bytes) > 0:
                mask_arr = np.frombuffer(mask_bytes, np.uint8)
                decoded_mask = cv2.imdecode(mask_arr, cv2.IMREAD_UNCHANGED)
                if decoded_mask is not None:
                    if decoded_mask.ndim == 3:
                        decoded_mask = cv2.cvtColor(decoded_mask, cv2.COLOR_BGR2GRAY)
                    if decoded_mask.shape[:2] == (h0, w0):
                        mask = decoded_mask.astype(np.uint8)
                    else:
                        mask = cv2.resize(decoded_mask, (w0, h0), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        except Exception:
            mask = None

    if mask is None:
        # Bi-temporal intensity difference interpretation
        if np.array_equal(img_before, img_after):
            mask = np.zeros((h0, w0), dtype=np.uint8)
        else:
            t0_gray = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY).astype(np.int16)
            t1_gray = cv2.cvtColor(img_after, cv2.COLOR_BGR2GRAY).astype(np.int16)
            delta = t1_gray - t0_gray
            mask = np.zeros((h0, w0), dtype=np.uint8)
            # FPCD convention:
            # 1: Farm Pond Constructed (excavation / water filling leads to lower reflectance, delta < -30)
            # 2: Farm Pond Demolished (pond filled / dried leads to higher ground reflectance, delta > 30)
            mask[delta < -30] = 1
            mask[delta > 30] = 2

    # 8. Compute change statistics using ChangeDetector
    try:
        detector = ChangeDetector(resolution_meters_per_pixel=1.0)
        stats = detector.compute_change_statistics(mask)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Change detection engine computation error: {str(e)}",
        )

    # 9. Format response metrics and change details
    total_pixels = mask.size
    changed_stats = [s for s in stats if s.class_id != 0 and s.pixel_count > 0]
    total_changed_pixels = sum(s.pixel_count for s in changed_stats)
    change_detected = total_changed_pixels > 0

    if change_detected:
        change_percentage = round((total_changed_pixels / total_pixels) * 100.0, 2) if total_pixels > 0 else 0.0
        primary_stat = max(changed_stats, key=lambda s: s.pixel_count)
        change_type = primary_stat.class_name
        changes = [
            AIChangeDetail(
                type=s.class_name,
                confidence=None,  # Truthful: ChangeDetector produces no model confidence score
            )
            for s in changed_stats
        ]
    else:
        change_percentage = 0.0
        change_type = None
        changes = []

    # 10. Generate and save composite 3-panel visualization
    request_id = str(uuid.uuid4())
    change_filename = f"change_{request_id}.jpg"
    change_disk_path = AI_OUTPUTS_DIR / change_filename

    try:
        # If mask contains no changed pixels, supply 1-px dummy for overlay rendering
        # to prevent OpenCV addWeighted() empty slice error; (0,0) is cleanly covered by sublabel banner.
        if np.count_nonzero(mask > 0) == 0:
            mask_vis = mask.copy()
            mask_vis[0, 0] = 1
        else:
            mask_vis = mask

        panel = create_temporal_comparison_panel(
            t0_img=img_before,
            t1_img=img_after,
            mask=mask_vis,
            stats=stats,
            title=f"WATERSCOPE-AI: Change Detection [{request_id[:8]}]",
        )
        cv2.imwrite(str(change_disk_path), panel)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create change visualization panel: {str(e)}",
        )

    change_map_url = f"/files/ai/{change_filename}"
    elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)

    return AIChangeDetectionResponse(
        success=True,
        request_id=request_id,
        change_detected=change_detected,
        change_type=change_type,
        confidence=None,
        change_percentage=change_percentage,
        changes=changes,
        outputs=AIChangeDetectionOutputs(
            change_map_url=change_map_url,
            annotated_before_url=None,
            annotated_after_url=None,
        ),
        processing_time_ms=elapsed_ms,
    )
