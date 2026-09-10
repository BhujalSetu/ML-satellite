"""
WATERSCOPE-AI — Visualization Utilities
Renders detection bounding boxes, change masks, and temporal comparison panels.
"""

from pathlib import Path
from typing import List, Optional, Dict, Tuple, Union
import cv2
import numpy as np

from .config import CHANGE_CLASS_COLORS, FPCD_CHANGE_CLASSES, OBJECT_CLASS_COLORS, FPCD_OBJECT_CLASSES
from .schemas import DetectionResult, ChangeCategoryStat

def draw_detections(
    image: np.ndarray,
    detections: List[DetectionResult],
    class_colors: Optional[Dict[int, Tuple[int, int, int]]] = None
) -> np.ndarray:
    """
    Draws bounding boxes and labels onto an image.

    Args:
        image: Source BGR image.
        detections: List of DetectionResult objects.
        class_colors: Dict mapping class_id to BGR color tuple.

    Returns:
        np.ndarray: Annotated BGR image.
    """
    canvas = image.copy()
    colors = class_colors or OBJECT_CLASS_COLORS

    for det in detections:
        box = det.bbox
        x1, y1, x2, y2 = int(box.x1), int(box.y1), int(box.x2), int(box.y2)
        color = colors.get(det.class_id, (0, 255, 0))

        # Draw box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        # Label background and text
        label = f"{det.class_name} {det.confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        label_y = max(y1, text_h + 6)
        cv2.rectangle(
            canvas,
            (x1, label_y - text_h - 6),
            (x1 + text_w + 6, label_y),
            color,
            -1
        )
        cv2.putText(
            canvas,
            label,
            (x1 + 3, label_y - 4),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

    return canvas

def draw_change_mask(
    mask: np.ndarray,
    alpha: float = 0.6,
    bg_image: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Renders an indexed multi-class change mask (classes 0-4) as a colored BGR image or blended overlay.

    Args:
        mask: 2D array with class indices 0..4.
        alpha: Blending weight for mask when bg_image is provided.
        bg_image: Optional background BGR image to blend onto.

    Returns:
        np.ndarray: Colored or blended BGR image.
    """
    h, w = mask.shape[:2]
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for cls_id, color in CHANGE_CLASS_COLORS.items():
        if cls_id == 0:
            continue  # Keep background dark/uncolored
        colored_mask[mask == cls_id] = color

    if bg_image is not None:
        if bg_image.shape[:2] != (h, w):
            bg_image = cv2.resize(bg_image, (w, h))
        # Where mask > 0, blend colored mask with background
        fg_pixels = mask > 0
        blended = bg_image.copy()
        blended[fg_pixels] = cv2.addWeighted(
            bg_image[fg_pixels], 1.0 - alpha,
            colored_mask[fg_pixels], alpha, 0
        )
        return blended

    return colored_mask

def create_temporal_comparison_panel(
    t0_img: np.ndarray,
    t1_img: np.ndarray,
    mask: np.ndarray,
    stats: Optional[List[ChangeCategoryStat]] = None,
    title: str = "WATERSCOPE-AI: FPCD Temporal Change Detection"
) -> np.ndarray:
    """
    Creates a comprehensive 3-panel visualization:
    [ T0 (Initial) ]  [ T1 (Changed) ]  [ Change Mask Overlay ]
    with an informative statistics and legend footer banner.

    Args:
        t0_img: T0 BGR image.
        t1_img: T1 BGR image.
        mask: Multi-class change mask (0..4).
        stats: List of ChangeCategoryStat summary statistics.
        title: Header text for panel.

    Returns:
        np.ndarray: Composite dashboard image ready to save or display.
    """
    # Standardize image sizes to T1 size
    target_h, target_w = t1_img.shape[:2]
    if t0_img.shape[:2] != (target_h, target_w):
        t0_img = cv2.resize(t0_img, (target_w, target_h))
    if mask.shape[:2] != (target_h, target_w):
        mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    # Generate colored overlay on T1
    overlay = draw_change_mask(mask, alpha=0.55, bg_image=t1_img)

    # Add individual headers on each image
    def add_sublabel(img: np.ndarray, text: str) -> np.ndarray:
        c = img.copy()
        cv2.rectangle(c, (0, 0), (target_w, 36), (20, 20, 20), -1)
        cv2.putText(c, text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        return c

    p0 = add_sublabel(t0_img, "T0: Initial Aerial Capture")
    p1 = add_sublabel(t1_img, "T1: Subsequent Capture")
    p2 = add_sublabel(overlay, "Change Mask Overlay (0-4)")

    # Concatenate side by side
    panels = np.hstack([p0, p1, p2])
    total_w = panels.shape[1]

    # Header bar
    header_h = 50
    header = np.full((header_h, total_w, 3), 30, dtype=np.uint8)
    cv2.putText(
        header, title, (20, 34),
        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 215, 255), 2, cv2.LINE_AA
    )

    # Footer banner with legend & stats
    footer_h = 100
    footer = np.full((footer_h, total_w, 3), 24, dtype=np.uint8)

    # Render class color legend chips
    chip_x = 20
    chip_y = 25
    cv2.putText(footer, "Legend:", (chip_x, chip_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    chip_x += 70

    for cls_id in [1, 2, 3, 4]:
        name = FPCD_CHANGE_CLASSES.get(cls_id, f"Class {cls_id}")
        color = CHANGE_CLASS_COLORS.get(cls_id, (255, 255, 255))

        # Color square
        cv2.rectangle(footer, (chip_x, chip_y), (chip_x + 18, chip_y + 18), color, -1)
        cv2.rectangle(footer, (chip_x, chip_y), (chip_x + 18, chip_y + 18), (255, 255, 255), 1)

        # Class text
        cv2.putText(footer, name, (chip_x + 24, chip_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1)
        chip_x += len(name) * 8 + 35

    # Render stats if available
    if stats:
        stat_y = 68
        stat_text_parts = []
        for s in stats:
            if s.class_id != 0 and s.pixel_count > 0:
                stat_text_parts.append(f"{s.class_name}: {s.pixel_count} px ({s.percentage:.2f}%)")
        stat_line = " | ".join(stat_text_parts) if stat_text_parts else "No change detected in active classes."
        cv2.putText(footer, f"Metrics: {stat_line}", (20, stat_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 255, 180), 1)

    # Stack full dashboard
    dashboard = np.vstack([header, panels, footer])
    return dashboard
