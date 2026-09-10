"""
WATERSCOPE-AI — Farm Pond Change Detection Module
Performs temporal change analysis between T0 and T1 paired images using multi-class masks.
Calculates per-class pixel counts, percentages, spatial areas, and composite visualizations.
"""

from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import cv2
import numpy as np

from .config import FPCD_CHANGE_CLASSES
from .schemas import ChangeCategoryStat, ChangeDetectionOutput
from .preprocessing import load_image, validate_image
from .visualize import create_temporal_comparison_panel

class ChangeDetector:
    """
    Analyzes temporal change between T0 and T1 farm-pond imagery using ground-truth
    or predicted multi-class change masks.
    """

    def __init__(self, resolution_meters_per_pixel: float = 1.0):
        # FPCD dataset zoom level 18 corresponds to approximately ~0.6 - 1.0 m/pixel
        self.resolution_m = resolution_meters_per_pixel

    def compute_change_statistics(
        self,
        mask: np.ndarray,
        resolution_m: Optional[float] = None
    ) -> List[ChangeCategoryStat]:
        """
        Computes pixel distribution and physical area estimates for all 5 change classes.

        Args:
            mask: 2D integer array with pixel classes in 0..4.
            resolution_m: Spatial resolution in meters per pixel.

        Returns:
            List[ChangeCategoryStat]: Statistics per change class.
        """
        if mask.ndim != 2:
            raise ValueError(f"Mask must be a 2D single-channel array, got shape {mask.shape}")

        res = resolution_m if resolution_m is not None else self.resolution_m
        pixel_area_m2 = res * res
        total_pixels = mask.size

        stats: List[ChangeCategoryStat] = []
        for cls_id in range(5):
            cls_name = FPCD_CHANGE_CLASSES.get(cls_id, f"Class {cls_id}")
            count = int(np.count_nonzero(mask == cls_id))
            pct = (count / total_pixels) * 100.0 if total_pixels > 0 else 0.0
            area_m2 = count * pixel_area_m2
            area_ha = area_m2 / 10000.0

            stats.append(
                ChangeCategoryStat(
                    class_id=cls_id,
                    class_name=cls_name,
                    pixel_count=count,
                    percentage=round(pct, 3),
                    estimated_area_m2=round(area_m2, 2),
                    estimated_area_hectares=round(area_ha, 4),
                )
            )

        return stats

    def analyze_temporal_pair(
        self,
        t0_path: Union[str, Path],
        t1_path: Union[str, Path],
        mask_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        save_visualization: bool = True
    ) -> ChangeDetectionOutput:
        """
        Executes end-to-end temporal analysis on an image pair and mask.

        Args:
            t0_path: Filepath to T0 image.
            t1_path: Filepath to T1 image.
            mask_path: Filepath to indexed PNG mask.
            output_dir: Directory to store generated visualization.
            save_visualization: Whether to render and save the 3-panel dashboard.

        Returns:
            ChangeDetectionOutput: Structured result object ready for API responses.
        """
        t0_p = Path(t0_path).resolve()
        t1_p = Path(t1_path).resolve()
        mask_p = Path(mask_path).resolve()

        if not t0_p.is_file():
            raise FileNotFoundError(f"T0 image not found: {t0_p}")
        if not t1_p.is_file():
            raise FileNotFoundError(f"T1 image not found: {t1_p}")
        if not mask_p.is_file():
            raise FileNotFoundError(f"Mask file not found: {mask_p}")

        t0_img = load_image(t0_p)
        t1_img = load_image(t1_p)
        mask = cv2.imread(str(mask_p), cv2.IMREAD_UNCHANGED)

        if mask is None:
            raise ValueError(f"Failed to read mask image: {mask_p}")

        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        h, w = mask.shape[:2]
        stats = self.compute_change_statistics(mask)

        vis_path_str = None
        if save_visualization:
            out_dir = Path(output_dir or "outputs/change_detection").resolve()
            out_dir.mkdir(parents=True, exist_ok=True)
            vis_file = out_dir / f"{mask_p.stem}_comparison.jpg"

            dashboard = create_temporal_comparison_panel(
                t0_img=t0_img,
                t1_img=t1_img,
                mask=mask,
                stats=stats,
                title=f"WATERSCOPE-AI: Farm Pond Change Detection [{mask_p.stem}]"
            )
            cv2.imwrite(str(vis_file), dashboard)
            vis_path_str = str(vis_file)

        return ChangeDetectionOutput(
            t0_path=str(t0_p),
            t1_path=str(t1_p),
            mask_path=str(mask_p),
            image_dimensions=(h, w),
            total_pixels=mask.size,
            category_statistics=stats,
            visualization_path=vis_path_str,
        )
