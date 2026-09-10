"""
WATERSCOPE-AI — AI Configuration
Manages model paths, inference device resolution (CUDA/CPU), thresholds, and class definitions.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional
import torch

# Root of the repository
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# FPCD Object Detection Classes (4 classes)
FPCD_OBJECT_CLASSES: Dict[int, str] = {
    0: "Wet Farm Pond - Lined",
    1: "Wet Farm Pond - Unlined",
    2: "Dry Farm Pond - Lined",
    3: "Dry Farm Pond - Unlined",
}

# FPCD Temporal Change Detection Classes (5 classes)
FPCD_CHANGE_CLASSES: Dict[int, str] = {
    0: "Background",
    1: "Farm Pond Constructed",
    2: "Farm Pond Demolished",
    3: "Farm Pond Dried",
    4: "Farm Pond Wetted",
}

# BGR Colors for visualization (OpenCV convention)
CHANGE_CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 0),         # Background: Black (transparent overlay)
    1: (0, 200, 0),       # Farm Pond Constructed: Green
    2: (0, 0, 220),       # Farm Pond Demolished: Red
    3: (0, 165, 255),     # Farm Pond Dried: Orange
    4: (255, 191, 0),     # Farm Pond Wetted: Cyan/Light Blue
}

OBJECT_CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (255, 140, 0),     # Wet Farm Pond - Lined: Deep Blue
    1: (205, 90, 106),    # Wet Farm Pond - Unlined: Steel Blue
    2: (50, 205, 50),     # Dry Farm Pond - Lined: Lime Green
    3: (34, 139, 34),     # Dry Farm Pond - Unlined: Forest Green
}

@dataclass
class AIConfig:
    model_path: str = str(ROOT_DIR / "yolo26n.pt")
    device: str = "auto"  # 'auto', 'cuda', 'cpu'
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    imgsz: int = 640
    is_pretrained_baseline: bool = True
    model_domain: str = "generic_coco"  # 'generic_coco' or 'watershed_fine_tuned'

    def resolve_device(self) -> str:
        """
        Selects CUDA automatically if available and requested, otherwise falls back to CPU.
        """
        if self.device.lower() in ("cuda", "gpu"):
            if torch.cuda.is_available():
                return "cuda"
            return "cpu"
        elif self.device.lower() == "cpu":
            return "cpu"
        else:  # 'auto'
            return "cuda" if torch.cuda.is_available() else "cpu"

    def get_device_name(self) -> str:
        """Returns the human-readable device name."""
        resolved = self.resolve_device()
        if resolved == "cuda":
            return f"CUDA: {torch.cuda.get_device_name(0)}"
        return "CPU"
