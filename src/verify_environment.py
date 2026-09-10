"""
WATERSCOPE-AI — Environment Verification Script
Member 3: AI / Computer Vision Module

Verifies that all required packages are installed and functional.
"""

import sys

print("=" * 60)
print("  WATERSCOPE-AI — Environment Verification")
print("=" * 60)

# --- Python ---
print(f"\n[Python]")
print(f"  Version : {sys.version}")

# --- OpenCV ---
print(f"\n[OpenCV]")
try:
    import cv2
    print(f"  Version : {cv2.__version__}")
    print(f"  Status  : OK")
except ImportError as e:
    print(f"  Status  : FAILED — {e}")

# --- NumPy ---
print(f"\n[NumPy]")
try:
    import numpy as np
    print(f"  Version : {np.__version__}")
    print(f"  Status  : OK")
except ImportError as e:
    print(f"  Status  : FAILED — {e}")

# --- PyTorch ---
print(f"\n[PyTorch]")
try:
    import torch
    print(f"  Version : {torch.__version__}")
    print(f"  Status  : OK")
except ImportError as e:
    print(f"  Status  : FAILED — {e}")
    torch = None

# --- CUDA / GPU ---
print(f"\n[CUDA / GPU]")
if torch is not None:
    cuda_available = torch.cuda.is_available()
    print(f"  CUDA Available : {cuda_available}")
    if cuda_available:
        gpu_count = torch.cuda.device_count()
        print(f"  GPU Count      : {gpu_count}")
        for i in range(gpu_count):
            print(f"  GPU [{i}]        : {torch.cuda.get_device_name(i)}")
    else:
        print(f"  Note           : Running on CPU (no CUDA-capable GPU detected)")
else:
    print(f"  Skipped — PyTorch not available")

# --- Ultralytics YOLO ---
print(f"\n[Ultralytics YOLO]")
try:
    from ultralytics import YOLO
    import ultralytics
    print(f"  Version : {ultralytics.__version__}")
    print(f"  Status  : OK — YOLO class imported successfully")
except ImportError as e:
    print(f"  Status  : FAILED — {e}")

# --- Summary ---
print("\n" + "=" * 60)
print("  Verification Complete")
print("=" * 60)
