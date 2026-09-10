"""
WATERSCOPE-AI — Step 2: First YOLO Inference Test
Member 3: AI / Computer Vision Module

Loads the pretrained YOLO26n model and runs object detection
on a test image. Uses CUDA/GPU if available.
"""

import time
import sys
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
IMAGE_PATH = ROOT / "data" / "test_image.jpg"
OUTPUT_PATH = ROOT / "data" / "yolo_test_result.jpg"
MODEL_NAME  = "yolo26n.pt"

print("=" * 65)
print("  WATERSCOPE-AI — Step 2: YOLO Inference Test")
print("=" * 65)

# ── Validate input image ────────────────────────────────────────────────────
if not IMAGE_PATH.exists():
    print(f"\n[ERROR] Test image not found: {IMAGE_PATH}")
    sys.exit(1)

print(f"\n[Input Image]")
size_kb = IMAGE_PATH.stat().st_size / 1024
print(f"  Path : {IMAGE_PATH}")
print(f"  Size : {size_kb:.1f} KB")

# ── Imports ──────────────────────────────────────────────────────────────────
import torch
from ultralytics import YOLO

# ── Device ───────────────────────────────────────────────────────────────────
print(f"\n[Device]")
if torch.cuda.is_available():
    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    print(f"  CUDA Available : True")
    print(f"  GPU            : {gpu_name}")
    print(f"  Using          : CUDA (GPU)")
else:
    device = "cpu"
    print(f"  CUDA Available : False")
    print(f"  Using          : CPU")

# ── Load model ───────────────────────────────────────────────────────────────
print(f"\n[Model]")
print(f"  Loading : {MODEL_NAME}")
t_load_start = time.time()
model = YOLO(MODEL_NAME)
model.to(device)
t_load_end = time.time()
print(f"  Model loaded in {t_load_end - t_load_start:.2f}s")
print(f"  Task   : {model.task}")

# ── Run inference ─────────────────────────────────────────────────────────────
print(f"\n[Inference]")
print(f"  Running detection on: {IMAGE_PATH.name}")

t_infer_start = time.time()
results = model(
    source=str(IMAGE_PATH),
    device=device,
    save=False,     # we'll save manually below
    verbose=False,
)
t_infer_end = time.time()
wall_time_ms = (t_infer_end - t_infer_start) * 1000
print(f"  Wall-clock inference time : {wall_time_ms:.1f} ms")

# ── Parse results ─────────────────────────────────────────────────────────────
result = results[0]

# Inference speed reported by Ultralytics
speed = result.speed  # dict: preprocess, inference, postprocess (ms)
print(f"  Preprocess  : {speed.get('preprocess', 0):.1f} ms")
print(f"  Inference   : {speed.get('inference', 0):.1f} ms")
print(f"  Postprocess : {speed.get('postprocess', 0):.1f} ms")

# Detections
boxes      = result.boxes
names      = result.names          # {class_id: class_name}
num_det    = len(boxes) if boxes is not None else 0

print(f"\n[Detections]")
print(f"  Total detections : {num_det}")
print()

if num_det == 0:
    print("  No objects detected in this image.")
else:
    print(f"  {'#':<4} {'Class':<20} {'Confidence':>10}   {'Bounding Box (x1,y1,x2,y2)'}")
    print(f"  {'-'*4} {'-'*20} {'-'*10}   {'-'*35}")
    for i, box in enumerate(boxes):
        cls_id      = int(box.cls[0].item())
        cls_name    = names[cls_id]
        confidence  = float(box.conf[0].item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        print(f"  {i+1:<4} {cls_name:<20} {confidence:>10.4f}   "
              f"({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")

# ── Save annotated image ──────────────────────────────────────────────────────
print(f"\n[Output]")
annotated = result.plot()   # returns BGR numpy array with boxes drawn

import cv2
cv2.imwrite(str(OUTPUT_PATH), annotated)
print(f"  Annotated image saved : {OUTPUT_PATH}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  Inference Complete")
print("=" * 65)
