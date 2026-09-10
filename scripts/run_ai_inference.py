"""
WATERSCOPE-AI — Run YOLO Object Detection Pipeline
Executes object detection using WatershedYOLO, outputs structured results,
and saves annotated visualization.
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ai.config import AIConfig
from src.ai.inference import WatershedYOLO

def main():
    parser = argparse.ArgumentParser(
        description="WATERSCOPE-AI: YOLO Inference Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image",
        type=str,
        default=str(ROOT / "data" / "test_image.jpg"),
        help="Path to input image for inference",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(ROOT / "yolo26n.pt"),
        help="Path to YOLO weights (.pt file)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for bounding boxes",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Execution device",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "outputs" / "inference" / "yolo_annotated.jpg"),
        help="Destination path for annotated image",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default=str(ROOT / "outputs" / "inference" / "detection_results.json"),
        help="Destination path for structured JSON results",
    )

    args = parser.parse_args()

    print("=" * 65)
    print("  WATERSCOPE-AI — Object Detection Inference")
    print("=" * 65)

    config = AIConfig(
        model_path=args.model,
        device=args.device,
        confidence_threshold=args.conf,
        is_pretrained_baseline=True,
        model_domain="generic_coco",
    )

    print(f"\n[Config]")
    print(f"  Model Path  : {config.model_path}")
    print(f"  Target Device: {args.device} -> Resolved: {config.resolve_device()} ({config.get_device_name()})")
    print(f"  Confidence  : {config.confidence_threshold}")

    print(f"\n[Loading Model...]")
    try:
        engine = WatershedYOLO(config=config)
        print(f"  Model loaded successfully.")
    except Exception as e:
        print(f"  [ERROR] Failed to load model: {e}")
        sys.exit(1)

    print(f"\n[Inference]")
    print(f"  Processing image: {args.image}")
    try:
        output = engine.predict(
            image_input=args.image,
            save_annotated=True,
            output_path=args.output,
        )
    except Exception as e:
        print(f"  [ERROR] Inference failed: {e}")
        sys.exit(1)

    meta = output.metadata
    print(f"\n[Timing]")
    print(f"  Preprocess  : {meta.preprocess_time_ms:.1f} ms")
    print(f"  Inference   : {meta.inference_time_ms:.1f} ms")
    print(f"  Postprocess : {meta.postprocess_time_ms:.1f} ms")
    print(f"  Total Wall  : {meta.total_time_ms:.1f} ms")

    print(f"\n[Detections]")
    print(f"  Total Found : {output.num_detections}")
    if output.num_detections == 0:
        print("  Notice: 0 detections found.")
        print("  Note: Using pretrained generic COCO weights. Pretrained COCO models have not")
        print("        been fine-tuned on aerial watershed imagery, so 0 detections on farm-pond")
        print("        scenes is completely normal and expected.")
    else:
        for idx, det in enumerate(output.detections):
            b = det.bbox
            print(f"  #{idx+1}: {det.class_name} (conf: {det.confidence:.3f}) at [{b.x1:.1f}, {b.y1:.1f}, {b.x2:.1f}, {b.y2:.1f}]")

    if meta.warning:
        print(f"\n[Model Attribution]")
        print(f"  {meta.warning}")

    # Save JSON results
    if args.save_json:
        json_p = Path(args.save_json).resolve()
        json_p.parent.mkdir(parents=True, exist_ok=True)
        with open(json_p, "w", encoding="utf-8") as f:
            json.dump(output.to_dict(), f, indent=2)
        print(f"\n[Artifacts]")
        print(f"  Annotated Image : {output.annotated_image_path}")
        print(f"  Structured JSON : {json_p}")

    print("\n" + "=" * 65)
    print("  Inference Complete")
    print("=" * 65)

if __name__ == "__main__":
    main()
