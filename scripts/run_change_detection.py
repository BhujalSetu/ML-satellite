"""
WATERSCOPE-AI — Run FPCD Farm Pond Change Detection Pipeline
Processes temporal pairs (T0, T1, multi-class mask), calculates per-class change metrics,
and renders a composite 3-panel dashboard.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ai.fpcd_dataset import FPCDDataset
from src.ai.change_detection import ChangeDetector

def main():
    parser = argparse.ArgumentParser(
        description="WATERSCOPE-AI: FPCD Temporal Change Detection Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pair-idx",
        type=int,
        default=0,
        help="Index of matched pair in FPCD dataset to analyze (0 to 692)",
    )
    parser.add_argument(
        "--t0",
        type=str,
        default=None,
        help="Custom T0 image path (optional)",
    )
    parser.add_argument(
        "--t1",
        type=str,
        default=None,
        help="Custom T1 image path (optional)",
    )
    parser.add_argument(
        "--mask",
        type=str,
        default=None,
        help="Custom mask PNG path (optional)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "outputs" / "change_detection"),
        help="Directory to save visual dashboard and metrics",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Run only the dataset integrity and annotation audit",
    )

    args = parser.parse_args()

    print("=" * 68)
    print("  WATERSCOPE-AI — Farm Pond Change Detection (FPCD)")
    print("=" * 68)

    dataset = FPCDDataset()
    audit = dataset.check_dataset_integrity()

    print(f"\n[Dataset Audit]")
    print(f"  Root Dir                 : {audit['root_dir']}")
    print(f"  T0 Aerial Images Found   : {audit['t0_images_found']}")
    print(f"  T1 Aerial Images Found   : {audit['t1_images_found']}")
    print(f"  Multi-class Masks Found  : {audit['multi_class_masks_found']}")
    print(f"  Complete 3-Way Pairs     : {audit['complete_temporal_pairs']}")
    print(f"  Test COCO Annotations    : {'Available (92 images, 210 annotations)' if audit['test_coco_annotations_present'] else 'Missing'}")
    print(f"  Train COCO Annotations   : {'Available' if audit['train_coco_annotations_present'] else 'NOT FOUND ON DISK'}")
    print(f"  Governance Policy        : {audit['train_annotations_note']}")

    if args.audit_only:
        print("\nAudit completed.")
        return

    # Select paths to analyze
    if args.t0 and args.t1 and args.mask:
        t0_path = args.t0
        t1_path = args.t1
        mask_path = args.mask
        pair_id = Path(mask_path).stem
    else:
        pairs = dataset.get_matched_pairs()
        if not pairs:
            print("[ERROR] No matched FPCD pairs found.")
            sys.exit(1)
        idx = max(0, min(args.pair_idx, len(pairs) - 1))
        p = pairs[idx]
        pair_id = p["pair_id"]
        t0_path = p["t0_path"]
        t1_path = p["t1_path"]
        mask_path = p["mask_path"]
        print(f"\n[Analyzing Matched Pair #{idx}]")
        print(f"  Pair ID : {pair_id}")
        print(f"  T0 Date : {p.get('t0_date', 'N/A')}")
        print(f"  T1 Date : {p.get('t1_date', 'N/A')}")

    detector = ChangeDetector(resolution_meters_per_pixel=1.0)
    result = detector.analyze_temporal_pair(
        t0_path=t0_path,
        t1_path=t1_path,
        mask_path=mask_path,
        output_dir=args.output_dir,
        save_visualization=True,
    )

    print(f"\n[Spatial Change Statistics]")
    print(f"  Dimensions   : {result.image_dimensions[0]} x {result.image_dimensions[1]} ({result.total_pixels:,} px)")
    print(f"  {'ID':<3} {'Class Name':<28} {'Pixels':>10} {'% Area':>8} {'Est. Area (ha)':>15}")
    print(f"  {'-'*3} {'-'*28} {'-'*10} {'-'*8} {'-'*15}")
    for stat in result.category_statistics:
        ha_str = f"{stat.estimated_area_hectares:.4f} ha" if stat.estimated_area_hectares is not None else "N/A"
        print(f"  {stat.class_id:<3} {stat.class_name:<28} {stat.pixel_count:>10,d} {stat.percentage:>7.2f}% {ha_str:>15}")

    # Write output JSON
    out_dir_p = Path(args.output_dir).resolve()
    out_dir_p.mkdir(parents=True, exist_ok=True)
    json_path = out_dir_p / f"{pair_id}_change_stats.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)

    print(f"\n[Artifacts]")
    print(f"  Composite Dashboard  : {result.visualization_path}")
    print(f"  Change Statistics JSON: {json_path}")
    print("\n" + "=" * 68)
    print("  Change Detection Complete")
    print("=" * 68)

if __name__ == "__main__":
    main()
