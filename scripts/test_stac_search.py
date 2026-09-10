"""
WATERSCOPE-AI — Test Copernicus Sentinel-2 STAC Search
Connects to the official Copernicus Data Space STAC API, executes a small AOI search
for Sentinel-2 Level-2A scenes with cloud filtering, prints scene metadata,
audits asset accessibility, and saves structured results to JSON.
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure repo root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.satellite.stac_client import CopernicusSTACClient

def main():
    parser = argparse.ArgumentParser(
        description="WATERSCOPE-AI: Official Copernicus Sentinel-2 STAC Search Test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Small test AOI: ~0.1 deg (~10 km x 10 km) in Odisha (Bhubaneswar / Daya River basin)
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        default=[85.75, 20.20, 85.85, 20.30],
        help="Small test bounding box: min_lon min_lat max_lon max_lat",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2024-01-01",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2024-02-28",
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--max-cloud",
        type=float,
        default=20.0,
        help="Maximum cloud cover percentage (0-100)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of scenes to return",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "outputs" / "satellite" / "stac_search_results.json"),
        help="Output destination for STAC search JSON results",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("  WATERSCOPE-AI — Official Copernicus Sentinel-2 STAC Search Test")
    print("=" * 70)

    client = CopernicusSTACClient()
    print(f"\n[STAC Endpoint Connection]")
    print(f"  Base URL    : {client.base_url}")
    print(f"  Collection  : {client.collection} (Sentinel-2 Level-2A BOA Surface Reflectance)")

    print(f"\n[Search Parameters]")
    print(f"  AOI Bounding Box : {args.bbox} (Small test footprint: ~10km x 10km)")
    print(f"  Date Range       : {args.start_date} to {args.end_date}")
    print(f"  Cloud Cover Max  : <= {args.max_cloud}%")
    print(f"  Result Limit     : {args.limit}")

    print(f"\n[Executing Real STAC Query...]")
    try:
        scenes = client.search(
            bbox=args.bbox,
            date_range=(args.start_date, args.end_date),
            max_cloud_cover=args.max_cloud,
            limit=args.limit,
        )
    except Exception as e:
        print(f"\n[ERROR] STAC search failed: {e}")
        sys.exit(1)

    print(f"  Actual scenes returned: {len(scenes)}")

    if not scenes:
        print("  Notice: No scenes matched the search criteria.")
        return

    print("\n" + "-" * 70)
    print(f"  {'#':<3} {'Scene ID':<62} {'Date / Time (UTC)':<24} {'Cloud %':>7}")
    print("-" * 70)
    for idx, sc in enumerate(scenes):
        print(f"  {idx+1:<3} {sc.scene_id:<62} {sc.datetime:<24} {sc.cloud_cover_pct:>6.2f}%")
    print("-" * 70)

    # Detailed inspection of the best (least cloudy) scene
    best_scene = min(scenes, key=lambda s: s.cloud_cover_pct)
    print(f"\n[Detailed Asset Inspection — Best Scene: {best_scene.scene_id}]")
    print(f"  Acquisition Datetime : {best_scene.datetime}")
    print(f"  Cloud Cover          : {best_scene.cloud_cover_pct}%")
    print(f"  Platform             : {best_scene.platform}")
    print(f"  Thumbnail URL        : {best_scene.thumbnail_url}")

    # Inspect key bands for NDVI / NDWI
    b04 = best_scene.get_band("B04")
    b03 = best_scene.get_band("B03")
    b08 = best_scene.get_band("B08")

    print(f"\n  Identified Key Spectral Bands:")
    print(f"    - B04 (Red 10m)   : {b04.asset_key if b04 else 'MISSING'}")
    if b04:
        print(f"        S3 URI   : {b04.s3_href}")
        print(f"        HTTPS    : {b04.https_href}")
        print(f"        MIME/Size: {b04.mime_type} ({b04.file_size_bytes / (1024*1024):.1f} MB)" if b04.file_size_bytes else "")

    print(f"    - B03 (Green 10m) : {b03.asset_key if b03 else 'MISSING'}")
    if b03:
        print(f"        S3 URI   : {b03.s3_href}")
        print(f"        HTTPS    : {b03.https_href}")
        print(f"        MIME/Size: {b03.mime_type} ({b03.file_size_bytes / (1024*1024):.1f} MB)" if b03.file_size_bytes else "")

    print(f"    - B08 (NIR 10m)   : {b08.asset_key if b08 else 'MISSING'}")
    if b08:
        print(f"        S3 URI   : {b08.s3_href}")
        print(f"        HTTPS    : {b08.https_href}")
        print(f"        MIME/Size: {b08.mime_type} ({b08.file_size_bytes / (1024*1024):.1f} MB)" if b08.file_size_bytes else "")

    # Audit Public Accessibility
    print(f"\n[Asset Accessibility & Authentication Audit]")
    access_audit = client.check_asset_accessibility(best_scene)
    print(f"  Thumbnail Accessible without Auth : {access_audit['thumbnail_accessible']} (HTTP status: {access_audit['details'].get('thumbnail_status')})")
    print(f"  Band Rasters Accessible without Auth : {access_audit['band_assets_accessible_without_auth']} (HTTP status: {access_audit['details'].get('band_https_status')})")
    print(f"  Authentication Required for Bands   : {access_audit['authentication_required']}")
    print(f"  Auth Scheme Required                : {access_audit['auth_type']}")

    if access_audit["authentication_required"]:
        print("\n  [Access Limitation Disclosure]")
        print("  Copernicus Data Space Ecosystem (CDSE) policy requires free registered user authentication")
        print("  (OAuth2 Bearer token) or S3 EODATA credentials to stream/download raw full-resolution .jp2 bands.")
        print("  The STAC catalog search and metadata discovery are 100% open and functional without credentials.")

    # Save results to JSON
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "stac_endpoint": client.base_url,
        "collection": client.collection,
        "search_parameters": {
            "bbox": args.bbox,
            "date_range": [args.start_date, args.end_date],
            "max_cloud_cover_pct": args.max_cloud,
            "limit": args.limit,
        },
        "total_scenes_found": len(scenes),
        "scenes": [s.to_dict() for s in scenes],
        "accessibility_audit": access_audit,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n[Artifacts]")
    print(f"  Saved STAC Search Results : {out_path}")
    print("\n" + "=" * 70)
    print("  STAC Search Test Complete")
    print("=" * 70)

if __name__ == "__main__":
    main()
