"""
WATERSCOPE-AI — Step 4: FPCD Dataset Downloader
Downloads FPCD dataset files directly from HuggingFace public URLs.
"""

import os
import sys
import requests
from pathlib import Path

BASE_URL = "https://huggingface.co/datasets/ctundia/FPCD/resolve/main"
DEST_DIR = Path("data/fpcd/raw")
DEST_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    "T0.zip",
    "T1.zip",
    "object_annotations_test_coco.json",
    "cd_dataset_train.txt",
    "cd_dataset_test.txt",
]

def download_file(filename):
    url = f"{BASE_URL}/{filename}"
    dest = DEST_DIR / filename
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"  SKIP (already exists): {filename} ({dest.stat().st_size/1024/1024:.1f} MB)")
        return True

    print(f"  Downloading: {filename}", flush=True)
    try:
        r = requests.get(url, stream=True, timeout=60,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024*512):  # 512KB chunks
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        mb = downloaded / 1024 / 1024
                        print(f"\r    {pct:5.1f}%  {mb:.1f}/{total/1024/1024:.1f} MB", end="", flush=True)
        print(f"\n  Done: {filename} ({dest.stat().st_size/1024/1024:.1f} MB)")
        return True
    except Exception as e:
        print(f"\n  ERROR downloading {filename}: {e}")
        return False

print("=" * 55)
print("  WATERSCOPE-AI — FPCD Dataset Download")
print("=" * 55)

success = True
for f in FILES:
    ok = download_file(f)
    success = success and ok

print()
print("=" * 55)
if success:
    print("  All files downloaded successfully.")
else:
    print("  WARNING: Some files failed. Check errors above.")
print("=" * 55)
