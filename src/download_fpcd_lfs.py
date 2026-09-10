"""
WATERSCOPE-AI — FPCD LFS Download
Downloads the actual LFS content for T0.zip and T1.zip using huggingface_hub
which properly handles LFS pointer resolution.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

DEST = Path("data/fpcd/raw")
DEST.mkdir(parents=True, exist_ok=True)

LFS_FILES = [
    ("T0.zip", 227134211),   # 216.6 MB
    ("T1.zip", 202995596),   # 193.6 MB
]

for fname, expected_bytes in LFS_FILES:
    dest_path = DEST / fname
    if dest_path.exists() and dest_path.stat().st_size == expected_bytes:
        print(f"SKIP (already complete): {fname} ({dest_path.stat().st_size/1024/1024:.1f} MB)")
        continue

    print(f"Downloading {fname} ({expected_bytes/1024/1024:.1f} MB via LFS)...", flush=True)
    try:
        # hf_hub_download properly follows LFS pointers
        local = hf_hub_download(
            repo_id="ctundia/FPCD",
            filename=fname,
            repo_type="dataset",
            local_dir=str(DEST),
        )
        actual = Path(local).stat().st_size
        print(f"  Done: {local}")
        print(f"  Size: {actual/1024/1024:.1f} MB (expected {expected_bytes/1024/1024:.1f} MB)")
        if actual != expected_bytes:
            print(f"  WARNING: Size mismatch!")
        else:
            print(f"  Size verified OK.")
    except Exception as e:
        print(f"  ERROR: {e}")

print("\nAll LFS downloads attempted.")
