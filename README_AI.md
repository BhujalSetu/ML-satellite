# WATERSCOPE-AI: AI & Remote Sensing Module Documentation

## 1. Overview & Responsibilities
This module implements the core **Computer Vision (AI)** and **Satellite / Remote Sensing** processing engines for **WATERSCOPE-AI** (AI + GIS Based Watershed Monitoring & Decision Support System).

All modules are structured with strict separation of concerns, automated device selection (`cuda` if available and accessible, cleanly falling back to `cpu`), structured outputs (JSON/dataclasses ready for FastAPI integration), and zero fabricated claims.

---

## 2. What Is Implemented

### A. Object Detection Pipeline (`src/ai/`)
- **`config.py`**: Centralized configuration management for model weight paths, dynamic CUDA/CPU device resolution, confidence/IoU thresholds, and explicit domain attribution flags (`generic_coco` vs `watershed_fine_tuned`).
- **`schemas.py`**: Dataclasses defining typed data structures for bounding boxes, detection results, execution timings, and metadata (`BoundingBox`, `DetectionResult`, `InferenceMetadata`, `InferenceOutput`).
- **`preprocessing.py`**: Validated image ingestion (`load_image`), boundary verification (`validate_image`), aspect-ratio preserving letterbox resizing (`resize_image_aspect_ratio`), and float32 normalization (`normalize_image`).
- **`inference.py`**: `WatershedYOLO` inference engine wrapping YOLO26 models. Accurately profiles preprocessing, inference, and postprocessing latencies, annotates images, and explicitly tags pretrained COCO baselines with transparency disclaimers.
- **`visualize.py`**: High-contrast bounding box rendering (`draw_detections`), change mask overlay blending (`draw_change_mask`), and composite 3-panel dashboard generation (`create_temporal_comparison_panel`).

### B. Farm Pond Change Detection (`src/ai/fpcd_dataset.py`, `src/ai/change_detection.py`)
- **Dataset Discovery & Audit**: Discovers and indexes 694 T0 aerial images, 693 T1 aerial images, and 694 multi-class PNG masks on disk.
- **Temporal Pair Matching**: Deterministically links 693 complete 3-way temporal triplets `(T0, T1, Mask)` across Maharashtra villages (e.g. Akola, Amravati, Washim, Yavatmal).
- **Change Category Quantitation**: Evaluates indexed uint8 masks (classes 0–4) to calculate exact pixel counts, area percentages, and metric surface area (m² and hectares).
- **Composite Visual Dashboard**: Generates side-by-side dashboards featuring `[T0 Initial] [T1 Subsequent] [Mask Overlay]` alongside a color-coded legend and quantitative metrics footer.

### C. Satellite Remote Sensing (`src/satellite/`)
- **`ndvi.py`**: Normalized Difference Vegetation Index:
  $$\text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}}$$
  Handles zero-division, invalid denominator masking, vegetation canopy strata metrics (dense, sparse, barren, water proxy), and GeoTIFF output via `rasterio`.
- **`ndwi.py`**: Normalized Difference Water Index (McFeeters 1996 default; Gao 1996 supported):
  $$\text{NDWI} = \frac{\text{Green} - \text{NIR}}{\text{Green} + \text{NIR}}$$
  Calculates water index, water body surface area (m² and hectares), extracts binary water masks (NDWI > 0.0), and exports single-band uint8 GeoTIFFs.
- **`temporal.py`**: Pixel-wise difference and 4-state surface water dynamics:
  - `0`: Persistent Land / Non-Water
  - `1`: Persistent Water
  - `2`: Water Gain (Inundated / Constructed / Filled)
  - `3`: Water Loss (Desiccated / Demolished / Dried)
  - `255`: NoData

### D. Automated Testing (`tests/`)
- 14 automated unit tests covering device resolution, preprocessing, schemas, YOLO inference, FPCD discovery, annotation auditing, mask bounds, change detection metrics, NDVI/NDWI raster math, edge cases, dynamics, and GeoTIFF I/O.

---

## 3. What Is NOT Implemented (Truthfulness & Governance)

1. **No Fine-Tuned Watershed YOLO Model**:
   - The repository currently utilizes pretrained `yolo26n.pt` (trained on generic COCO classes).
   - We have **NOT** yet fine-tuned YOLO on farm ponds. Obtaining 0 detections or generic detections on aerial scenes is expected and normal.
   - **No metrics (mAP, precision, recall, training epochs) have been fabricated.**

2. **No Training on FPCD Test Annotations**:
   - The FPCD dataset contains `object_annotations_test_coco.json` (92 test images, 210 annotations).
   - The upstream training annotation file `object_annotations_train_coco.json` is **missing from disk and not provided in the source repository**.
   - Training on test annotations or fabricating synthetic training annotations is strictly prohibited by project governance.

3. **No Live Satellite API / Earth Engine / SRISHTI-DRISHTI Accounts**:
   - The satellite module currently executes local raster math (GeoTIFF/numpy/rasterio).
   - Live streaming from Sentinel-2 L2A STAC or ISRO SRISHTI-DRISHTI is not connected to active live accounts; the code is architected with clear interfaces ready to ingest real imagery rasters when downloaded or authenticated.

4. **Odisha Waterbodies Dataset Separation**:
   - The Odisha dataset (`data/odisha/waterbodies/`) contains 282,642 GIS polygon features (EPSG:7755). It is vector GIS reference data for mapping and spatial overlay, **NOT** AI training imagery.

---

## 4. Class Definitions

### FPCD Object Detection Classes (4 Classes)
| Class ID | Class Name | Description |
|---|---|---|
| `0` | Wet Farm Pond - Lined | Water-retaining pond with polymer/plastic liner |
| `1` | Wet Farm Pond - Unlined | Water-retaining pond without artificial liner |
| `2` | Dry Farm Pond - Lined | Empty/dry farm pond with visible liner |
| `3` | Dry Farm Pond - Unlined | Empty/dry excavated pit without liner |

### FPCD Temporal Change Detection Classes (5 Classes)
| Class ID | Class Name | Mask Pixel Value | Overlay Color (BGR) |
|---|---|---|---|
| `0` | Background | `0` | Black / Transparent |
| `1` | Farm Pond Constructed | `1` | Green `(0, 200, 0)` |
| `2` | Farm Pond Demolished | `2` | Red `(0, 0, 220)` |
| `3` | Farm Pond Dried | `3` | Orange `(0, 165, 255)` |
| `4` | Farm Pond Wetted | `4` | Cyan / Light Blue `(255, 191, 0)` |

### Satellite Water Dynamics Classes
| Value | Dynamic State | Definition |
|---|---|---|
| `0` | Persistent Land | NDWI $\le$ 0 at T0 and T1 |
| `1` | Persistent Water | NDWI > 0 at T0 and T1 |
| `2` | Water Gain | NDWI $\le$ 0 at T0 $\rightarrow$ NDWI > 0 at T1 |
| `3` | Water Loss | NDWI > 0 at T0 $\rightarrow$ NDWI $\le$ 0 at T1 |
| `255` | NoData | Boundary or invalid pixels |

---

## 5. How to Run

Activate virtual environment:
```powershell
.venv\Scripts\Activate.ps1
```

### 1. Run Unit Tests
```powershell
python -m unittest discover -s tests -v
```

### 2. Run Object Detection Inference
```powershell
python scripts/run_ai_inference.py --image data/test_image.jpg --output outputs/inference/yolo_annotated.jpg
```

### 3. Run FPCD Change Detection
To audit the dataset and analyze a matched temporal pair:
```powershell
# Analyze first matched pair (index 0)
python scripts/run_change_detection.py --pair-idx 0

# Or only run integrity audit
python scripts/run_change_detection.py --audit-only
```

### 4. Run Satellite Remote Sensing Pipeline
```powershell
python scripts/run_satellite_pipeline.py --output-dir outputs/satellite
```

---

## 6. Expected Outputs

All scripts generate structured artifacts in the `outputs/` directory:
- `outputs/inference/`:
  - `yolo_annotated.jpg`: Bounding box visualization
  - `detection_results.json`: JSON output containing bounding boxes, confidence, class names, execution latencies, and attribution notices.
- `outputs/change_detection/`:
  - `{PairID}_comparison.jpg`: 3-panel composite dashboard `[T0 | T1 | Change Mask]` with legend.
  - `{PairID}_change_stats.json`: Per-class pixel counts, percentages, and area estimates in hectares.
- `outputs/satellite/`:
  - `ndvi_t0.tif`, `ndvi_t1.tif`: Single-band GeoTIFFs with vegetation index.
  - `water_mask_t0.tif`, `water_mask_t1.tif`: Binary water surface masks.
  - `water_dynamics.tif`: Categorized temporal change raster.
  - `satellite_analytics_summary.json`: Aggregated metrics and georeferencing metadata.
