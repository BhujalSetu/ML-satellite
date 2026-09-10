# WATERSCOPE-AI: Demo Evidence & Data Inventory
**SIH Problem Statement 26015** — *AI + GIS Based Watershed Monitoring & Decision Support System*

This document provides a comprehensive inventory of real data artifacts, model outputs, and verified evidence available for demonstration during the Smart India Hackathon (SIH) evaluation.

---

## 1. AI Object & Feature Detection (Member 3)

### Endpoint
- `POST /api/v1/ai/analyze-image`

### Demonstration Workflow
1. **Input Payload**: An aerial or field photograph uploaded as `multipart/form-data` with location metadata:
   - File: e.g. `data/test_image.jpg` or any user-uploaded field drone image
   - Latitude: `20.4625`
   - Longitude: `85.8828`
2. **Model Inference**:
   - Executes real inference using the loaded YOLO engine (`WatershedYOLO`) with PyTorch.
   - Automatically selects hardware acceleration (`CUDA` on RTX GPU if available, otherwise optimized `CPU`).
   - Runs genuine inference without mock data or simulated detections.
3. **Generated Artifacts**:
   - Annotated visualization: `outputs/ai/img_<id>_annotated.jpg`
   - JSON response: includes bounding boxes `[x1, y1, x2, y2]`, class labels, individual confidences, model name (`yolo26n.pt`), and latency timing.
   - API-relative URL: `/files/ai/img_<id>_annotated.jpg` (servable directly to web frontend).
4. **Attribution & Transparency**:
   - Model weights are `yolo26n.pt` (generic COCO pretrained baseline).
   - Only genuine water pond classes (`"pond"`, `"farm_pond"`) increment the `ponds_detected` summary metric; generic background classes are reported honestly.

---

## 2. AI Bi-Temporal Farm Pond Change Detection (Member 3)

### Endpoint
- `POST /api/v1/ai/change-detection`

### Demonstration Workflow
1. **Input Payload**: Two co-registered temporal observations (T0 before vs T1 after) uploaded as `multipart/form-data`:
   - `before_image`: e.g. T0 aerial image from the FPCD dataset (`data/fpcd/`)
   - `after_image`: e.g. T1 aerial image from the FPCD dataset
   - Optional location: `latitude=20.4625`, `longitude=85.8828`
2. **Analysis & Metrics**:
   - Evaluates multi-temporal pixel reflectance differences and multi-class change masks using standard FPCD categories:
     - Class 0: Background
     - Class 1: Farm Pond Constructed (excavation / filling)
     - Class 2: Farm Pond Demolished (drying / filling in)
     - Class 3: Farm Pond Dried
     - Class 4: Farm Pond Wetted
   - Computes exact pixel counts, surface change percentages, and physical metric areas (hectares).
   - Identical images produce `change_detected: false` and `change_percentage: 0.0%` with zero fabricated change.
3. **Generated Artifacts**:
   - 3-Panel Composite Comparison Dashboard: `outputs/ai/change_<request_id>.jpg`
     - Features `[T0 Initial Observation] [T1 Subsequent Observation] [Classified Change Mask Overlay]` with color-coded legend and quantitative statistics banner.
   - API-relative URL: `/files/ai/change_<request_id>.jpg` (served via `GET /files/...`).
4. **Honest Confidence Reporting**:
   - `confidence` is returned as `null` because physical area difference calculations derive directly from pixel arithmetic and spatial ground-sampling distance, not machine learning classification probabilities.

---

## 3. Real Sentinel-2 Remote Sensing Evidence (Member 4)

### Endpoint
- `POST /api/v1/satellite/analyze`

### Authenticated Real Data Observation
- **Sentinel-2 L2A Product ID**: `S2A_MSIL2A_20240221T044821_N0510_R076_T45QUC_20240221T075652`
- **Acquisition Date**: February 21, 2024 (`2024-02-21`)
- **Cloud Cover**: `0.01%`
- **Spectral Bands Processed**:
  - Band 03 (Green, 560 nm, 10 m resolution)
  - Band 04 (Red, 665 nm, 10 m resolution)
  - Band 08 (NIR, 842 nm, 10 m resolution)

### Verified Physical Raster Products
Located in `outputs/satellite/real/`:
1. **`ndvi_real.tif`** (530,958,886 bytes, 10,980 × 10,980 pixels, EPSG:32645):
   - $\text{NDVI} = (\text{B08} - \text{B04}) / (\text{B08} + \text{B04})$
   - Valid Mean: `0.2468`
   - Minimum: `-0.4462`
   - Maximum: `0.6649`
2. **`ndwi_real.tif`** (522,005,940 bytes, 10,980 × 10,980 pixels, EPSG:32645):
   - $\text{NDWI} = (\text{B03} - \text{B08}) / (\text{B03} + \text{B08})$
   - Valid Mean: `-0.2574`
   - Minimum: `-0.6008`
   - Maximum: `0.4673`
3. **`water_mask_real.tif`** (2,504,882 bytes):
   - Classified binary water mask ($\text{NDWI} > 0.0$).
4. **`real_satellite_analytics_summary.json`**:
   - Contains canopy classification (dense vegetation, sparse vegetation, barren soil, water proxy) and surface area quantitation in hectares.

> **Important Presentation Note**: These statistics represent **one authentic Sentinel-2 Level-2A observation tile (Tile 45QUC)** over Odisha, **not state-wide totals**.

---

## 4. Satellite Multi-Temporal Change Detection (Member 4)

### Endpoint
- `POST /api/v1/satellite/change`

### Implemented Capability vs. Data Boundary
1. **Implemented Engine**:
   - Full bi-temporal raster processing module (`src/satellite/temporal_change.py`):
     - Pair alignment validation (`validate_temporal_pair`) verifying CRS, affine transform, pixel dimensions, and resolution.
     - Difference arithmetic: $\Delta\text{NDVI} = \text{NDVI}_{T1} - \text{NDVI}_{T0}$ and $\Delta\text{NDWI} = \text{NDWI}_{T1} - \text{NDWI}_{T0}$.
     - Heuristic classification: Vegetation Loss ($<-0.10$), Stable ($[-0.10, +0.10]$), Gain ($>+0.10$).
     - Surface water dynamics: Persistent Land, Persistent Water, Water Gain (Inundation/Filling), Water Loss (Drying/Desiccation).
     - Export to LZW-compressed GeoTIFFs and JSON summary reports.
2. **Truthful Authentication Boundary**:
   - When a user requests temporal comparison against an observation whose raw JP2 bands reside remotely on the Copernicus Data Space Ecosystem (CDSE) requiring user login, the API returns **`HTTP 503 Service Unavailable`** with an honest explanatory message.
   - The system **never fabricates fake temporal rasters or fake percentage numbers** when second-observation assets are unauthenticated.

---

## 5. Odisha Waterbody GIS Reference Dataset (Member 1 & 2 Context)

### Evidence Location
- `data/odisha/waterbodies/`

### Dataset Inventory & Specifications
- **Total Features**: `282,642` waterbody polygon features
- **Coordinate Reference Systems**: Projected `EPSG:7755` and Geographic `EPSG:4326` (WGS84)
- **Formats Available**:
  - `odisha_waterbodies_clean.parquet` / `.geoparquet` (Optimized column-oriented spatial storage)
  - `odisha_waterbodies_clean.geojson` (Standard spatial vector exchange)
- **Geometry Validity**: 100% valid WGS84 polygon and multipolygon geometries with zero unclosed rings or topology inversions.
- **Role in Demonstration**: Provides authentic GIS vector reference layers for basemap overlays, district/block boundaries, and spatial intersection against AI and satellite detections. It is **reference GIS data, not AI training imagery**.

---

## 6. Demonstration Script Quick Reference

| Feature to Demonstrate | Recommended Call | Key Highlight |
|---|---|---|
| **System Health** | `GET /health` | Instant operational status probe (`1.0.0`) |
| **Object Detection** | `POST /api/v1/ai/analyze-image` with field photo | Real YOLO execution, bounding boxes, honest COCO baseline attribution |
| **Pond Change Detection** | `POST /api/v1/ai/change-detection` with T0 & T1 images | 3-panel comparison visual dashboard, metric hectares, honest `confidence=null` |
| **Satellite Indices** | `POST /api/v1/satellite/analyze` with `(20.4625, 85.8828)` | Real Sentinel-2 L2A tile processing, verified NDVI `0.2468`, NDWI `-0.2574` |
| **CDSE Governance** | `POST /api/v1/satellite/analyze` (remote date) | Truthful `503 Service Unavailable` on unauthenticated CDSE assets |
| **Temporal Dynamics** | `POST /api/v1/satellite/change` (dates T0 & T1) | Dual-observation validation, water gain/loss raster, difference GeoTIFFs |
| **Static Raster Serving** | `GET /files/satellite/<raster>.tif` | Browser/GIS accessible GeoTIFF via API-relative paths, zero host path leaks |
