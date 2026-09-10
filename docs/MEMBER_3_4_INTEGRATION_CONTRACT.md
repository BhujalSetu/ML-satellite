# WATERSCOPE-AI: Member 3 & Member 4 Integration Contract
**SIH Problem Statement 26015** — *AI + GIS Based Watershed Monitoring & Decision Support System*

This document defines the production integration contract for the **Computer Vision (AI)** and **Satellite Remote Sensing** FastAPI service layer.

---

## A. Endpoint Specification Table

| Method | Route Path | Content-Type | Required Fields | Optional Fields |
|---|---|---|---|---|
| `GET` | `/health` | *None* | *None* | *None* |
| `POST` | `/api/v1/ai/analyze-image` | `multipart/form-data` | `file`, `latitude`, `longitude` | *None* |
| `POST` | `/api/v1/ai/change-detection` | `multipart/form-data` | `before_image`, `after_image` | `latitude`, `longitude`, `mask_file` |
| `POST` | `/api/v1/satellite/analyze` | `application/json` | `latitude`, `longitude` | `buffer`, `start_date`, `end_date`, `max_cloud` |
| `POST` | `/api/v1/satellite/change` | `application/json` | `latitude`, `longitude` | `buffer`, `before_date`, `after_date` |

Static file outputs are hosted at:
- `GET /files/...` (Maps to project `outputs/` directory)

---

## B. Example Requests for All Endpoints

### 1. `POST /api/v1/ai/analyze-image`
**Request (cURL)**:
```bash
curl -X POST "http://localhost:8000/api/v1/ai/analyze-image" \
  -H "Accept: application/json" \
  -F "file=@/path/to/aerial_observation.jpg;type=image/jpeg" \
  -F "latitude=20.4625" \
  -F "longitude=85.8828"
```

### 2. `POST /api/v1/ai/change-detection`
**Request (cURL)**:
```bash
curl -X POST "http://localhost:8000/api/v1/ai/change-detection" \
  -H "Accept: application/json" \
  -F "before_image=@/path/to/site_t0.jpg;type=image/jpeg" \
  -F "after_image=@/path/to/site_t1.jpg;type=image/jpeg" \
  -F "latitude=20.4625" \
  -F "longitude=85.8828"
```

### 3. `POST /api/v1/satellite/analyze`
**Request (JSON)**:
```json
{
  "latitude": 20.4625,
  "longitude": 85.8828,
  "buffer": 0.05,
  "start_date": "2024-02-01",
  "end_date": "2024-03-01",
  "max_cloud": 15.0
}
```

### 4. `POST /api/v1/satellite/change`
**Request (JSON)**:
```json
{
  "latitude": 20.4625,
  "longitude": 85.8828,
  "buffer": 0.05,
  "before_date": "2024-02-21",
  "after_date": "2024-03-02"
}
```

---

## C. Important Multipart Upload Rule
- The AI endpoints (`/api/v1/ai/analyze-image` and `/api/v1/ai/change-detection`) **MUST receive actual binary image files** via `multipart/form-data`.
- Do **NOT** send local host filesystem paths (e.g. `C:\Users\...` or `/tmp/...`).
- Do **NOT** send image URLs as string parameters.
- Accepted formats: standard raster images (JPEG, PNG, WebP).
- Both images in bi-temporal change detection must share matching pixel dimensions `(width, height)`.

---

## D. Important Satellite Parameter Rule
- The satellite endpoints (`/api/v1/satellite/analyze` and `/api/v1/satellite/change`) receive geographic coordinates (`latitude`, `longitude`), optional AOI buffer degree radius, and ISO date constraints.
- Do **NOT** send local raster paths (`.jp2`, `.tif`) or Copernicus scene IDs in request bodies.
- The satellite service dynamically discovers Sentinel-2 Level-2A observations via the Copernicus STAC API for the specified Area of Interest (AOI).
- For temporal change comparisons, `before_date` must be strictly earlier than `after_date` (`before_date < after_date`). Equality is rejected.

---

## E. Response Structure Summary

### 1. `AIAnalyzeImageResponse`
```json
{
  "success": true,
  "request_id": "string (UUID)",
  "image_id": "string",
  "location": { "latitude": 20.4625, "longitude": 85.8828 },
  "model": "yolo26n.pt",
  "detections": [
    {
      "class_name": "boat",
      "confidence": 0.8523,
      "bbox": [10.0, 20.0, 100.0, 150.0]
    }
  ],
  "summary": { "objects_detected": 1, "ponds_detected": 0 },
  "outputs": {
    "annotated_image_url": "/files/ai/img_xxx_annotated.jpg"
  },
  "processing_time_ms": 45.20
}
```

### 2. `AIChangeDetectionResponse`
```json
{
  "success": true,
  "request_id": "string (UUID)",
  "change_detected": true,
  "change_type": "Farm Pond Constructed",
  "confidence": null,
  "change_percentage": 9.77,
  "changes": [
    { "type": "Farm Pond Constructed", "confidence": null }
  ],
  "outputs": {
    "change_map_url": "/files/ai/change_xxx.jpg",
    "annotated_before_url": null,
    "annotated_after_url": null
  },
  "processing_time_ms": 112.40
}
```

### 3. `SatelliteAnalyzeResponse`
```json
{
  "success": true,
  "request_id": "string (UUID)",
  "location": { "latitude": 20.4625, "longitude": 85.8828 },
  "aoi": { "latitude": 20.4625, "longitude": 85.8828, "buffer": 0.05 },
  "scene": {
    "scene_id": "S2A_MSIL2A_20240221T044821_N0510_R076_T45QUC_20240221T075652",
    "date": "2024-02-21",
    "cloud_cover": 0.01,
    "acquisition_datetime": "2024-02-21T04:48:21Z"
  },
  "indices": {
    "ndvi": { "mean": 0.2468, "min": -0.4462, "max": 0.6649, "std": 0.1852 },
    "ndwi": { "mean": -0.2574, "min": -0.6008, "max": 0.4673, "std": 0.1421 }
  },
  "outputs": {
    "ndvi_raster_url": "/files/satellite/ndvi_xxx.tif",
    "ndwi_raster_url": "/files/satellite/ndwi_xxx.tif",
    "water_mask_url": "/files/satellite/water_mask_xxx.tif"
  },
  "processing_time_ms": 320.15
}
```

### 4. `SatelliteChangeResponse`
```json
{
  "success": true,
  "request_id": "string (UUID)",
  "status": "completed",
  "location": { "latitude": 20.4625, "longitude": 85.8828 },
  "aoi": { "latitude": 20.4625, "longitude": 85.8828, "buffer": 0.05 },
  "before": { "date": "2024-02-21", "scene_id": "S2A_MSIL2A_20240221T044821_..." },
  "after": { "date": "2024-03-02", "scene_id": "S2A_MSIL2A_20240302T044711_..." },
  "before_scene": { "date": "2024-02-21", "scene_id": "S2A_MSIL2A_20240221T044821_..." },
  "after_scene": { "date": "2024-03-02", "scene_id": "S2A_MSIL2A_20240302T044711_..." },
  "ndvi_change": {
    "gain_percent": 12.4,
    "loss_percent": 4.1,
    "stable_percent": 83.5
  },
  "ndwi_change": {
    "water_gain_percent": 9.14,
    "water_loss_percent": 0.0,
    "stable_percent": 90.86
  },
  "outputs": {
    "ndvi_change_raster_url": "/files/satellite/ndvi_change_xxx.tif",
    "ndwi_change_raster_url": "/files/satellite/ndwi_change_xxx.tif",
    "water_change_raster_url": "/files/satellite/water_change_xxx.tif",
    "ndvi_classified_raster_url": "/files/satellite/ndvi_classified_xxx.tif",
    "ndwi_classified_raster_url": "/files/satellite/ndwi_classified_xxx.tif",
    "summary_json_url": "/files/satellite/temporal_change_xxx.json"
  },
  "processing_time_ms": 38.56,
  "notice": null
}
```

---

## F. Error Handling & HTTP Status Codes

| Status Code | Reason | Example Trigger |
|---|---|---|
| `400 Bad Request` | Client upload error | Empty image file, non-image payload, or dimension mismatch between T0 and T1 images |
| `404 Not Found` | Resource unavailable | No Sentinel-2 Level-2A scene found for the specified AOI and date range, or no distinct subsequent scene found |
| `422 Unprocessable Content` | Schema / parameter validation error | Latitude out of `[-90, 90]`, longitude out of `[-180, 180]`, buffer `> 5.0`, or `before_date >= after_date` |
| `503 Service Unavailable` | Protected upstream dependency | CDSE full-resolution band download requires registered user authentication (OAuth2/OIDC), or STAC catalog network timeout |
| `500 Internal Server Error` | Pipeline processing exception | Unhandled internal raster/model execution exception (returns sanitised error string, no stack traces) |

---

## G. Output URL Convention
- All output artifacts generated by the API are served through the relative URL path `/files/...`:
  - AI products: `/files/ai/...`
  - Satellite products: `/files/satellite/...`
- The backend and frontend must concatenate these relative paths to the API base URL (e.g. `http://localhost:8000/files/ai/...`).
- No host filesystem paths (`C:\Users\...`, `/var/www/...`) are ever returned in API response JSON.

---

## H. Truthfulness & Scientific Governance Notes
1. **YOLO Model Attribution**: Pretrained `yolo26n.pt` is a generic COCO baseline. Watershed-specific fine-tuning was not completed because the upstream FPCD training annotations (`object_annotations_train_coco.json`) were unavailable in the source repository. No fabricated training metrics or accuracy numbers are reported.
2. **Confidence Governance**: Change detection reports `confidence: null` honestly because the pixel area comparison reflects spatial surface area calculations, not machine learning classification probabilities.
3. **Copernicus CDSE Authentication**: Full-resolution Sentinel-2 JP2 band assets require registered user credentials (OAuth2/OIDC). If remote bands are not locally accessible, the API returns `HTTP 503 Service Unavailable` truthfully instead of substituting synthetic data or unrelated rasters.
4. **Temporal Observation Pairs**: Satellite temporal change detection strictly requires two distinct observations. Identical observations are rejected.
5. **Real Local Observation**: Authentic Sentinel-2 Level-2A data exists for scene `S2A_MSIL2A_20240221T044821_N0510_R076_T45QUC_20240221T075652` covering the verified test area in Odisha (NDVI mean `~0.2468`, NDWI mean `~-0.2574`). It represents a single observation tile, not whole-state totals.
6. **Synthetic Isolation**: Synthetic arrays and mock data are strictly isolated to offline automated test fixtures and are never emitted by production API handlers.
