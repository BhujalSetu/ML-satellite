# WATERSCOPE-AI

**AI / Computer Vision Module**  
*WATERSCOPE — AI + GIS Based Watershed Monitoring & Decision Support System*

---

## SIH Problem Statement 26015
> Application of Geospatial Techniques for visualization and analysis to interpret Geo-Coded Images to enhance watershed Development Outcomes.

---

## Module Responsibility
**Member 3 — AI / Computer Vision**

---

## Technology Stack
- Python
- OpenCV
- PyTorch
- Ultralytics YOLO
- NumPy
- FastAPI *(planned — later phase)*

---

## Project Structure

```
WATERSCOPE-AI/
├── data/           # Training data, images, annotations
├── models/         # Saved model weights
├── src/            # Source code
│   └── verify_environment.py
├── tests/          # Unit and integration tests
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify environment
python src/verify_environment.py
```
