"""
WATERSCOPE-AI — FastAPI Application Foundation
Configures the core FastAPI app, CORS middleware for frontend communication,
health check probe, and mounts the project output directory for static file access.
"""

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .schemas import HealthResponse
from .routes.ai import router as ai_router
from .routes.satellite import router as satellite_router

# Resolve project root safely
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="WATERSCOPE-AI API",
    description="AI + GIS Based Watershed Monitoring & Decision Support System API",
    version="1.0.0",
)

# CORS middleware for development (e.g. React frontend on Vite / localhost:3000 / localhost:5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount outputs directory as API-relative URL path (/files/...)
# Resolves internal files without leaking local absolute host filesystem paths
app.mount("/files", StaticFiles(directory=str(OUTPUTS_DIR)), name="files")

# Register Routers
app.include_router(ai_router)
app.include_router(satellite_router)

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Returns API health status.
    """
    return HealthResponse(
        status="healthy",
        service="WATERSCOPE-AI API",
        version="1.0.0",
    )
