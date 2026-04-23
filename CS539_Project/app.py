"""FastAPI entrypoint for the Data Analysis Agent.

This module intentionally stays close to HTTP concerns: request validation,
response shaping, artifact serving, and application lifecycle management.
The heavier Gemini orchestration lives in ``src.agent`` so maintainers can
evolve prompts and execution logic without rewriting API handlers.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import os
import json
import time
import uuid
from pathlib import Path
import shutil

from src.agent import DataAnalysisAgent


# ==================== Filesystem Layout ====================

UPLOAD_DIR = Path("uploads")
ARTIFACTS_DIR = Path("artifacts")
OUTPUT_DIR = Path("outputs")
STATIC_DIR = Path("static")

# These folders are created eagerly so handlers can assume they exist.
UPLOAD_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)


# ==================== Request / Response Models ====================

class AnalysisRequest(BaseModel):
    """Request model for analysis endpoint"""
    file_path: str = Field(..., description="Path to the CSV file to analyze")
    question: str = Field(..., description="Natural language query for analysis")
    top_k: Optional[int] = Field(5, description="Number of top results to return")
    save_artifacts: Optional[bool] = Field(True, description="Whether to save analysis artifacts")

    class Config:
        schema_extra = {
            "example": {
                "file_path": "data/sample_data.csv",
                "question": "What is the distribution of sales by region?",
                "top_k": 5,
                "save_artifacts": True
            }
        }


class QuickAnalysisRequest(BaseModel):
    """Request model for quick analysis with file upload"""
    question: str = Field(..., description="Natural language query for analysis")


class MLSolutionRequest(BaseModel):
    """Request model for ML solution generation"""
    question: str = Field(..., description="Machine learning question or task description")
    topic: Optional[str] = Field(None, description="Optional algorithm/topic hint")


class AnalysisResponse(BaseModel):
    """Response model for analysis results"""
    artifact_id: str
    query: str
    success: bool
    summary: str
    analysis_plan: List[Dict[str, Any]]
    steps_executed: int
    visualizations: List[str]
    evaluation: Optional[Dict[str, Any]] = None
    latency: float
    artifact_path: Optional[str] = None
    generated_code: Optional[str] = None
    explanation: Optional[str] = None
    libraries: Optional[List[str]] = None


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    version: str
    agent_initialized: bool
    detail: Optional[str] = None


class ErrorResponse(BaseModel):
    """Response model for errors"""
    error: str
    detail: Optional[str] = None


# ==================== Application Factory ====================

def create_app() -> FastAPI:
    """
    FastAPI application factory
    
    Returns:
        Configured FastAPI application instance
    """
    app = FastAPI(
        title="Data Analysis Agent API",
        description="RESTful API for automated exploratory data analysis using LLM-based agent",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # The project currently ships a local browser UI, so permissive CORS keeps
    # development simple. Tighten this before deploying to a shared environment.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # The API can run without the static app, but serving it here keeps local
    # development and demos self-contained.
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    
    # The agent is shared for the lifetime of the process so model setup and
    # visualization tooling do not need to be rebuilt on every request.
    app.state.agent_init_error = None
    try:
        app.state.agent = DataAnalysisAgent()
        app.state.agent_initialized = True
    except Exception as e:
        error_message = str(e)
        print(f"Warning: Could not initialize agent: {error_message}")
        app.state.agent = None
        app.state.agent_initialized = False
        app.state.agent_init_error = error_message
    
    return app


# ==================== Application Instance ====================

app = create_app()


# ==================== Helper Functions ====================

def _ensure_agent_initialized() -> bool:
    """Lazily initialize the shared agent if startup initialization failed."""
    if app.state.agent_initialized and app.state.agent is not None:
        return True

    try:
        app.state.agent = DataAnalysisAgent()
        app.state.agent_initialized = True
        app.state.agent_init_error = None
        return True
    except Exception as e:
        app.state.agent = None
        app.state.agent_initialized = False
        app.state.agent_init_error = str(e)
        return False

def save_analysis_artifact(analysis_id: str, results: Dict[str, Any]) -> str:
    """Persist a JSON artifact so the browser can download a stable snapshot."""
    artifact_path = ARTIFACTS_DIR / f"{analysis_id}.json"
    with open(artifact_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    return str(artifact_path)


def cleanup_old_files(directory: Path, max_age_hours: int = 24):
    """Remove old generated files to keep local development directories tidy."""
    current_time = time.time()
    for file_path in directory.glob("*"):
        if file_path.is_file():
            file_age_hours = (current_time - file_path.stat().st_mtime) / 3600
            if file_age_hours > max_age_hours:
                file_path.unlink()


# ==================== API Endpoints ====================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - serves the web interface"""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    else:
        # Fallback to API information if static files not available
        return {
            "service": "Data Analysis Agent API",
            "version": "1.0.0",
            "status": "operational",
            "endpoints": {
                "health": "/health",
                "analyze": "/analyze",
                "upload_and_analyze": "/upload-analyze",
                "docs": "/docs"
            }
        }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint
    
    Returns:
        Health status and agent initialization state
    """
    return HealthResponse(
        status="healthy" if app.state.agent_initialized else "degraded",
        version="1.0.0",
        agent_initialized=app.state.agent_initialized,
        detail=app.state.agent_init_error
    )


@app.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_dataset(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """
    Analyze a dataset with a natural language query
    
    Args:
        request: Analysis request with file path and question
        background_tasks: FastAPI background tasks for cleanup
        
    Returns:
        Analysis results with summary, visualizations, and evaluation
        
    Raises:
        HTTPException: If agent not initialized or analysis fails
    """
    if not _ensure_agent_initialized():
        raise HTTPException(
            status_code=503,
            detail=(
                "Agent not initialized. "
                f"{app.state.agent_init_error or 'Please check API key configuration.'}"
            )
        )
    
    # Validate file exists
    if not os.path.exists(request.file_path):
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {request.file_path}"
        )
    
    # Generate unique analysis ID
    analysis_id = str(uuid.uuid4())
    
    # Measure latency
    start_time = time.time()
    
    try:
        # Run analysis
        results = app.state.agent.analyze(request.file_path, request.question)
        
        if not results.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=f"Analysis failed: {results.get('error', 'Unknown error')}"
            )
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Save artifacts if requested
        artifact_path = None
        if request.save_artifacts:
            artifact_path = save_analysis_artifact(analysis_id, results)
        
        # Schedule cleanup of old files
        background_tasks.add_task(cleanup_old_files, ARTIFACTS_DIR, max_age_hours=24)
        background_tasks.add_task(cleanup_old_files, OUTPUT_DIR, max_age_hours=24)
        
        # Extract visualization filenames from the results
        viz_files = []
        for viz in results.get("visualizations", []):
            if isinstance(viz, dict) and "file_path" in viz:
                file_path = viz["file_path"]
                viz_files.append(os.path.basename(str(file_path)))
            elif isinstance(viz, str):
                viz_files.append(os.path.basename(viz))
        
        return AnalysisResponse(
            artifact_id=analysis_id,
            query=request.question,
            success=True,
            summary=results["summary"],
            analysis_plan=results.get("analysis_plan", []),
            steps_executed=len(results.get("steps", [])),
            visualizations=viz_files,
            evaluation=None,
            latency=round(latency_ms / 1000, 2),
            artifact_path=artifact_path,
            generated_code=results.get("generated_code")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal error during analysis: {str(e)}"
        )


@app.post("/upload-analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def upload_and_analyze(
    file: UploadFile = File(...),
    query: str = Form(..., description="Analysis question"),
    background_tasks: BackgroundTasks = None
):
    """
    Upload a CSV file and analyze it in one request
    
    Args:
        file: CSV file upload
        question: Natural language query
        background_tasks: FastAPI background tasks
        
    Returns:
        Analysis results
        
    Raises:
        HTTPException: If file type invalid or analysis fails
    """
    if not _ensure_agent_initialized():
        raise HTTPException(
            status_code=503,
            detail=(
                "Agent not initialized. "
                f"{app.state.agent_init_error or 'Please check API key configuration.'}"
            )
        )
    
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported"
        )
    
    # Generate unique filename
    upload_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{upload_id}_{file.filename}"
    
    try:
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Create analysis request
        request = AnalysisRequest(
            file_path=str(file_path),
            question=query,
            save_artifacts=True
        )
        
        # Perform analysis
        response = await analyze_dataset(request, background_tasks)
        
        # Schedule cleanup of uploaded file
        if background_tasks:
            background_tasks.add_task(cleanup_old_files, UPLOAD_DIR, max_age_hours=1)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        # Clean up file on error
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(
            status_code=500,
            detail=f"Error processing upload: {str(e)}"
        )


@app.get("/artifact/{analysis_id}", tags=["Artifacts"])
async def get_artifact(analysis_id: str):
    """
    Retrieve saved analysis artifact by ID
    
    Args:
        analysis_id: UUID of the analysis
        
    Returns:
        JSON artifact file
        
    Raises:
        HTTPException: If artifact not found
    """
    artifact_path = ARTIFACTS_DIR / f"{analysis_id}.json"
    
    if not artifact_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Artifact not found: {analysis_id}"
        )
    
    return FileResponse(
        artifact_path,
        media_type="application/json",
        filename=f"analysis_{analysis_id}.json"
    )


@app.get("/visualization/{filename}", tags=["Visualizations"])
async def get_visualization(filename: str):
    """
    Retrieve generated visualization image
    
    Args:
        filename: Name of the visualization file
        
    Returns:
        PNG image file
        
    Raises:
        HTTPException: If visualization not found
    """
    safe_filename = os.path.basename(filename)
    viz_path = OUTPUT_DIR / safe_filename
    
    if not viz_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Visualization not found: {filename}"
        )
    
    return FileResponse(
        viz_path,
        media_type="image/png",
        filename=safe_filename
    )


@app.post("/ml-solution", response_model=AnalysisResponse, tags=["ML"])
async def ml_solution(
    request: MLSolutionRequest,
    background_tasks: BackgroundTasks = None
):
    """
    Generate and execute an ML solution from a natural-language question.

    No file upload required. Gemini generates self-contained Python code
    that is executed on the server; any matplotlib figures are captured
    and returned as visualization URLs.

    Args:
        request: ML question and optional algorithm/topic hint.

    Returns:
        AnalysisResponse with explanation, generated code, and visualizations.
    """
    if not _ensure_agent_initialized():
        raise HTTPException(
            status_code=503,
            detail=(
                "Agent not initialized. "
                f"{app.state.agent_init_error or 'Please check API key configuration.'}"
            )
        )

    analysis_id = str(uuid.uuid4())
    start_time = time.time()

    try:
        results = app.state.agent.generate_ml_solution(
            request.question, request.topic
        )

        latency_ms = (time.time() - start_time) * 1000

        if not results.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=f"ML solution failed: {results.get('error', 'Unknown error')}"
            )

        viz_files = []
        for viz in results.get("visualizations", []):
            if isinstance(viz, dict) and "file_path" in viz:
                viz_files.append(os.path.basename(str(viz["file_path"])))
            elif isinstance(viz, str):
                viz_files.append(os.path.basename(viz))

        if background_tasks:
            background_tasks.add_task(cleanup_old_files, OUTPUT_DIR, max_age_hours=24)

        return AnalysisResponse(
            artifact_id=analysis_id,
            query=request.question,
            success=True,
            summary=results.get("summary", ""),
            analysis_plan=results.get("steps", []),
            steps_executed=len(results.get("steps", [])),
            visualizations=viz_files,
            latency=round(latency_ms / 1000, 2),
            generated_code=results.get("generated_code", ""),
            explanation=results.get("explanation", ""),
            libraries=results.get("libraries", []),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal error during ML solution generation: {str(e)}"
        )


@app.delete("/cleanup", tags=["Maintenance"])
async def cleanup_all(max_age_hours: int = 24):
    """
    Manually trigger cleanup of old files
    
    Args:
        max_age_hours: Maximum age of files to keep
        
    Returns:
        Cleanup summary
    """
    cleanup_old_files(UPLOAD_DIR, max_age_hours)
    cleanup_old_files(ARTIFACTS_DIR, max_age_hours)
    cleanup_old_files(OUTPUT_DIR, max_age_hours)
    
    return {
        "status": "cleanup_completed",
        "max_age_hours": max_age_hours,
        "directories_cleaned": ["uploads", "artifacts", "outputs"]
    }


# ==================== Error Handlers ====================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Custom 404 handler"""
    return JSONResponse(
        status_code=404,
        content={"error": "Not Found", "detail": str(exc.detail) if hasattr(exc, 'detail') else "Resource not found"}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Custom 500 handler"""
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "detail": "An unexpected error occurred"}
    )


# ==================== Startup/Shutdown Events ====================

@app.on_event("startup")
async def startup_event():
    """Application startup tasks"""
    print("=" * 70)
    print("Data Analysis Agent API - Starting Up")
    print("=" * 70)
    print(f"Agent initialized: {app.state.agent_initialized}")
    print(f"Upload directory: {UPLOAD_DIR.absolute()}")
    print(f"Artifacts directory: {ARTIFACTS_DIR.absolute()}")
    print(f"Output directory: {OUTPUT_DIR.absolute()}")
    print(f"Interactive docs: http://localhost:8000/docs")
    print("=" * 70)


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown tasks"""
    print("\nData Analysis Agent API - Shutting Down")


# ==================== Main ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
