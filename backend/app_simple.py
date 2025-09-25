"""
Simplified FastAPI backend for Multimodal Deepfake Detection
Demo mode with simulated responses for development
"""

import os
import tempfile
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import random
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Response models
class DetectionResponse(BaseModel):
    prediction: str  # "Real" or "Deepfake"
    confidence: float
    image_importance: float
    audio_importance: float
    text_importance: float
    transcript: str

class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None

# Global variables for demo mode
DEMO_MODE = True

def simulate_deepfake_detection(filename: str):
    """Simulate deepfake detection with random but realistic results"""
    # Simulate processing time
    time.sleep(2)  # 2 second delay to simulate analysis
    
    # Generate random but realistic results
    is_deepfake = random.choice([True, False])
    confidence = random.uniform(0.65, 0.95)
    
    # Simulate component importance scores
    image_importance = random.uniform(0.2, 0.5)
    audio_importance = random.uniform(0.1, 0.3)
    text_importance = random.uniform(0.2, 0.5)
    
    # Generate sample transcript
    sample_transcripts = [
        "Hello, this is a test video for deepfake detection.",
        "The quick brown fox jumps over the lazy dog.",
        "This video contains speech that has been analyzed.",
        "No clear speech detected in this video.",
        "Multiple speakers detected in this audio track."
    ]
    
    transcript = random.choice(sample_transcripts)
    
    return {
        "prediction": "Deepfake" if is_deepfake else "Real",
        "confidence": confidence,
        "image_importance": image_importance,
        "audio_importance": audio_importance,
        "text_importance": text_importance,
        "transcript": transcript
    }

# Initialize FastAPI app
app = FastAPI(
    title="Multimodal Deepfake Detection API",
    description="Demo API for deepfake detection",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Welcome to the Multimodal Deepfake Detection API (Demo Mode)"}

@app.get("/favicon.ico")
async def favicon():
    return FileResponse(os.path.join(os.path.dirname(__file__), "favicon.ico"))

@app.on_event("startup")
async def startup_event():
    """Startup event for demo mode"""
    logger.info("🚀 Starting Deepfake Detection API in DEMO mode")
    logger.info("📝 All predictions will be simulated for development")
    logger.info("✅ API ready to accept requests!")

@app.post("/detect", response_model=DetectionResponse)
async def detect_deepfake(file: UploadFile = File(...)):
    """
    Detect deepfake in uploaded video (DEMO MODE)
    """
    # Validate file type
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.webm')):
        raise HTTPException(
            status_code=400, 
            detail="Only video files (.mp4, .avi, .mov, .webm) are supported"
        )
    
    # Save uploaded file temporarily (but don't actually process it in demo mode)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_video_path = temp_file.name
    
    try:
        # Process video with simulated results
        logger.info(f"Processing video: {file.filename} (DEMO MODE)")
        result = simulate_deepfake_detection(file.filename)
        
        return DetectionResponse(**result)
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "message": "API is running in DEMO mode",
        "mode": "demo",
        "demo_info": "All predictions are simulated for development"
    }

@app.get("/model-info")
async def model_info():
    """Get information about the API mode"""
    return {
        "mode": "demo",
        "message": "Running in demo mode with simulated predictions",
        "supported_formats": [".mp4", ".avi", ".mov", ".webm"],
        "max_file_size": "100MB",
        "processing_time": "~2 seconds (simulated)",
        "note": "All results are randomly generated for development purposes"
    }

if __name__ == "__main__":
    print("🚀 Starting Deepfake Detection API in DEMO mode")
    print("📝 All predictions will be simulated for development")
    print("🌐 Backend will be available at: http://localhost:8000")
    print("📚 API docs available at: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)