"""
Simplified FastAPI backend for Multimodal Deepfake Detection
Demo mode with simulated responses for development
"""

import os
import tempfile
import logging
import joblib
import cv2
import librosa
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from transformers import AutoModel, AutoTokenizer, pipeline
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

# Global variables for loaded models
classifier_model = None
resnet_model = None
bert_tokenizer = None
bert_model = None
asr_pipeline = None

# Configuration - UPDATE THIS PATH TO YOUR LOCAL PKL FILE
LOCAL_MODEL_PATH = "/Users/ashleydylan/Documents/python_projects/hth2.0/Hacksky2.0/backend/deepfake_detector1.pkl"  # Model file path

def load_all_models():
    """Load all required models - using LOCAL pkl file"""
    global classifier_model, resnet_model, bert_tokenizer, bert_model, asr_pipeline
    
    try:
        # 1. Load trained classifier from LOCAL pkl file
        if not os.path.exists(LOCAL_MODEL_PATH):
            logger.warning(f"⚠️  Model file not found at: {LOCAL_MODEL_PATH}")
            logger.warning("⚠️  Running in demo mode - predictions will be simulated")
            classifier_model = None
        else:
            classifier_model = joblib.load(LOCAL_MODEL_PATH)
            logger.info(f"✅ Classifier model loaded from: {LOCAL_MODEL_PATH}")
        
        # 2. Load ResNet18 for image features (EXACTLY as in your training)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resnet_model = models.resnet18(weights="ResNet18_Weights.DEFAULT")
        resnet_model = nn.Sequential(*list(resnet_model.children())[:-1])  # Remove final layer
        resnet_model.eval().to(device)
        logger.info("✅ ResNet18 model loaded")
        
        # 3. Load BERT for text features (EXACTLY as in your training)
        bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        bert_model = AutoModel.from_pretrained("bert-base-uncased")
        bert_model.to(device)
        logger.info("✅ BERT model loaded")
        
        # 4. Load ASR pipeline (EXACTLY as in your training)
        asr_pipeline = pipeline("automatic-speech-recognition", 
                               model="facebook/wav2vec2-base-960h")
        logger.info("✅ ASR pipeline loaded")
        
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise HTTPException(status_code=500, detail=f"Model loading failed: {e}")

def extract_frames_exactly_like_training(video_path, output_dir="/tmp/frames"):
    """Extract frames EXACTLY like in your training code"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Default to 30 if fps detection fails
    frame_interval = int(fps)  # Extract 1 frame per second
    count = 0
    frame_list = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if count % frame_interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{count//frame_interval}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_list.append(frame_path)
        count += 1
    
    cap.release()
    return frame_list

def extract_mfcc_exactly_like_training(audio_path):
    """Extract MFCC EXACTLY like in your training code"""
    try:
        y, sr = librosa.load(audio_path, sr=44100)  # Same SR as training
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Same n_mfcc as training
        return np.mean(mfccs, axis=1)  # Same aggregation as training
    except Exception as e:
        logger.error(f"Error processing audio {audio_path}: {e}")
        return np.zeros(13)

def transcribe_audio_exactly_like_training(audio_path):
    """Transcribe audio EXACTLY like in your training code"""
    try:
        transcription = asr_pipeline(audio_path)["text"]
        return transcription if transcription else "No transcription available"
    except Exception as e:
        logger.error(f"Error transcribing {audio_path}: {e}")
        return "No transcription available"

def get_image_embeddings_exactly_like_training(frame_paths):
    """Get image embeddings EXACTLY like in your training code"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeddings = []
    
    for frame_path in frame_paths:
        img = cv2.imread(frame_path)
        if img is None:
            continue
        
        # EXACTLY like your training: resize to 224x224, normalize to [0,1]
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = torch.tensor(img).permute(2, 0, 1).float().unsqueeze(0).to(device)
        
        with torch.no_grad():
            emb = resnet_model(img).flatten()
            embeddings.append(emb.cpu().numpy())
    
    # Return mean of all frame embeddings (512-dim)
    return np.mean(embeddings, axis=0) if embeddings else np.zeros(512)

def get_text_embeddings_exactly_like_training(text):
    """Get text embeddings EXACTLY like in your training code"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        # EXACTLY like training: mean pooling over sequence dimension
        return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

def combine_embeddings_exactly_like_training(image_emb, audio_emb, text_emb):
    """Combine embeddings EXACTLY like in your training code"""
    # Concatenate: 512 (image) + 13 (audio) + 768 (text) = 1293 features
    return np.concatenate([image_emb, audio_emb, text_emb])

def simulate_deepfake_detection(filename):
    """Simulate deepfake detection for demo mode"""
    # Simulate processing time
    time.sleep(2)
    
    # Generate random but realistic results
    prediction = random.choice(["Real", "Deepfake"])
    confidence = random.uniform(0.65, 0.95)
    image_importance = random.uniform(0.2, 0.4)
    audio_importance = random.uniform(0.1, 0.3)
    text_importance = random.uniform(0.3, 0.5)
    
    # Simulate transcript
    sample_transcripts = [
        "This is a sample audio transcript from the video.",
        "The speaker discusses various topics in this recording.",
        "Audio content has been transcribed for analysis.",
        "Speech recognition has processed the audio track."
    ]
    transcript = random.choice(sample_transcripts)
    
    return {
        "prediction": prediction,
        "confidence": confidence,
        "image_importance": image_importance,
        "audio_importance": audio_importance,
        "text_importance": text_importance,
        "transcript": transcript
    }

def predict_deepfake(video_path):
    """Main prediction function matching your exact pipeline"""
    temp_audio_path = None
    temp_frames_dir = None
    
    try:
        # 1. Extract audio from video (EXACTLY like training)
        temp_audio_path = f"/tmp/temp_audio_{os.path.basename(video_path)}.wav"
        os.system(f'ffmpeg -y -i "{video_path}" -vn -acodec pcm_s16le -ar 44100 -ac 1 "{temp_audio_path}"')
        
        # 2. Extract frames (EXACTLY like training)
        temp_frames_dir = "/tmp/frames"
        frame_paths = extract_frames_exactly_like_training(video_path, temp_frames_dir)
        
        # 3. Get embeddings (EXACTLY like training)
        image_emb = get_image_embeddings_exactly_like_training(frame_paths)
        audio_emb = extract_mfcc_exactly_like_training(temp_audio_path)
        text = transcribe_audio_exactly_like_training(temp_audio_path)
        text_emb = get_text_embeddings_exactly_like_training(text)
        
        # 4. Combine embeddings (EXACTLY like training)
        combined_emb = combine_embeddings_exactly_like_training(image_emb, audio_emb, text_emb)
        
        # 5. Make prediction
        if classifier_model is None:
            # Demo mode - return simulated results
            prediction = np.random.choice([0, 1])
            confidence = np.random.uniform(0.6, 0.95)
            image_imp = np.random.uniform(0.2, 0.4)
            audio_imp = np.random.uniform(0.1, 0.3)
            text_imp = np.random.uniform(0.3, 0.5)
            logger.info("⚠️  Using demo mode predictions (model not loaded)")
        else:
            prediction = classifier_model.predict([combined_emb])[0]
            probabilities = classifier_model.predict_proba([combined_emb])[0]
            confidence = float(probabilities[prediction])
            feature_importance = classifier_model.feature_importances_
            
            # Calculate component importances (matching your code)
            image_imp = np.mean(feature_importance[:512])
            audio_imp = np.mean(feature_importance[512:512+13]) 
            text_imp = np.mean(feature_importance[512+13:])
        
        return {
            "prediction": "Deepfake" if prediction == 1 else "Real",
            "confidence": float(confidence),
            "image_importance": float(image_imp),
            "audio_importance": float(audio_imp), 
            "text_importance": float(text_imp),
            "transcript": text
        }
        
    finally:
        # Cleanup
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        if temp_frames_dir and os.path.exists(temp_frames_dir):
            for frame_file in os.listdir(temp_frames_dir):
                os.remove(os.path.join(temp_frames_dir, frame_file))
            os.rmdir(temp_frames_dir)

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
    return {"message": "Welcome to the Multimodal Deepfake Detection API"}

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