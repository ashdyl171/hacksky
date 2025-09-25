# Deepfake Detection Backend

This is a FastAPI backend for multimodal deepfake detection using a pre-trained model.

## Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have FFmpeg installed on your system:
   - Windows: Download from https://ffmpeg.org/download.html
   - macOS: `brew install ffmpeg`
   - Ubuntu: `sudo apt install ffmpeg`

3. Ensure the model file `deepfake_detector (1).pkl` is in the same directory as `app.py`

## Running the Backend

```bash
python app.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

- `POST /detect` - Upload a video file for deepfake detection
- `GET /health` - Health check endpoint
- `GET /model-info` - Information about loaded models
- `GET /docs` - Interactive API documentation (Swagger UI)

## API Response Format

The `/detect` endpoint returns:
```json
{
  "prediction": "Real" | "Deepfake",
  "confidence": 0.95,
  "image_importance": 0.4,
  "audio_importance": 0.3,
  "text_importance": 0.3,
  "transcript": "Audio transcription text"
}
```

## Integration with Frontend

The frontend will make requests to this backend API for video analysis. The response format has been updated to match the new API structure. 