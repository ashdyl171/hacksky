# Deepfake Detection System Setup Guide

This guide will help you set up the complete deepfake detection system with FastAPI backend and Next.js frontend.

## Prerequisites

1. **Python 3.8+** - Download from [python.org](https://python.org)
2. **Node.js 18+** - Download from [nodejs.org](https://nodejs.org)
3. **FFmpeg** - Required for video processing
   - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH
   - macOS: `brew install ffmpeg`
   - Ubuntu: `sudo apt install ffmpeg`

## Backend Setup (FastAPI)

### 1. Navigate to Backend Directory
```bash
cd backend
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Model File
Ensure the model file `deepfake_detector (1).pkl` is in the backend directory.

### 4. Start the Backend
```bash
# Windows
start.bat

# Or manually
python app.py
```

The backend will be available at `http://localhost:8000`

### 5. Test the Backend
Visit `http://localhost:8000/docs` to see the interactive API documentation.

## Frontend Setup (Next.js)

### 1. Install Dependencies
```bash
npm install
# or
pnpm install
```

### 2. Start the Development Server
```bash
npm run dev
# or
pnpm dev
```

The frontend will be available at `http://localhost:3000`

## Usage

1. **Start both servers**:
   - Backend: `http://localhost:8000`
   - Frontend: `http://localhost:3000`

2. **Upload videos** for deepfake detection:
   - Supported formats: MP4, WebM, AVI, MOV, OGG
   - Maximum file size: 100MB

3. **View results**:
   - Real-time analysis with confidence scores
   - Detailed breakdown of image, audio, and text importance
   - Audio transcript (if available)
   - Risk assessment and recommendations

## API Endpoints

### Backend (FastAPI)
- `POST /detect` - Upload video for deepfake detection
- `GET /health` - Health check
- `GET /model-info` - Model information
- `GET /docs` - Interactive documentation

### Frontend (Next.js)
- `POST /api/deepfake/analyze` - Proxy to FastAPI backend

## Response Format

The system returns detailed analysis including:
- **Prediction**: "Real" or "Deepfake"
- **Confidence**: 0-100% confidence score
- **Importance Scores**: Image, audio, and text importance
- **Transcript**: Audio transcription (if available)
- **Risk Level**: Low, medium, or high
- **Recommendations**: Actionable advice

## Troubleshooting

### Backend Issues
1. **Model not found**: Ensure `deepfake_detector (1).pkl` is in the backend directory
2. **FFmpeg error**: Install FFmpeg and add to system PATH
3. **Dependencies error**: Run `pip install -r requirements.txt`

### Frontend Issues
1. **Backend connection error**: Ensure backend is running on port 8000
2. **File upload error**: Check file format and size limits
3. **Analysis timeout**: Large files may take longer to process

### Performance Tips
1. Use smaller video files for faster analysis
2. Ensure sufficient RAM (8GB+ recommended)
3. GPU acceleration available for faster processing

## Development

### Backend Development
- Edit `backend/app.py` for API changes
- Add new endpoints in the FastAPI app
- Update model path in `LOCAL_MODEL_PATH`

### Frontend Development
- Edit components in `components/` directory
- Update API integration in `app/api/deepfake/analyze/route.ts`
- Modify UI in `components/media-upload-analysis.tsx`

## Security Notes

- The backend processes files locally
- No data is sent to external services
- Temporary files are cleaned up automatically
- Model file should be kept secure

## Support

For issues or questions:
1. Check the troubleshooting section
2. Verify all prerequisites are installed
3. Check console logs for error messages
4. Ensure both servers are running 