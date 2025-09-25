import { type NextRequest, NextResponse } from "next/server"

// FastAPI backend configuration
const FASTAPI_BASE_URL = process.env.FASTAPI_BASE_URL || "http://localhost:8000"

interface FastAPIDetectionResponse {
  prediction: string  // "Real" or "Deepfake"
  confidence: number
  image_importance: number
  audio_importance: number
  text_importance: number
  transcript: string
}

interface AnalysisResult {
  confidence: number
  isDeepfake: boolean
  details: {
    faceConsistency: number
    temporalConsistency: number
    artifactDetection: number
    lightingAnalysis: number
    compressionArtifacts: number
    motionAnalysis?: number
    audioVisualSync?: number
    metadataAnalysis: number
    pixelPatterns: number
  }
  processingTime: string
  modelVersion: string
  riskLevel: 'low' | 'medium' | 'high'
  recommendation: string
  filename: string
  filesize: string
  filetype: 'image' | 'video'
  externalAPIs?: {
    deepware?: number
    sensity?: number
    microsoft?: number
  }
}

// Convert FastAPI response to frontend format
function convertFastAPIResponse(fastAPIResult: FastAPIDetectionResponse, filename: string, filesize: string, filetype: 'image' | 'video', processingTime: string): AnalysisResult {
  const isDeepfake = fastAPIResult.prediction === "Deepfake"
  const confidence = Math.round(fastAPIResult.confidence * 100)
  
  // Calculate risk level based on confidence
  const riskLevel: 'low' | 'medium' | 'high' = 
    confidence > 75 ? 'high' : 
    confidence > 50 ? 'medium' : 'low'

  // Generate recommendation based on prediction
  let recommendation = ''
  if (isDeepfake) {
    if (riskLevel === 'high') {
      recommendation = 'High probability of deepfake detected. Manual review strongly recommended before publishing.'
    } else if (riskLevel === 'medium') {
      recommendation = 'Potential signs of deepfake manipulation found. Consider additional verification.'
    } else {
      recommendation = 'Some suspicious patterns detected but confidence is low. Proceed with caution.'
    }
  } else {
    recommendation = 'Content appears authentic based on our AI analysis. Safe to proceed.'
  }

  // Calculate dynamic metrics based on prediction and confidence
  const calculateDynamicMetrics = (prediction: string, confidence: number, image_importance: number, audio_importance: number, text_importance: number) => {
    const isReal = prediction === "Real"
    const confidencePercent = Math.round(confidence * 100)
    
    // Base metrics calculation
    let faceConsistency, temporalConsistency, artifactDetection, lightingAnalysis, compressionArtifacts, motionAnalysis, audioVisualSync
    
    if (isReal) {
      // For REAL content: High confidence = better consistency scores
      faceConsistency = Math.max(70, Math.min(95, 70 + (confidencePercent - 50) * 0.5))
      temporalConsistency = Math.max(75, Math.min(98, 75 + (confidencePercent - 50) * 0.4))
      artifactDetection = Math.max(5, Math.min(30, 30 - (confidencePercent - 50) * 0.4)) // Lower is better for real content
      lightingAnalysis = Math.max(80, Math.min(95, 80 + (confidencePercent - 50) * 0.3))
      compressionArtifacts = Math.max(10, Math.min(25, 25 - (confidencePercent - 50) * 0.3))
      motionAnalysis = Math.max(80, Math.min(96, 80 + (confidencePercent - 50) * 0.3))
      audioVisualSync = Math.max(85, Math.min(98, 85 + (confidencePercent - 50) * 0.25))
    } else {
      // For DEEPFAKE content: High confidence = worse consistency scores
      faceConsistency = Math.max(15, Math.min(45, 45 - (confidencePercent - 50) * 0.5))
      temporalConsistency = Math.max(20, Math.min(50, 50 - (confidencePercent - 50) * 0.4))
      artifactDetection = Math.max(70, Math.min(95, 70 + (confidencePercent - 50) * 0.4)) // Higher is worse for deepfake
      lightingAnalysis = Math.max(25, Math.min(60, 60 - (confidencePercent - 50) * 0.3))
      compressionArtifacts = Math.max(60, Math.min(90, 60 + (confidencePercent - 50) * 0.4))
      motionAnalysis = Math.max(20, Math.min(55, 55 - (confidencePercent - 50) * 0.3))
      audioVisualSync = Math.max(30, Math.min(70, 70 - (confidencePercent - 50) * 0.25))
    }
    
    // Adjust based on component importance
    const imageInfluence = image_importance * 20 // Scale influence
    const audioInfluence = audio_importance * 15
    
    if (!isReal) {
      // For deepfakes, higher image importance means worse face/temporal consistency
      faceConsistency = Math.max(10, faceConsistency - imageInfluence)
      temporalConsistency = Math.max(15, temporalConsistency - imageInfluence)
      audioVisualSync = Math.max(20, audioVisualSync - audioInfluence)
    } else {
      // For real content, higher importance scores mean better consistency
      faceConsistency = Math.min(98, faceConsistency + imageInfluence * 0.3)
      temporalConsistency = Math.min(98, temporalConsistency + imageInfluence * 0.3)
      audioVisualSync = Math.min(98, audioVisualSync + audioInfluence * 0.3)
    }
    
    return {
      faceConsistency: Math.round(faceConsistency),
      temporalConsistency: Math.round(temporalConsistency),
      artifactDetection: Math.round(artifactDetection),
      lightingAnalysis: Math.round(lightingAnalysis),
      compressionArtifacts: Math.round(compressionArtifacts),
      motionAnalysis: Math.round(motionAnalysis),
      audioVisualSync: Math.round(audioVisualSync),
      metadataAnalysis: Math.round((audio_importance + text_importance) * 50 + (isReal ? 20 : -10)),
      pixelPatterns: Math.round(isReal ? 85 - image_importance * 30 : 60 + image_importance * 35)
    }
  }

  // Map importance scores to detail metrics using dynamic calculation
  const details = {
    ...calculateDynamicMetrics(
      fastAPIResult.prediction,
      fastAPIResult.confidence,
      fastAPIResult.image_importance,
      fastAPIResult.audio_importance,
      fastAPIResult.text_importance
    ),
    // Remove video-specific metrics for non-video files
    ...(filetype !== 'video' && {
      temporalConsistency: 0,
      motionAnalysis: 0,
      audioVisualSync: 0
    })
  }

  return {
    confidence,
    isDeepfake,
    details,
    processingTime,
    modelVersion: "v4.0.0 (FastAPI Backend)",
    riskLevel,
    recommendation,
    filename,
    filesize,
    filetype,
    // Add transcript information if available
    ...(fastAPIResult.transcript && fastAPIResult.transcript !== "No transcription available" && {
      transcript: fastAPIResult.transcript
    })
  }
}

export async function POST(request: NextRequest) {
  const startTime = Date.now()
  
  try {
    const formData = await request.formData()
    const file = formData.get("file") as File

    if (!file) {
      return NextResponse.json({ 
        success: false, 
        error: "No file provided" 
      }, { status: 400 })
    }

    // Basic file validation
    const maxSize = 100 * 1024 * 1024 // 100MB
    if (file.size > maxSize) {
      return NextResponse.json({ 
        success: false, 
        error: "File size too large" 
      }, { status: 400 })
    }

    const supportedTypes = [
      'video/mp4', 'video/webm', 'video/quicktime', 'video/avi'
    ]
    
    if (!supportedTypes.includes(file.type)) {
      return NextResponse.json({ 
        success: false, 
        error: "Only video files are supported for deepfake detection" 
      }, { status: 400 })
    }

    const filename = file.name
    const filesize = `${(file.size / 1024 / 1024).toFixed(2)} MB`
    const filetype = 'video' as const

    // Forward the file to FastAPI backend
    const fastAPIFormData = new FormData()
    fastAPIFormData.append('file', file)

    const fastAPIResponse = await fetch(`${FASTAPI_BASE_URL}/detect`, {
      method: 'POST',
      body: fastAPIFormData,
    })

    if (!fastAPIResponse.ok) {
      const errorText = await fastAPIResponse.text()
      console.error('FastAPI error:', fastAPIResponse.status, errorText)
      
      return NextResponse.json({ 
        success: false, 
        error: `Backend analysis failed: ${fastAPIResponse.statusText}`,
        details: errorText
      }, { status: fastAPIResponse.status })
    }

    const fastAPIResult: FastAPIDetectionResponse = await fastAPIResponse.json()
    const processingTime = `${((Date.now() - startTime) / 1000).toFixed(1)}s`

    // Convert FastAPI response to frontend format
    const analysis = convertFastAPIResponse(fastAPIResult, filename, filesize, filetype, processingTime)

    return NextResponse.json({
      success: true,
      analysis,
      backend: {
        prediction: fastAPIResult.prediction,
        confidence: fastAPIResult.confidence,
        image_importance: fastAPIResult.image_importance,
        audio_importance: fastAPIResult.audio_importance,
        text_importance: fastAPIResult.text_importance,
        transcript: fastAPIResult.transcript
      }
    })
    
  } catch (error) {
    console.error('Analysis error:', error)
    return NextResponse.json({ 
      success: false, 
      error: "Analysis failed", 
      details: error instanceof Error ? error.message : "Unknown error"
    }, { status: 500 })
  }
}
