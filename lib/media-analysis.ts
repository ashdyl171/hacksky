'use client'

export interface MediaAnalysisResult {
  confidence: number
  isDeepfake: boolean
  riskLevel: 'low' | 'medium' | 'high'
  score: number
  details: {
    faceConsistency: number
    temporalConsistency: number
    artifactDetection: number
    lightingAnalysis: number
    metadataAnalysis: number
    compressionArtifacts: number
    motionAnalysis?: number
    audioVisualSync?: number
    pixelPatterns?: number
  }
  recommendations: string[]
  processingTime: number
  fileInfo: {
    name: string
    size: number
    type: string
    dimensions?: { width: number; height: number }
    duration?: number
  }
  transcript?: string
  backendInfo?: {
    prediction: string
    confidence: number
    image_importance: number
    audio_importance: number
    text_importance: number
    transcript: string
  }
}

export interface MediaFile {
  file: File
  preview: string
  type: 'image' | 'video'
  id: string
}

export class MediaAnalysisService {
  static async analyzeMedia(file: File): Promise<MediaAnalysisResult> {
    const startTime = Date.now()
    
    try {
      const formData = new FormData()
      formData.append('file', file)
      formData.append('filename', file.name)
      formData.append('type', file.type)
      
      const response = await fetch('/api/deepfake/analyze', {
        method: 'POST',
        body: formData,
      })
      
      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.statusText}`)
      }
      
      const result = await response.json()
      const processingTime = Date.now() - startTime
      
      // Ensure we have valid analysis results with fallbacks
      const analysisResult: MediaAnalysisResult = {
        confidence: result.analysis?.confidence || 0,
        isDeepfake: result.analysis?.isDeepfake || false,
        riskLevel: result.analysis?.riskLevel || this.getRiskLevel(result.analysis?.confidence || 0),
        score: result.analysis?.confidence || 0,
        details: {
          faceConsistency: result.analysis?.details?.faceConsistency || 0,
          temporalConsistency: result.analysis?.details?.temporalConsistency || 0,
          artifactDetection: result.analysis?.details?.artifactDetection || 0,
          lightingAnalysis: result.analysis?.details?.lightingAnalysis || 0,
          metadataAnalysis: result.analysis?.details?.metadataAnalysis || 0,
          compressionArtifacts: result.analysis?.details?.compressionArtifacts || 0,
        },
        recommendations: result.analysis?.recommendation ? [result.analysis.recommendation] : [],
        processingTime,
        fileInfo: {
          name: file.name,
          size: file.size,
          type: file.type,
          dimensions: result.analysis?.fileInfo?.dimensions,
          duration: result.analysis?.fileInfo?.duration
        },
        // Add transcript if available
        ...(result.analysis?.transcript && {
          transcript: result.analysis.transcript
        }),
        // Add backend information if available
        ...(result.backend && {
          backendInfo: result.backend
        })
      }
      
      return analysisResult
    } catch (error) {
      console.error('Media analysis error:', error)
      
      // Return a safe fallback result instead of throwing
      const processingTime = Date.now() - startTime
      return {
        confidence: 0,
        isDeepfake: false,
        riskLevel: 'low' as const,
        score: 0,
        details: {
          faceConsistency: 0,
          temporalConsistency: 0,
          artifactDetection: 0,
          lightingAnalysis: 0,
          metadataAnalysis: 0,
          compressionArtifacts: 0,
        },
        recommendations: ['Analysis failed - file may be corrupted or unsupported'],
        processingTime,
        fileInfo: {
          name: file.name,
          size: file.size,
          type: file.type,
        }
      }
    }
  }
  
  static async analyzeMultipleFiles(files: File[]): Promise<MediaAnalysisResult[]> {
    const promises = files.map(file => this.analyzeMedia(file))
    return Promise.all(promises)
  }
  
  static getRiskLevel(score: number): 'low' | 'medium' | 'high' {
    if (score >= 70) return 'high'
    if (score >= 30) return 'medium'
    return 'low'
  }
  
  static getRiskColor(riskLevel: 'low' | 'medium' | 'high'): string {
    switch (riskLevel) {
      case 'high': return 'text-red-500'
      case 'medium': return 'text-yellow-500'
      case 'low': return 'text-green-500'
      default: return 'text-gray-500'
    }
  }
  
  static getRiskBgColor(riskLevel: 'low' | 'medium' | 'high'): string {
    switch (riskLevel) {
      case 'high': return 'bg-red-500'
      case 'medium': return 'bg-yellow-500'
      case 'low': return 'bg-green-500'
      default: return 'bg-gray-500'
    }
  }
  
  static formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }
  
  static createFilePreview(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => resolve(reader.result as string)
      reader.onerror = reject
      reader.readAsDataURL(file)
    })
  }
  
  static isValidMediaFile(file: File): boolean {
    const validTypes = [
      'video/mp4', 'video/webm', 'video/ogg', 'video/avi', 'video/quicktime'
    ]
    return validTypes.includes(file.type)
  }
  
  static getFileType(file: File): 'image' | 'video' | 'unknown' {
    if (file.type.startsWith('video/')) return 'video'
    if (file.type.startsWith('image/')) return 'image'
    return 'unknown'
  }
}
