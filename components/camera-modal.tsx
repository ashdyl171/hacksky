'use client'

import { useState, useRef, useCallback, useEffect } from 'react'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent } from '@/components/ui/card'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Camera, Square, RotateCcw, Upload, Shield, AlertTriangle, CheckCircle, X, Download } from 'lucide-react'
import { MediaAnalysisService, MediaAnalysisResult } from '@/lib/media-analysis'

interface CameraModalProps {
  isOpen: boolean
  onClose: () => void
  onPhotoTaken?: (photo: { file: File; analysis: MediaAnalysisResult }) => void
}

export function CameraModal({ isOpen, onClose, onPhotoTaken }: CameraModalProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  
  const [isRecording, setIsRecording] = useState(false)
  const [capturedImage, setCapturedImage] = useState<string | null>(null)
  const [capturedFile, setCapturedFile] = useState<File | null>(null)
  const [facingMode, setFacingMode] = useState<'user' | 'environment'>('user')
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [analysisResult, setAnalysisResult] = useState<MediaAnalysisResult | null>(null)
  const [showAuthDialog, setShowAuthDialog] = useState(false)
  const [authDecision, setAuthDecision] = useState<'pending' | 'approved' | 'rejected'>('pending')

  const startCamera = useCallback(async () => {
    try {
      const constraints = {
        video: {
          facingMode,
          width: { ideal: 1920 },
          height: { ideal: 1080 },
        },
        audio: false,
      }

      const stream = await navigator.mediaDevices.getUserMedia(constraints)
      streamRef.current = stream
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        videoRef.current.play()
        setIsRecording(true)
      }
    } catch (error) {
      console.error('Error accessing camera:', error)
      alert('Unable to access camera. Please check permissions.')
    }
  }, [facingMode])

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }
    setIsRecording(false)
  }, [])

  const capturePhoto = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current) return

    const video = videoRef.current
    const canvas = canvasRef.current
    const context = canvas.getContext('2d')

    if (!context) return

    // Set canvas dimensions to match video
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight

    // Draw current video frame to canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height)

    // Convert canvas to blob and create file
    canvas.toBlob(async (blob) => {
      if (!blob) return

      const file = new File([blob], `photo_${Math.random().toString(36).substr(2, 9)}.jpg`, { type: 'image/jpeg' })
      const imageUrl = URL.createObjectURL(blob)

      setCapturedImage(imageUrl)
      setCapturedFile(file)
      
      // Start AI analysis
      setIsAnalyzing(true)
      try {
        const analysis = await MediaAnalysisService.analyzeMedia(file)
        setAnalysisResult(analysis)
        
        // Check if content needs authorization
        if (analysis.isDeepfake || analysis.score > 30 || analysis.riskLevel === 'high' || analysis.riskLevel === 'medium') {
          setShowAuthDialog(true)
        } else {
          // Auto-approve low-risk content
          setAuthDecision('approved')
        }
      } catch (error) {
        console.error('Analysis failed:', error)
        // On analysis failure, still allow upload but with warning
        setAuthDecision('approved')
      } finally {
        setIsAnalyzing(false)
      }
    }, 'image/jpeg', 0.9)
  }, [])

  const retakePhoto = useCallback(() => {
    setCapturedImage(null)
    setCapturedFile(null)
    setAnalysisResult(null)
    setShowAuthDialog(false)
    setAuthDecision('pending')
    if (capturedImage) {
      URL.revokeObjectURL(capturedImage)
    }
  }, [capturedImage])

  const handleUploadDecision = (decision: 'approved' | 'rejected') => {
    setAuthDecision(decision)
    setShowAuthDialog(false)
    
    if (decision === 'approved' && capturedFile && analysisResult) {
      onPhotoTaken?.({ file: capturedFile, analysis: analysisResult })
      handleClose()
    }
  }

  const handleClose = () => {
    stopCamera()
    retakePhoto()
    setAuthDecision('pending')
    onClose()
  }

  const switchCamera = () => {
    setFacingMode(prev => prev === 'user' ? 'environment' : 'user')
  }

  const downloadPhoto = () => {
    if (capturedImage) {
      const link = document.createElement('a')
      link.href = capturedImage
      link.download = `photo_${Math.random().toString(36).substr(2, 9)}.jpg`
      link.click()
    }
  }

  // Start camera when modal opens
  useEffect(() => {
    if (isOpen && !capturedImage) {
      startCamera()
    }
    return () => {
      if (!isOpen) {
        stopCamera()
      }
    }
  }, [isOpen, capturedImage, startCamera, stopCamera])

  // Update camera when facing mode changes
  useEffect(() => {
    if (isRecording && !capturedImage) {
      stopCamera()
      setTimeout(() => startCamera(), 100)
    }
  }, [facingMode, isRecording, capturedImage, startCamera, stopCamera])

  const getRiskBadge = (result: MediaAnalysisResult) => {
    if (!result) return null
    
    const { riskLevel, score } = result
    const colorClass = MediaAnalysisService.getRiskColor(riskLevel || 'low')
    const bgClass = MediaAnalysisService.getRiskBgColor(riskLevel || 'low')
    
    return (
      <Badge variant="secondary" className={`text-white ${bgClass}`}>
        {(riskLevel || 'LOW').toUpperCase()}: {score || 0}%
      </Badge>
    )
  }

  return (
    <Dialog open={isOpen} onOpenChange={handleClose}>
      <DialogContent className="max-w-2xl h-[90vh] flex flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Camera className="h-5 w-5" />
            AI-Protected Camera
          </DialogTitle>
        </DialogHeader>

        <div className="flex-1 flex flex-col space-y-4">
          {!capturedImage ? (
            // Camera View
            <div className="relative flex-1 bg-black rounded-lg overflow-hidden">
              <video
                ref={videoRef}
                className="w-full h-full object-cover"
                autoPlay
                playsInline
                muted
              />
              
              {/* Camera Controls Overlay */}
              <div className="absolute bottom-4 left-0 right-0 flex justify-center items-center gap-4">
                <Button
                  variant="outline"
                  size="icon"
                  onClick={switchCamera}
                  className="bg-black/50 border-white/20 text-white hover:bg-black/70"
                >
                  <RotateCcw className="h-5 w-5" />
                </Button>
                
                <Button
                  size="lg"
                  onClick={capturePhoto}
                  disabled={!isRecording}
                  className="w-16 h-16 rounded-full bg-white text-black hover:bg-gray-200"
                >
                  <Camera className="h-8 w-8" />
                </Button>
                
                <Button
                  variant="outline"
                  size="icon"
                  onClick={handleClose}
                  className="bg-black/50 border-white/20 text-white hover:bg-black/70"
                >
                  <X className="h-5 w-5" />
                </Button>
              </div>
            </div>
          ) : (
            // Photo Review
            <div className="flex-1 space-y-4">
              <div className="relative">
                <img
                  src={capturedImage}
                  alt="Captured photo"
                  className="w-full h-64 object-cover rounded-lg"
                />
                
                {/* Analysis Badge */}
                {analysisResult && (
                  <div className="absolute top-2 right-2">
                    {getRiskBadge(analysisResult)}
                  </div>
                )}
              </div>

              {/* Analysis Results */}
              {isAnalyzing && (
                <Card>
                  <CardContent className="p-4">
                    <div className="flex items-center gap-2 text-sm">
                      <Shield className="h-4 w-4 animate-spin" />
                      Analyzing photo with AI...
                    </div>
                  </CardContent>
                </Card>
              )}

              {analysisResult && (
                <Card>
                  <CardContent className="p-4 space-y-3">
                    <div className="flex items-center justify-between">
                      <h3 className="font-semibold flex items-center gap-2">
                        <Shield className="h-4 w-4" />
                        AI Analysis Results
                      </h3>
                      {getRiskBadge(analysisResult)}
                    </div>

                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <span className="text-muted-foreground">Confidence:</span>
                        <span className="ml-2 font-medium">{analysisResult.confidence}%</span>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Deepfake:</span>
                        <span className={`ml-2 font-medium ${analysisResult.isDeepfake ? 'text-red-500' : 'text-green-500'}`}>
                          {analysisResult.isDeepfake ? 'Detected' : 'Not Detected'}
                        </span>
                      </div>
                    </div>

                    {analysisResult.details && (
                      <div className="grid grid-cols-2 gap-2 text-xs">
                        <div>Face: {analysisResult.details.faceConsistency || 0}%</div>
                        <div>Temporal: {analysisResult.details.temporalConsistency || 0}%</div>
                        <div>Artifacts: {analysisResult.details.artifactDetection || 0}%</div>
                        <div>Lighting: {analysisResult.details.lightingAnalysis || 0}%</div>
                      </div>
                    )}
                  </CardContent>
                </Card>
              )}

              {/* Authorization Dialog */}
              {showAuthDialog && analysisResult && (
                <Alert className="border-yellow-500 bg-yellow-50 dark:bg-yellow-950">
                  <AlertTriangle className="h-4 w-4 text-yellow-600" />
                  <AlertDescription className="space-y-3">
                    <div>
                      <strong>Content Review Required</strong>
                      <p className="text-sm mt-1">
                        Our AI has detected potential concerns with this image. 
                        Risk Level: <strong>{analysisResult.riskLevel?.toUpperCase()}</strong> ({analysisResult.score}% confidence)
                      </p>
                    </div>
                    
                    <div className="flex gap-2">
                      <Button
                        size="sm"
                        onClick={() => handleUploadDecision('approved')}
                        className="bg-green-600 hover:bg-green-700"
                      >
                        <CheckCircle className="h-4 w-4 mr-1" />
                        Upload Anyway
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => handleUploadDecision('rejected')}
                        className="border-red-500 text-red-600 hover:bg-red-50"
                      >
                        <X className="h-4 w-4 mr-1" />
                        Don't Upload
                      </Button>
                    </div>
                  </AlertDescription>
                </Alert>
              )}

              {/* Action Buttons */}
              <div className="flex gap-2">
                <Button onClick={retakePhoto} variant="outline" className="flex-1">
                  <Camera className="h-4 w-4 mr-2" />
                  Retake
                </Button>
                
                <Button onClick={downloadPhoto} variant="outline">
                  <Download className="h-4 w-4 mr-2" />
                  Save
                </Button>

                {authDecision === 'approved' && capturedFile && analysisResult && (
                  <Button 
                    onClick={() => onPhotoTaken?.({ file: capturedFile, analysis: analysisResult })}
                    className="flex-1"
                  >
                    <Upload className="h-4 w-4 mr-2" />
                    Use Photo
                  </Button>
                )}

                {!showAuthDialog && authDecision === 'pending' && !isAnalyzing && analysisResult && (
                  <Button 
                    onClick={() => handleUploadDecision('approved')}
                    className="flex-1"
                  >
                    <Upload className="h-4 w-4 mr-2" />
                    Use Photo
                  </Button>
                )}
              </div>
            </div>
          )}

          {/* Hidden canvas for photo capture */}
          <canvas ref={canvasRef} className="hidden" />
        </div>
      </DialogContent>
    </Dialog>
  )
}
