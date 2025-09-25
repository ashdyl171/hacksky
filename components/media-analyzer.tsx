'use client'

import { useState, useCallback, useRef } from 'react'
import { useTheme } from 'next-themes'
import Link from 'next/link'
import { 
  Upload, 
  FileVideo, 
  FileImage, 
  AlertTriangle, 
  CheckCircle, 
  XCircle, 
  Shield, 
  Eye, 
  Zap,
  Clock,
  Brain,
  Scan,
  Info,
  RotateCcw,
  Download,
  Share2,
  ArrowLeft,
  Home
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Separator } from '@/components/ui/separator'
import { 
  Dialog, 
  DialogContent, 
  DialogDescription, 
  DialogFooter, 
  DialogHeader, 
  DialogTitle 
} from '@/components/ui/dialog'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { ThemeSwitcher } from '@/components/theme-switcher'
import { cn } from '@/lib/utils'

interface AnalysisResult {
  confidence: number
  isDeepfake: boolean
  details: {
    faceConsistency: number
    temporalConsistency: number
    artifactDetection: number
    lightingAnalysis?: number
    compressionArtifacts?: number
    motionAnalysis?: number
    audioVisualSync?: number
  }
  processingTime: string
  modelVersion: string
  riskLevel: 'low' | 'medium' | 'high'
  recommendation: string
  filename?: string
  filesize?: string
  filetype?: string
}

interface FileAnalysis {
  file: File
  preview: string
  analysis?: AnalysisResult
  status: 'pending' | 'analyzing' | 'completed' | 'error'
  error?: string
}

interface DetailedAnalysisProps {
  fileAnalysis: FileAnalysis
  isOpen: boolean
  onClose: () => void
}

const DetailedAnalysisDialog = ({ fileAnalysis, isOpen, onClose }: DetailedAnalysisProps) => {
  const analysis = fileAnalysis.analysis
  
  if (!analysis) return null

  const detailsArray = [
    { label: 'Face Consistency', value: analysis.details.faceConsistency, description: 'Consistency of facial features and expressions' },
    { label: 'Temporal Consistency', value: analysis.details.temporalConsistency, description: 'Frame-to-frame consistency (video only)' },
    { label: 'Artifact Detection', value: analysis.details.artifactDetection, description: 'Digital manipulation artifacts' },
    { label: 'Lighting Analysis', value: analysis.details.lightingAnalysis, description: 'Natural lighting patterns' },
    { label: 'Compression Artifacts', value: analysis.details.compressionArtifacts, description: 'Unusual compression patterns' },
  ]

  if (analysis.filetype === 'video') {
    detailsArray.push(
      { label: 'Motion Analysis', value: analysis.details.motionAnalysis || 0, description: 'Natural motion patterns' },
      { label: 'Audio-Visual Sync', value: analysis.details.audioVisualSync || 0, description: 'Synchronization between audio and video' }
    )
  }

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="sm:max-w-2xl max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center space-x-2">
            <Brain className="h-5 w-5" />
            <span>Detailed Analysis</span>
          </DialogTitle>
          <DialogDescription>
            Comprehensive AI analysis results for {analysis.filename}
          </DialogDescription>
        </DialogHeader>
        
        <div className="space-y-6">
          {/* Overview */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card>
              <CardContent className="p-4 text-center">
                <div className="text-2xl font-bold text-primary">{analysis.confidence}%</div>
                <div className="text-sm text-muted-foreground">Confidence</div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-4 text-center">
                <Badge 
                  variant={analysis.riskLevel === 'high' ? 'destructive' : 
                           analysis.riskLevel === 'medium' ? 'default' : 'secondary'}
                  className="text-lg px-3 py-1"
                >
                  {analysis.riskLevel.toUpperCase()}
                </Badge>
                <div className="text-sm text-muted-foreground mt-1">Risk Level</div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-4 text-center">
                <div className="text-2xl font-bold">{analysis.processingTime}</div>
                <div className="text-sm text-muted-foreground">Processing Time</div>
              </CardContent>
            </Card>
          </div>

          {/* File Information */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">File Information</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-muted-foreground">Filename:</span>
                  <div className="font-medium break-all">{analysis.filename}</div>
                </div>
                <div>
                  <span className="text-muted-foreground">File Size:</span>
                  <div className="font-medium">{analysis.filesize}</div>
                </div>
                <div>
                  <span className="text-muted-foreground">File Type:</span>
                  <div className="font-medium capitalize">{analysis.filetype}</div>
                </div>
                <div>
                  <span className="text-muted-foreground">Model Version:</span>
                  <div className="font-medium">{analysis.modelVersion}</div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Analysis Details */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Analysis Metrics</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {detailsArray.map((detail, index) => (
                  <div key={index} className="space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-medium">{detail.label}</span>
                      <span className="text-sm font-bold">{detail.value}%</span>
                    </div>
                    <Progress value={detail.value} className="h-2" />
                    <p className="text-xs text-muted-foreground">{detail.description}</p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Recommendation */}
          <Alert>
            <Info className="h-4 w-4" />
            <AlertDescription className="text-sm">
              <strong>Recommendation:</strong> {analysis.recommendation}
            </AlertDescription>
          </Alert>
        </div>
        
        <DialogFooter className="flex justify-between">
          <div className="flex space-x-2">
            <Button variant="outline" size="sm">
              <Download className="h-4 w-4 mr-1" />
              Export Report
            </Button>
            <Button variant="outline" size="sm">
              <Share2 className="h-4 w-4 mr-1" />
              Share
            </Button>
          </div>
          <Button onClick={onClose}>Close</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

const SUPPORTED_TYPES = {
  image: [],
  video: ['video/mp4', 'video/webm', 'video/quicktime', 'video/avi', 'video/ogg']
}

export default function MediaAnalyzer() {
  const [files, setFiles] = useState<FileAnalysis[]>([])
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [showAuthDialog, setShowAuthDialog] = useState(false)
  const [showDetailedAnalysis, setShowDetailedAnalysis] = useState(false)
  const [currentFileForAuth, setCurrentFileForAuth] = useState<FileAnalysis | null>(null)
  const [currentFileForDetails, setCurrentFileForDetails] = useState<FileAnalysis | null>(null)
  const [dragActive, setDragActive] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const isSupported = (file: File) => {
    return SUPPORTED_TYPES.video.includes(file.type)
  }

  const getFileType = (file: File): 'image' | 'video' | 'unknown' => {
    if (SUPPORTED_TYPES.video.includes(file.type)) return 'video'
    return 'unknown'
  }

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  const createPreview = (file: File): Promise<string> => {
    return new Promise((resolve) => {
      const reader = new FileReader()
      reader.onload = () => resolve(reader.result as string)
      reader.readAsDataURL(file)
    })
  }

  const analyzeFileContent = async (file: File): Promise<AnalysisResult> => {
    const formData = new FormData()
    formData.append('file', file)
    
    try {
      const response = await fetch('/api/deepfake/analyze', {
        method: 'POST',
        body: formData
      })
      
      if (!response.ok) {
        throw new Error('Analysis failed')
      }
      
      const data = await response.json()
      
      if (!data.success) {
        throw new Error(data.error || 'Analysis failed')
      }
      
      // USE THE ACTUAL BACKEND RESPONSE - NO MORE RANDOM DATA!
      return data.analysis
    } catch (error) {
      console.error('Backend analysis failed:', error)
      
      // Only support video files for deepfake detection
      const filetype = getFileType(file)
      if (filetype !== 'video') {
        throw new Error('Only video files are supported for deepfake detection')
      }
      
      // Return error state instead of fake data
      return {
        confidence: 0,
        isDeepfake: false,
        details: {
          faceConsistency: 0,
          temporalConsistency: 0,
          artifactDetection: 0,
          lightingAnalysis: 0,
          compressionArtifacts: 0,
          motionAnalysis: 0,
          audioVisualSync: 0
        },
        processingTime: '0s',
        modelVersion: 'Backend Unavailable',
        riskLevel: 'low' as const,
        recommendation: 'Backend analysis failed. Please ensure FastAPI server is running on localhost:8000',
        filename: file.name,
        filesize: formatFileSize(file.size),
        filetype
      }
    }
  }

  const handleFiles = async (fileList: FileList) => {
    const newFiles: FileAnalysis[] = []
    
    for (let i = 0; i < fileList.length; i++) {
      const file = fileList[i]
      
      if (!isSupported(file)) {
        continue
      }
      
      const preview = await createPreview(file)
      newFiles.push({
        file,
        preview,
        status: 'pending'
      })
    }
    
    setFiles(prev => [...prev, ...newFiles])
  }

  const analyzeFile = async (fileIndex: number) => {
    setFiles(prev => prev.map((f, i) => 
      i === fileIndex ? { ...f, status: 'analyzing' } : f
    ))

    try {
      const analysis = await analyzeFileContent(files[fileIndex].file)
      
      setFiles(prev => prev.map((f, i) => 
        i === fileIndex ? { 
          ...f, 
          status: 'completed', 
          analysis 
        } : f
      ))

      // Show authorization dialog for suspicious content
      if (analysis.isDeepfake && analysis.riskLevel === 'high') {
        setCurrentFileForAuth(files[fileIndex])
        setShowAuthDialog(true)
      }
    } catch (error) {
      setFiles(prev => prev.map((f, i) => 
        i === fileIndex ? { 
          ...f, 
          status: 'error', 
          error: 'Analysis failed' 
        } : f
      ))
    }
  }

  const analyzeAllFiles = async () => {
    setIsAnalyzing(true)
    
    for (let i = 0; i < files.length; i++) {
      if (files[i].status === 'pending') {
        await analyzeFile(i)
      }
    }
    
    setIsAnalyzing(false)
  }

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true)
    } else if (e.type === "dragleave") {
      setDragActive(false)
    }
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFiles(e.dataTransfer.files)
    }
  }, [])

  const getRiskColor = (riskLevel: string, isDeepfake: boolean) => {
    if (isDeepfake) {
      switch (riskLevel) {
        case 'high': return 'destructive'
        case 'medium': return 'default'
        case 'low': return 'secondary'
        default: return 'secondary'
      }
    }
    return 'default'
  }

  const getRiskIcon = (riskLevel: string, isDeepfake: boolean) => {
    if (isDeepfake) {
      switch (riskLevel) {
        case 'high': return <XCircle className="h-4 w-4" />
        case 'medium': return <AlertTriangle className="h-4 w-4" />
        case 'low': return <Eye className="h-4 w-4" />
        default: return <AlertTriangle className="h-4 w-4" />
      }
    }
    return <CheckCircle className="h-4 w-4" />
  }

  return (
    <div className="min-h-screen gradient-bg">
      {/* Header */}
      <div className="border-b glass sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <Link href="/">
                <Button variant="ghost" size="icon" title="Back to Home">
                  <Home className="h-5 w-5" />
                </Button>
              </Link>
              <div className="flex items-center space-x-2">
                <Shield className="h-8 w-8 text-primary" />
                <h1 className="text-2xl font-bold">Media Authenticator</h1>
              </div>
              <Badge variant="secondary" className="hidden sm:inline-flex">
                <Brain className="h-3 w-3 mr-1" />
                AI-Powered Detection
              </Badge>
            </div>
            
            <div className="flex items-center space-x-2">
              <ThemeSwitcher />
            </div>
          </div>
        </div>
      </div>

      <div className="container mx-auto px-4 py-8 space-y-8">
        {/* Upload Area */}
        <Card className="relative overflow-hidden">
          <CardHeader>
                         <CardTitle className="flex items-center space-x-2">
               <Upload className="h-5 w-5" />
               <span>Upload Videos for Deepfake Detection</span>
             </CardTitle>
          </CardHeader>
          <CardContent>
            <div
              className={cn(
                "relative border-2 border-dashed rounded-lg p-8 text-center transition-all duration-200",
                dragActive 
                  ? "border-primary bg-primary/5 scale-105" 
                  : "border-muted-foreground/25 hover:border-muted-foreground/50",
                "group cursor-pointer"
              )}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
                             <input
                 ref={fileInputRef}
                 type="file"
                 multiple
                 accept="video/*"
                 onChange={(e) => e.target.files && handleFiles(e.target.files)}
                 className="hidden"
               />
              
              <div className="space-y-4">
                <div className="mx-auto w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center group-hover:bg-primary/20 transition-colors">
                  <Upload className="h-8 w-8 text-primary" />
                </div>
                
                                 <div>
                   <p className="text-lg font-medium">Drop video files here or click to upload</p>
                   <p className="text-sm text-muted-foreground mt-1">
                     Supports videos (MP4, WebM, AVI, MOV, OGG) for deepfake detection
                   </p>
                 </div>
                 
                 <div className="flex items-center justify-center space-x-4 text-xs text-muted-foreground">
                   <div className="flex items-center space-x-1">
                     <FileVideo className="h-3 w-3" />
                     <span>Videos Only</span>
                   </div>
                 </div>
              </div>
            </div>
            
            {files.length > 0 && (
              <div className="mt-6 flex items-center justify-between">
                <p className="text-sm text-muted-foreground">
                  {files.length} file{files.length !== 1 ? 's' : ''} ready for analysis
                </p>
                <Button 
                  onClick={analyzeAllFiles}
                  disabled={isAnalyzing || files.every(f => f.status !== 'pending')}
                  className="flex items-center space-x-2"
                >
                  <Scan className="h-4 w-4" />
                  <span>{isAnalyzing ? 'Analyzing...' : 'Analyze All'}</span>
                </Button>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Files Grid */}
        {files.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {files.map((fileAnalysis, index) => (
              <Card key={index} className="overflow-hidden">
                <div className="aspect-video relative bg-muted">
                  {getFileType(fileAnalysis.file) === 'image' ? (
                    <img 
                      src={fileAnalysis.preview} 
                      alt={fileAnalysis.file.name}
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <video 
                      src={fileAnalysis.preview}
                      className="w-full h-full object-cover"
                      controls={false}
                      muted
                    />
                  )}
                  
                  {/* Status Overlay */}
                  {fileAnalysis.status === 'analyzing' && (
                    <div className="absolute inset-0 bg-background/80 backdrop-blur-sm flex items-center justify-center">
                      <div className="text-center space-y-2">
                        <div className="animate-spin h-8 w-8 border-2 border-primary border-t-transparent rounded-full mx-auto" />
                        <p className="text-sm font-medium">Analyzing...</p>
                      </div>
                    </div>
                  )}
                </div>
                
                <CardContent className="p-4 space-y-4">
                  <div>
                    <h3 className="font-medium text-sm truncate" title={fileAnalysis.file.name}>
                      {fileAnalysis.file.name}
                    </h3>
                    <div className="flex items-center space-x-2 text-xs text-muted-foreground mt-1">
                      {getFileType(fileAnalysis.file) === 'image' ? (
                        <FileImage className="h-3 w-3" />
                      ) : (
                        <FileVideo className="h-3 w-3" />
                      )}
                      <span>{formatFileSize(fileAnalysis.file.size)}</span>
                    </div>
                  </div>
                  
                  {fileAnalysis.status === 'completed' && fileAnalysis.analysis && (
                    <div className="space-y-3">
                      {/* Main Result */}
                      <div className="flex items-center justify-between">
                        <Badge 
                          variant={getRiskColor(fileAnalysis.analysis.riskLevel, fileAnalysis.analysis.isDeepfake) as any}
                          className="flex items-center space-x-1"
                        >
                          {getRiskIcon(fileAnalysis.analysis.riskLevel, fileAnalysis.analysis.isDeepfake)}
                          <span>
                            {fileAnalysis.analysis.isDeepfake ? 'Suspicious' : 'Authentic'}
                          </span>
                        </Badge>
                        <span className="text-sm font-medium">
                          {fileAnalysis.analysis.confidence}% confidence
                        </span>
                      </div>
                      
                      {/* Confidence Bar */}
                      <Progress 
                        value={fileAnalysis.analysis.confidence} 
                        className="h-2" 
                      />
                      
                      {/* Quick Stats */}
                      <div className="grid grid-cols-2 gap-2 text-xs">
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Face:</span>
                          <span>{fileAnalysis.analysis.details.faceConsistency}%</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Artifacts:</span>
                          <span>{fileAnalysis.analysis.details.artifactDetection}%</span>
                        </div>
                      </div>
                      
                      {/* Processing Time */}
                      <div className="flex items-center space-x-1 text-xs text-muted-foreground">
                        <Clock className="h-3 w-3" />
                        <span>Processed in {fileAnalysis.analysis.processingTime}</span>
                      </div>
                      
                      {/* View Details Button */}
                      <Button 
                        size="sm" 
                        variant="outline" 
                        className="w-full mt-2"
                        onClick={() => {
                          setCurrentFileForDetails(fileAnalysis)
                          setShowDetailedAnalysis(true)
                        }}
                      >
                        <Info className="h-3 w-3 mr-1" />
                        View Details
                      </Button>
                    </div>
                  )}
                  
                  {fileAnalysis.status === 'pending' && (
                    <Button 
                      size="sm" 
                      className="w-full"
                      onClick={() => analyzeFile(index)}
                    >
                      <Zap className="h-3 w-3 mr-1" />
                      Analyze
                    </Button>
                  )}
                  
                  {fileAnalysis.status === 'error' && (
                    <Alert>
                      <AlertTriangle className="h-4 w-4" />
                      <AlertDescription>
                        {fileAnalysis.error || 'Analysis failed'}
                      </AlertDescription>
                    </Alert>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
        )}

        {/* Authorization Dialog */}
        <Dialog open={showAuthDialog} onOpenChange={setShowAuthDialog}>
          <DialogContent className="sm:max-w-md">
            <DialogHeader>
              <DialogTitle className="flex items-center space-x-2">
                <AlertTriangle className="h-5 w-5 text-destructive" />
                <span>Suspicious Content Detected</span>
              </DialogTitle>
              <DialogDescription>
                Our AI analysis has detected potential signs of manipulation in this media file. 
                Please review the analysis details and confirm if you want to proceed with the upload.
              </DialogDescription>
            </DialogHeader>
            
            {currentFileForAuth?.analysis && (
              <div className="space-y-4">
                <div className="p-4 border rounded-lg bg-muted/50 space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Confidence Level:</span>
                    <span className="font-medium">{currentFileForAuth.analysis.confidence}%</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span>Risk Level:</span>
                    <Badge variant="destructive">{currentFileForAuth.analysis.riskLevel}</Badge>
                  </div>
                  <div className="text-xs text-muted-foreground mt-2">
                    {currentFileForAuth.analysis.recommendation}
                  </div>
                </div>
              </div>
            )}
            
            <DialogFooter className="flex-col sm:flex-row gap-2">
              <Button variant="outline" onClick={() => setShowAuthDialog(false)}>
                Cancel Upload
              </Button>
              <Button 
                variant="destructive" 
                onClick={() => {
                  setShowAuthDialog(false)
                  // Proceed with upload logic here
                }}
              >
                Proceed Anyway
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {/* Detailed Analysis Dialog */}
        {currentFileForDetails && (
          <DetailedAnalysisDialog 
            fileAnalysis={currentFileForDetails}
            isOpen={showDetailedAnalysis}
            onClose={() => {
              setShowDetailedAnalysis(false)
              setCurrentFileForDetails(null)
            }}
          />
        )}
      </div>
    </div>
  )
}
