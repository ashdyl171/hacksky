'use client'

import { useState, useCallback } from 'react'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Shield, AlertTriangle, CheckCircle, Camera, Clock } from 'lucide-react'
import { MediaUploadWithAnalysis } from '@/components/media-upload-analysis'
import { MediaAnalysisResult } from '@/lib/media-analysis'

interface CreateStoryModalProps {
  isOpen: boolean
  onClose: () => void
  onStoryCreated: (story: any) => void
}

export function CreateStoryModal({ isOpen, onClose, onStoryCreated }: CreateStoryModalProps) {
  const [analysisResults, setAnalysisResults] = useState<MediaAnalysisResult[]>([])
  const [selectedFiles, setSelectedFiles] = useState<File[]>([])
  const [isUploading, setIsUploading] = useState(false)
  const [showWarning, setShowWarning] = useState(false)

  const handleFilesAnalyzed = useCallback((results: MediaAnalysisResult[]) => {
    setAnalysisResults(results)
    
    // Check if any files have high risk - with safety check
    const hasHighRisk = results.some(r => r && r.riskLevel === 'high')
    setShowWarning(hasHighRisk)
  }, [])

  const handleFilesSelected = useCallback((files: File[]) => {
    setSelectedFiles(files)
  }, [])

  const handleCreateStory = async () => {
    if (selectedFiles.length === 0) {
      return
    }

    setIsUploading(true)

    try {
      // Simulate story creation
      const newStory = {
        id: `story_${Math.random().toString(36).substr(2, 9)}`,
        user: {
          id: "current_user",
          username: JSON.parse(localStorage.getItem("currentUser") || "{}").username || "demo_user",
          profilePic: "/placeholder.svg?height=40&width=40",
        },
        mediaUrl: URL.createObjectURL(selectedFiles[0]),
        timestamp: "now",
        viewed: false,
        deepfakeScore: analysisResults[0]?.score || 0,
        riskLevel: analysisResults[0]?.riskLevel || 'low'
      }

      onStoryCreated(newStory)
      handleClose()
    } catch (error) {
      console.error("Error creating story:", error)
    } finally {
      setIsUploading(false)
    }
  }

  const handleClose = () => {
    setAnalysisResults([])
    setSelectedFiles([])
    setShowWarning(false)
    onClose()
  }

  const canPost = selectedFiles.length > 0 && !isUploading
  const hasHighRiskContent = analysisResults.some(r => r && r.riskLevel === 'high')

  return (
    <Dialog open={isOpen} onOpenChange={handleClose}>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Camera className="h-5 w-5" />
            Create Story with AI Protection
          </DialogTitle>
        </DialogHeader>

        <div className="space-y-6">
          {/* Story Duration Info */}
          <Alert>
            <Clock className="h-4 w-4" />
            <AlertDescription>
              Stories are visible for 24 hours and are automatically analyzed for authenticity.
            </AlertDescription>
          </Alert>

          {/* Media Upload with Analysis */}
          <MediaUploadWithAnalysis
            onFilesAnalyzed={handleFilesAnalyzed}
            onFilesSelected={handleFilesSelected}
            maxFiles={1}
            autoAnalyze={true}
            showDetailedResults={true}
            allowedTypes={['image/*', 'video/*']}
          />

          {/* High Risk Warning */}
          {showWarning && hasHighRiskContent && (
            <Alert variant="destructive">
              <AlertTriangle className="h-4 w-4" />
              <AlertDescription>
                <strong>Warning:</strong> AI analysis detected potential deepfake or manipulated content. 
                Stories with high manipulation risk may be flagged to viewers.
              </AlertDescription>
            </Alert>
          )}

          {/* Analysis Summary for Story */}
          {analysisResults.length > 0 && (
            <div className="bg-muted/50 p-4 rounded-lg">
              <h4 className="font-medium mb-2 flex items-center gap-2">
                <Shield className="h-4 w-4" />
                Story Authenticity Check
              </h4>
              {analysisResults.map((result, index) => (
                <div key={index} className="space-y-2">
                  <div className="flex items-center gap-2 text-sm">
                    {result?.isDeepfake ? (
                      <AlertTriangle className="h-4 w-4 text-red-500" />
                    ) : (
                      <CheckCircle className="h-4 w-4 text-green-500" />
                    )}
                    <span>
                      {result?.isDeepfake ? 'Potential manipulation detected' : 'Content appears authentic'} 
                      ({result?.score || 0}% confidence)
                    </span>
                  </div>
                  
                  {result?.riskLevel === 'high' && (
                    <div className="text-xs text-red-600 bg-red-50 p-2 rounded">
                      High-risk content may display a warning badge to viewers
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex justify-end gap-3 pt-4 border-t">
            <Button variant="outline" onClick={handleClose}>
              Cancel
            </Button>
            <Button 
              onClick={handleCreateStory} 
              disabled={!canPost}
              className="min-w-[100px]"
            >
              {isUploading ? "Creating..." : "Share Story"}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
