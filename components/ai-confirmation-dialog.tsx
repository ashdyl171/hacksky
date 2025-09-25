"use client"

import React from 'react'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { 
  AlertTriangle, 
  Shield, 
  XCircle, 
  Eye, 
  Brain,
  CheckCircle,
  Info 
} from 'lucide-react'

interface AnalysisResult {
  confidence: number
  isDeepfake: boolean
  riskLevel: 'low' | 'medium' | 'high'
  recommendation: string
  filename?: string
  details: {
    faceConsistency: number
    temporalConsistency: number
    artifactDetection: number
    lightingAnalysis?: number
  }
}

interface AIConfirmationDialogProps {
  isOpen: boolean
  onClose: () => void
  onConfirm: () => void
  onCancel: () => void
  analysis: AnalysisResult
  contentType: 'post' | 'story' | 'content'
}

export function AIConfirmationDialog({
  isOpen,
  onClose,
  onConfirm,
  onCancel,
  analysis,
  contentType
}: AIConfirmationDialogProps) {
  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel) {
      case 'high': return 'destructive'
      case 'medium': return 'default'  
      case 'low': return 'secondary'
      default: return 'secondary'
    }
  }

  const getRiskIcon = (riskLevel: string, isDeepfake: boolean) => {
    if (isDeepfake) {
      switch (riskLevel) {
        case 'high': return <XCircle className="h-5 w-5 text-destructive" />
        case 'medium': return <AlertTriangle className="h-5 w-5 text-yellow-500" />
        case 'low': return <Eye className="h-5 w-5 text-blue-500" />
        default: return <AlertTriangle className="h-5 w-5" />
      }
    }
    return <CheckCircle className="h-5 w-5 text-green-500" />
  }

  const getAlertVariant = (riskLevel: string) => {
    return riskLevel === 'high' ? 'destructive' : 'default'
  }

  const getConfirmButtonVariant = (riskLevel: string) => {
    return riskLevel === 'high' ? 'destructive' : 'default'
  }

  return (
    <AlertDialog open={isOpen} onOpenChange={onClose}>
      <AlertDialogContent className="sm:max-w-lg">
        <AlertDialogHeader>
          <AlertDialogTitle className="flex items-center space-x-2">
            <Shield className="h-6 w-6 text-primary" />
            <span>AI Detection Alert</span>
          </AlertDialogTitle>
          <AlertDialogDescription className="text-base">
            Our AI system has detected potential signs of manipulation in your {contentType}. 
            Please review the analysis and confirm if you want to proceed.
          </AlertDialogDescription>
        </AlertDialogHeader>

        <div className="space-y-6">
          {/* Analysis Overview */}
          <div className="p-4 border rounded-lg bg-muted/30 space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                {getRiskIcon(analysis.riskLevel, analysis.isDeepfake)}
                <span className="font-medium">
                  {analysis.isDeepfake ? 'Potentially Manipulated' : 'Appears Authentic'}
                </span>
              </div>
              <Badge variant={getRiskColor(analysis.riskLevel) as any}>
                {analysis.riskLevel.toUpperCase()} RISK
              </Badge>
            </div>
            
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Confidence Level:</span>
                <span className="font-bold">{analysis.confidence}%</span>
              </div>
              <Progress value={analysis.confidence} className="h-2" />
            </div>

            {analysis.filename && (
              <div className="text-sm text-muted-foreground">
                <strong>File:</strong> {analysis.filename}
              </div>
            )}
          </div>

          {/* Detailed Metrics */}
          <div className="space-y-3">
            <h4 className="font-medium flex items-center space-x-2">
              <Brain className="h-4 w-4" />
              <span>Analysis Details</span>
            </h4>
            
            <div className="grid grid-cols-2 gap-3 text-sm">
              <div className="space-y-1">
                <div className="flex justify-between">
                  <span>Face Consistency:</span>
                  <span>{analysis.details.faceConsistency}%</span>
                </div>
                <Progress value={analysis.details.faceConsistency} className="h-1" />
              </div>
              
              <div className="space-y-1">
                <div className="flex justify-between">
                  <span>Temporal Analysis:</span>
                  <span>{analysis.details.temporalConsistency}%</span>
                </div>
                <Progress value={analysis.details.temporalConsistency} className="h-1" />
              </div>
              
              <div className="space-y-1">
                <div className="flex justify-between">
                  <span>Artifact Detection:</span>
                  <span>{analysis.details.artifactDetection}%</span>
                </div>
                <Progress value={analysis.details.artifactDetection} className="h-1" />
              </div>
              
              {analysis.details.lightingAnalysis && (
                <div className="space-y-1">
                  <div className="flex justify-between">
                    <span>Lighting Analysis:</span>
                    <span>{analysis.details.lightingAnalysis}%</span>
                  </div>
                  <Progress value={analysis.details.lightingAnalysis} className="h-1" />
                </div>
              )}
            </div>
          </div>

          {/* Recommendation */}
          <Alert variant={getAlertVariant(analysis.riskLevel) as any}>
            <Info className="h-4 w-4" />
            <AlertDescription>
              <strong>Recommendation:</strong> {analysis.recommendation}
            </AlertDescription>
          </Alert>

          {/* Warning for high-risk content */}
          {analysis.riskLevel === 'high' && (
            <Alert variant="destructive">
              <AlertTriangle className="h-4 w-4" />
              <AlertDescription>
                <strong>Warning:</strong> This content has been flagged as high-risk for manipulation. 
                Uploading such content may violate platform policies and could mislead other users.
              </AlertDescription>
            </Alert>
          )}
        </div>

        <AlertDialogFooter className="flex-col sm:flex-row gap-2">
          <AlertDialogCancel onClick={onCancel}>
            Cancel Upload
          </AlertDialogCancel>
          <AlertDialogAction 
            onClick={onConfirm}
            className={`flex items-center space-x-2 ${
              analysis.riskLevel === 'high' 
                ? 'bg-destructive text-destructive-foreground hover:bg-destructive/90' 
                : ''
            }`}
          >
            <Shield className="h-4 w-4" />
            <span>
              {analysis.riskLevel === 'high' 
                ? 'Upload Anyway (Not Recommended)' 
                : 'Confirm & Upload'
              }
            </span>
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}
