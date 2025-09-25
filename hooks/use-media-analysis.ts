'use client'

import { useState, useCallback } from 'react'
import { MediaAnalysisService, MediaAnalysisResult, MediaFile } from '@/lib/media-analysis'

interface UseMediaAnalysisOptions {
  autoAnalyze?: boolean
  maxFiles?: number
  onAnalysisComplete?: (results: MediaAnalysisResult[]) => void
  onError?: (error: Error) => void
}

export function useMediaAnalysis(options: UseMediaAnalysisOptions = {}) {
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [results, setResults] = useState<MediaAnalysisResult[]>([])
  const [files, setFiles] = useState<MediaFile[]>([])
  const [error, setError] = useState<string | null>(null)
  const [progress, setProgress] = useState(0)

  const analyzeFiles = useCallback(async (filesToAnalyze: File[]) => {
    if (filesToAnalyze.length === 0) return

    setIsAnalyzing(true)
    setError(null)
    setProgress(0)

    try {
      const analysisResults: MediaAnalysisResult[] = []
      
      for (let i = 0; i < filesToAnalyze.length; i++) {
        const file = filesToAnalyze[i]
        
        // Update progress
        setProgress((i / filesToAnalyze.length) * 100)
        
        // Analyze the file
        const result = await MediaAnalysisService.analyzeMedia(file)
        analysisResults.push(result)
      }
      
      setProgress(100)
      setResults(analysisResults)
      options.onAnalysisComplete?.(analysisResults)
      
    } catch (err) {
      const error = err instanceof Error ? err : new Error('Analysis failed')
      setError(error.message)
      options.onError?.(error)
    } finally {
      setIsAnalyzing(false)
    }
  }, [options])

  const addFiles = useCallback(async (newFiles: File[]) => {
    const validFiles = newFiles.filter(MediaAnalysisService.isValidMediaFile)
    
    if (validFiles.length === 0) {
      setError('No valid media files selected')
      return
    }

    if (options.maxFiles && files.length + validFiles.length > options.maxFiles) {
      setError(`Maximum ${options.maxFiles} files allowed`)
      return
    }

    const mediaFiles: MediaFile[] = []
    
    for (const file of validFiles) {
      const preview = await MediaAnalysisService.createFilePreview(file)
      const fileType = MediaAnalysisService.getFileType(file)
      
      if (fileType === 'unknown') continue // Skip unknown file types
      
      const mediaFile: MediaFile = {
        file,
        preview,
        type: fileType,
        id: Math.random().toString(36).substr(2, 9)
      }
      mediaFiles.push(mediaFile)
    }

    setFiles(prev => [...prev, ...mediaFiles])

    if (options.autoAnalyze) {
      await analyzeFiles(validFiles)
    }
  }, [files.length, options.maxFiles, options.autoAnalyze, analyzeFiles])

  const removeFile = useCallback((fileId: string) => {
    setFiles(prev => prev.filter(f => f.id !== fileId))
    setResults(prev => prev.filter((_, index) => files[index]?.id !== fileId))
  }, [files])

  const clearAll = useCallback(() => {
    setFiles([])
    setResults([])
    setError(null)
    setProgress(0)
  }, [])

  const getResultForFile = useCallback((fileId: string): MediaAnalysisResult | undefined => {
    const fileIndex = files.findIndex(f => f.id === fileId)
    return fileIndex >= 0 ? results[fileIndex] : undefined
  }, [files, results])

  return {
    // State
    isAnalyzing,
    results,
    files,
    error,
    progress,
    
    // Actions
    analyzeFiles,
    addFiles,
    removeFile,
    clearAll,
    getResultForFile,
    
    // Computed
    hasFiles: files.length > 0,
    hasResults: results.length > 0,
    canAnalyze: files.length > 0 && !isAnalyzing,
  }
}
