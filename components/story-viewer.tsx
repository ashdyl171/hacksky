"use client"

import { useState, useEffect } from "react"
import { Dialog, DialogContent } from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { X, Heart, Send } from "lucide-react"

interface Story {
  id: string
  user: {
    id: string
    username: string
    profilePic: string
  }
  mediaUrl: string
  timestamp: string
  viewed: boolean
}

interface StoryViewerProps {
  story: Story
  onClose: () => void
}

export function StoryViewer({ story, onClose }: StoryViewerProps) {
  const [progress, setProgress] = useState(0)

  useEffect(() => {
    const timer = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 100) {
          onClose()
          return 0
        }
        return prev + 1
      })
    }, 50) // 5 second story duration

    return () => clearInterval(timer)
  }, [onClose])

  return (
    <Dialog open={true} onOpenChange={onClose}>
      <DialogContent className="max-w-sm p-0 bg-black">
        <div className="relative h-[600px] w-full">
          {/* Progress Bar */}
          <div className="absolute top-2 left-2 right-2 z-10">
            <div className="w-full bg-gray-600 rounded-full h-1">
              <div
                className="bg-white h-1 rounded-full transition-all duration-100"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>

          {/* Header */}
          <div className="absolute top-6 left-4 right-4 z-10 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Avatar className="h-8 w-8">
                <AvatarImage src={story.user.profilePic || "/placeholder.svg"} />
                <AvatarFallback>{story.user.username[0].toUpperCase()}</AvatarFallback>
              </Avatar>
              <span className="text-white font-semibold">{story.user.username}</span>
              <span className="text-gray-300 text-sm">{story.timestamp}</span>
            </div>
            <Button variant="ghost" size="icon" onClick={onClose}>
              <X className="h-5 w-5 text-white" />
            </Button>
          </div>

          {/* Story Content */}
          <img src={story.mediaUrl || "/placeholder.svg"} alt="Story content" className="w-full h-full object-cover" />

          {/* Bottom Actions */}
          <div className="absolute bottom-4 left-4 right-4 z-10 flex items-center gap-4">
            <Button variant="ghost" size="icon">
              <Heart className="h-6 w-6 text-white" />
            </Button>
            <Button variant="ghost" size="icon">
              <Send className="h-6 w-6 text-white" />
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
