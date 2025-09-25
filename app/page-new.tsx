"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Heart, MessageCircle, Share, Bookmark, MoreHorizontal, Camera, Send, Shield, AlertTriangle } from "lucide-react"
import { CreatePostModal } from "@/components/create-post-modal"
import { CreateStoryModal } from "@/components/create-story-modal"
import { StoryViewer } from "@/components/story-viewer"
import { MessagingPanel } from "@/components/messaging-panel"
import { ThemeSwitcher } from "@/components/theme-switcher"

interface Post {
  id: string
  user: {
    id: string
    username: string
    profilePic: string
  }
  mediaUrl: string
  caption: string
  deepfakeScore: number
  likes: number
  comments: number
  timestamp: string
  isLiked: boolean
}

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
  deepfakeScore?: number
  riskLevel?: 'low' | 'medium' | 'high'
}

export default function HomePage() {
  const [posts, setPosts] = useState<Post[]>([])
  const [stories, setStories] = useState<Story[]>([])
  const [isCreatePostOpen, setIsCreatePostOpen] = useState(false)
  const [isCreateStoryOpen, setIsCreateStoryOpen] = useState(false)
  const [selectedStory, setSelectedStory] = useState<Story | null>(null)
  const [isMessagingOpen, setIsMessagingOpen] = useState(false)
  const [currentUser, setCurrentUser] = useState<any>(null)
  const [isLoading, setIsLoading] = useState(true)
  const router = useRouter()

  useEffect(() => {
    // Check authentication after component mounts to prevent hydration issues
    const checkAuth = () => {
      const user = localStorage.getItem("currentUser")
      if (!user) {
        router.push("/auth")
        return
      }
      setCurrentUser(JSON.parse(user))
      setIsLoading(false)
      
      // Load mock data after auth check
      loadMockData()
    }

    // Use a small delay to ensure proper hydration
    const timer = setTimeout(checkAuth, 100)
    return () => clearTimeout(timer)
  }, [router])

  const loadMockData = () => {
    const mockStories: Story[] = [
      {
        id: "1",
        user: { id: "1", username: "alice_doe", profilePic: "/placeholder.svg?height=40&width=40" },
        mediaUrl: "/placeholder.svg?height=600&width=400",
        timestamp: "2h",
        viewed: false,
        riskLevel: "low"
      },
      {
        id: "2",
        user: { id: "2", username: "john_smith", profilePic: "/placeholder.svg?height=40&width=40" },
        mediaUrl: "/placeholder.svg?height=600&width=400",
        timestamp: "4h",
        viewed: true,
        riskLevel: "medium"
      },
    ]

    const mockPosts: Post[] = [
      {
        id: "1",
        user: { id: "1", username: "alice_doe", profilePic: "/placeholder.svg?height=40&width=40" },
        mediaUrl: "/placeholder.svg?height=600&width=600",
        caption: "Beautiful sunset today! 🌅 #nature #photography",
        deepfakeScore: 15,
        likes: 234,
        comments: 12,
        timestamp: "2 hours ago",
        isLiked: false,
      },
      {
        id: "2",
        user: { id: "2", username: "john_smith", profilePic: "/placeholder.svg?height=40&width=40" },
        mediaUrl: "/placeholder.svg?height=600&width=600",
        caption: "Homemade pasta night! 🍝 Recipe in my bio",
        deepfakeScore: 8,
        likes: 156,
        comments: 23,
        timestamp: "5 hours ago",
        isLiked: true,
      },
    ]

    setStories(mockStories)
    setPosts(mockPosts)
  }

  const handleLike = (postId: string) => {
    setPosts(
      posts.map((post) =>
        post.id === postId
          ? { ...post, isLiked: !post.isLiked, likes: post.isLiked ? post.likes - 1 : post.likes + 1 }
          : post,
      ),
    )
  }

  const handlePostCreated = (newPost: Post) => {
    setPosts([newPost, ...posts])
  }

  const handleStoryCreated = (newStory: Story) => {
    setStories([newStory, ...stories])
  }

  if (isLoading) {
    return (
      <div className="min-h-screen gradient-bg flex items-center justify-center">
        <div className="text-center space-y-4">
          <div className="w-16 h-16 bg-gradient-to-r from-purple-500 via-pink-500 to-blue-500 rounded-full flex items-center justify-center shadow-2xl mx-auto animate-pulse">
            <Shield className="h-8 w-8 text-white" />
          </div>
          <div className="text-lg font-medium text-foreground">Loading SocialGuard...</div>
          <div className="text-sm text-muted-foreground">Initializing your secure social experience</div>
        </div>
      </div>
    )
  }

  if (!currentUser) {
    return null // Prevent flash of content before redirect
  }

  return (
    <div className="min-h-screen gradient-bg">
      {/* Header */}
      <header className="glass border-b sticky top-0 z-40">
        <div className="max-w-4xl mx-auto px-4 py-3 flex items-center justify-between">
          <h1 className="text-2xl font-bold bg-gradient-to-r from-purple-600 via-pink-600 to-blue-600 bg-clip-text text-transparent">
            SocialGuard
          </h1>
          <div className="flex items-center gap-4">
            <Link href="/analyze">
              <Button variant="ghost" size="icon" title="Analyze Media">
                <Shield className="h-6 w-6" />
              </Button>
            </Link>
            <Button variant="ghost" size="icon" onClick={() => setIsCreatePostOpen(true)}>
              <Camera className="h-6 w-6" />
            </Button>
            <Button variant="ghost" size="icon" onClick={() => setIsMessagingOpen(true)}>
              <Send className="h-6 w-6" />
            </Button>
            <ThemeSwitcher />
            <Avatar className="h-8 w-8 cursor-pointer" onClick={() => router.push("/profile")}>
              <AvatarImage src={currentUser.profilePic || "/placeholder.svg"} />
              <AvatarFallback>{currentUser.username[0].toUpperCase()}</AvatarFallback>
            </Avatar>
          </div>
        </div>
      </header>

      <div className="max-w-4xl mx-auto px-4 py-6">
        {/* Stories Section */}
        <div className="mb-8">
          <div className="flex gap-4 overflow-x-auto pb-4">
            {/* Add Story */}
            <div className="flex-shrink-0 text-center">
              <div 
                className="w-16 h-16 rounded-full bg-gradient-to-r from-purple-500 to-pink-500 flex items-center justify-center cursor-pointer transition-transform hover:scale-105"
                onClick={() => setIsCreateStoryOpen(true)}
              >
                <Camera className="h-6 w-6 text-white" />
              </div>
              <p className="text-xs mt-1 text-muted-foreground">Your Story</p>
            </div>

            {/* Stories */}
            {stories.map((story) => (
              <div key={story.id} className="flex-shrink-0 text-center relative">
                <div
                  className={`w-16 h-16 rounded-full p-0.5 cursor-pointer transition-transform hover:scale-105 ${
                    story.viewed ? "bg-muted-foreground/30" : "bg-gradient-to-r from-purple-500 to-pink-500"
                  }`}
                  onClick={() => setSelectedStory(story)}
                >
                  <Avatar className="w-full h-full">
                    <AvatarImage src={story.user.profilePic || "/placeholder.svg"} />
                    <AvatarFallback>{story.user.username[0].toUpperCase()}</AvatarFallback>
                  </Avatar>
                </div>
                
                {/* Risk indicator for stories */}
                {story.riskLevel === 'high' && (
                  <div className="absolute -top-1 -right-1 w-4 h-4 bg-red-500 rounded-full flex items-center justify-center">
                    <AlertTriangle className="h-2 w-2 text-white" />
                  </div>
                )}
                {story.riskLevel === 'medium' && (
                  <div className="absolute -top-1 -right-1 w-4 h-4 bg-yellow-500 rounded-full flex items-center justify-center">
                    <AlertTriangle className="h-2 w-2 text-white" />
                  </div>
                )}
                
                <p className="text-xs mt-1 truncate w-16 text-muted-foreground">{story.user.username}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Posts Feed */}
        <div className="space-y-6">
          {posts.map((post) => (
            <Card key={post.id} className="overflow-hidden border border-border/50 shadow-lg hover:shadow-xl transition-all duration-300">
              {/* Post Header */}
              <div className="flex items-center justify-between p-4 border-b border-border/30">
                <div className="flex items-center gap-3">
                  <Avatar className="ring-2 ring-primary/20">
                    <AvatarImage src={post.user.profilePic || "/placeholder.svg"} />
                    <AvatarFallback>{post.user.username[0].toUpperCase()}</AvatarFallback>
                  </Avatar>
                  <div>
                    <p className="font-semibold text-foreground">{post.user.username}</p>
                    <p className="text-sm text-muted-foreground">{post.timestamp}</p>
                  </div>
                </div>
                <Button variant="ghost" size="icon" className="hover:bg-muted/50">
                  <MoreHorizontal className="h-5 w-5" />
                </Button>
              </div>

              {/* Post Media */}
              <div className="relative">
                <img
                  src={post.mediaUrl || "/placeholder.svg"}
                  alt="Post content"
                  className="w-full aspect-square object-cover"
                />
                {/* Deepfake Score Indicator */}
                {post.deepfakeScore > 50 && (
                  <div className="absolute top-2 right-2 bg-destructive text-destructive-foreground px-3 py-1.5 rounded-full text-xs font-medium shadow-lg backdrop-blur-sm">
                    <div className="flex items-center space-x-1">
                      <Shield className="h-3 w-3" />
                      <span>AI Detected: {post.deepfakeScore}%</span>
                    </div>
                  </div>
                )}
                {post.deepfakeScore <= 50 && post.deepfakeScore > 20 && (
                  <div className="absolute top-2 right-2 bg-yellow-500 text-white px-3 py-1.5 rounded-full text-xs font-medium shadow-lg backdrop-blur-sm">
                    <div className="flex items-center space-x-1">
                      <AlertTriangle className="h-3 w-3" />
                      <span>Caution: {post.deepfakeScore}%</span>
                    </div>
                  </div>
                )}
              </div>

              {/* Post Actions */}
              <CardContent className="p-4 bg-card">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-4">
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => handleLike(post.id)}
                      className={`transition-colors hover:bg-muted/50 ${post.isLiked ? "text-red-500 hover:text-red-600" : "hover:text-red-500"}`}
                    >
                      <Heart className={`h-6 w-6 transition-all ${post.isLiked ? "fill-current scale-110" : ""}`} />
                    </Button>
                    <Button variant="ghost" size="icon" className="hover:bg-muted/50 hover:text-blue-500">
                      <MessageCircle className="h-6 w-6" />
                    </Button>
                    <Button variant="ghost" size="icon" className="hover:bg-muted/50 hover:text-green-500">
                      <Share className="h-6 w-6" />
                    </Button>
                  </div>
                  <Button variant="ghost" size="icon" className="hover:bg-muted/50 hover:text-yellow-500">
                    <Bookmark className="h-6 w-6" />
                  </Button>
                </div>

                <p className="font-semibold mb-1 text-foreground">{post.likes.toLocaleString()} likes</p>
                <p className="mb-2 text-foreground">
                  <span className="font-semibold">{post.user.username}</span> 
                  <span className="ml-2">{post.caption}</span>
                </p>
                {post.comments > 0 && (
                  <p className="text-muted-foreground text-sm cursor-pointer hover:text-foreground transition-colors">
                    View all {post.comments} comments
                  </p>
                )}
              </CardContent>
            </Card>
          ))}
        </div>
      </div>

      {/* Modals */}
      <CreatePostModal
        isOpen={isCreatePostOpen}
        onClose={() => setIsCreatePostOpen(false)}
        onPostCreated={handlePostCreated}
      />

      <CreateStoryModal
        isOpen={isCreateStoryOpen}
        onClose={() => setIsCreateStoryOpen(false)}
        onStoryCreated={handleStoryCreated}
      />

      {selectedStory && <StoryViewer story={selectedStory} onClose={() => setSelectedStory(null)} />}

      {isMessagingOpen && <MessagingPanel onClose={() => setIsMessagingOpen(false)} />}
    </div>
  )
}
