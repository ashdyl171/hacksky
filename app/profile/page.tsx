"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ArrowLeft, Settings, Grid, Bookmark, Tag, LogOut, Shield } from "lucide-react"
import { ThemeSwitcher } from "@/components/theme-switcher"
import Link from "next/link"

export default function ProfilePage() {
  const [currentUser, setCurrentUser] = useState<any>(null)
  const [posts, setPosts] = useState<any[]>([])
  const [stats, setStats] = useState({
    posts: 0,
    followers: 0,
    following: 0,
  })
  const [mounted, setMounted] = useState(false)
  const router = useRouter()

  useEffect(() => {
    setMounted(true)
    
    const checkAuth = () => {
      const user = localStorage.getItem("currentUser")
      if (!user) {
        router.push("/auth")
        return
      }

      const userData = JSON.parse(user)
      setCurrentUser(userData)

      // Load user stats from database with initial 0 counts
      loadUserStats(userData.id)

      // Load user posts from database
      loadUserPosts(userData.id)
    }

    const timer = setTimeout(checkAuth, 100)
    return () => clearTimeout(timer)
  }, [router])

  const loadUserStats = async (userId: string) => {
    try {
      const token = localStorage.getItem('authToken')
      if (!token) {
        setStats({ posts: 0, followers: 0, following: 0 })
        return
      }

      // Get posts count from database
      const postsResponse = await fetch('/api/posts', {
        headers: { 'Authorization': `Bearer ${token}` }
      })
      
      let postsCount = 0
      if (postsResponse.ok) {
        const postsData = await postsResponse.json()
        postsCount = postsData.posts?.filter((post: any) => post.user.id === userId).length || 0
      }

      setStats({
        posts: postsCount,
        followers: 0,
        following: 0,
      })
    } catch (error) {
      console.error('Error loading user stats:', error)
      setStats({ posts: 0, followers: 0, following: 0 })
    }
  }

  const loadUserPosts = async (userId: string) => {
    try {
      const token = localStorage.getItem('authToken')
      if (!token) {
        setPosts([])
        return
      }

      const response = await fetch('/api/posts', {
        headers: { 'Authorization': `Bearer ${token}` }
      })
      
      if (response.ok) {
        const data = await response.json()
        const userPosts = data.posts?.filter((post: any) => post.user.id === userId) || []
        setPosts(userPosts)
      } else {
        setPosts([])
      }
    } catch (error) {
      console.error('Error loading user posts:', error)
      setPosts([])
    }
  }

  const handleLogout = () => {
    localStorage.removeItem("currentUser")
    localStorage.removeItem("authToken")
    router.push("/auth")
  }

  if (!mounted) {
    return (
      <div className="min-h-screen gradient-bg flex items-center justify-center">
        <div className="text-center space-y-4">
          <div className="w-16 h-16 bg-gradient-to-r from-purple-500 via-pink-500 to-blue-500 rounded-full flex items-center justify-center shadow-2xl mx-auto animate-pulse">
            <Shield className="h-8 w-8 text-white" />
          </div>
          <div className="text-lg font-medium text-foreground">Loading Profile...</div>
        </div>
      </div>
    )
  }

  if (!currentUser) {
    return null
  }

  return (
    <div className="min-h-screen gradient-bg">
      {/* Header */}
      <header className="glass border-b sticky top-0 z-40">
        <div className="max-w-4xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Button variant="ghost" size="icon" onClick={() => router.push("/")}>
              <ArrowLeft className="h-6 w-6" />
            </Button>
            <h1 className="text-xl font-semibold text-foreground">{currentUser.username}</h1>
          </div>
          <div className="flex items-center gap-2">
            <Link href="/analyze">
              <Button variant="ghost" size="icon" title="Analyze Media">
                <Shield className="h-6 w-6" />
              </Button>
            </Link>
            <ThemeSwitcher />
            <Button variant="ghost" size="icon">
              <Settings className="h-6 w-6" />
            </Button>
            <Button variant="ghost" size="icon" onClick={handleLogout}>
              <LogOut className="h-6 w-6" />
            </Button>
          </div>
        </div>
      </header>

      <div className="max-w-4xl mx-auto px-4 py-6">
        {/* Profile Info */}
        <div className="flex items-start gap-8 mb-8">
          <Avatar className="h-32 w-32 ring-4 ring-primary/20 shadow-xl">
            <AvatarImage src={currentUser.profilePic || "/placeholder.svg"} />
            <AvatarFallback className="text-2xl bg-gradient-to-r from-purple-500 to-pink-500 text-white">
              {currentUser.username[0].toUpperCase()}
            </AvatarFallback>
          </Avatar>

          <div className="flex-1">
            <div className="flex items-center gap-4 mb-4">
              <h2 className="text-2xl font-semibold text-foreground">{currentUser.username}</h2>
              <Button variant="outline" className="border-border hover:bg-muted/50">Edit Profile</Button>
            </div>

            <div className="flex gap-8 mb-4">
              <div className="text-center">
                <div className="font-semibold text-lg text-foreground">{stats.posts}</div>
                <div className="text-muted-foreground">posts</div>
              </div>
              <div className="text-center cursor-pointer hover:opacity-80 transition-opacity">
                <div className="font-semibold text-lg text-foreground">{stats.followers.toLocaleString()}</div>
                <div className="text-muted-foreground">followers</div>
              </div>
              <div className="text-center cursor-pointer hover:opacity-80 transition-opacity">
                <div className="font-semibold text-lg text-foreground">{stats.following}</div>
                <div className="text-muted-foreground">following</div>
              </div>
            </div>

            <div>
              <p className="font-semibold text-foreground">Demo User</p>
              <p className="text-muted-foreground">📸 Photography enthusiast</p>
              <p className="text-muted-foreground">🌍 Sharing moments from around the world</p>
              <p className="text-muted-foreground">🔒 Protected by AI deepfake detection</p>
            </div>
          </div>
        </div>

        {/* Content Tabs */}
        <Tabs defaultValue="posts" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="posts" className="flex items-center gap-2">
              <Grid className="h-4 w-4" />
              Posts
            </TabsTrigger>
            <TabsTrigger value="saved" className="flex items-center gap-2">
              <Bookmark className="h-4 w-4" />
              Saved
            </TabsTrigger>
            <TabsTrigger value="tagged" className="flex items-center gap-2">
              <Tag className="h-4 w-4" />
              Tagged
            </TabsTrigger>
          </TabsList>

          <TabsContent value="posts" className="mt-6">
            <div className="grid grid-cols-3 gap-2 md:gap-4">
              {posts.map((post) => (
                <Card
                  key={post.id}
                  className="aspect-square overflow-hidden cursor-pointer hover:scale-105 transition-all duration-300 border-border/50 shadow-lg hover:shadow-xl"
                >
                  <CardContent className="p-0 h-full relative group">
                    {/* Display video or image based on mediaType */}
                    {post.mediaType === 'video' || post.mediaUrl?.includes('data:video') || post.mediaUrl?.includes('.mp4') || post.mediaUrl?.includes('.webm') || post.mediaUrl?.includes('.mov') ? (
                      <video
                        src={post.mediaUrl}
                        className="w-full h-full object-cover"
                        muted
                        playsInline
                      />
                    ) : (
                      <img
                        src={post.mediaUrl || "/placeholder.svg"}
                        alt={`Post ${post.id}`}
                        className="w-full h-full object-cover"
                      />
                    )}
                    {/* Hover overlay */}
                    <div className="absolute inset-0 bg-black/0 group-hover:bg-black/30 transition-all duration-300 flex items-center justify-center">
                      <div className="opacity-0 group-hover:opacity-100 transition-opacity text-white text-center">
                        <p className="text-sm font-medium">{post.likes || 0} likes</p>
                        <p className="text-xs">{post.comments || 0} comments</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>

          <TabsContent value="saved" className="mt-6">
            <div className="text-center py-12">
              <Bookmark className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
              <p className="text-muted-foreground">No saved posts yet</p>
            </div>
          </TabsContent>

          <TabsContent value="tagged" className="mt-6">
            <div className="text-center py-12">
              <Tag className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
              <p className="text-muted-foreground">No tagged posts yet</p>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
