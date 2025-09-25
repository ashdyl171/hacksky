import { NextRequest, NextResponse } from "next/server"
import { verifyToken } from "@/lib/auth"
import { createPost, getPosts, likePost } from "@/lib/content"
import { getDatabase } from "@/lib/database"

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const limit = Math.min(parseInt(searchParams.get('limit') || '20'), 50) // Cap at 50
    const offset = parseInt(searchParams.get('offset') || '0')
    
    // Get user ID from token if available
    const token = request.headers.get('authorization')?.replace('Bearer ', '')
    let userId: number | undefined
    
    if (token && token.startsWith('demo-token-')) {
      userId = parseInt(token.replace('demo-token-', '')) || 1
    } else if (token) {
      const decoded = verifyToken(token)
      userId = decoded?.userId
    }

    const posts = await getPosts(userId, limit, offset)
    
    // Optimize: Pre-create IST formatter
    const istFormatter = new Intl.DateTimeFormat('en-IN', {
      timeZone: 'Asia/Kolkata',
      year: 'numeric',
      month: 'short', 
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
    
    // Transform for frontend compatibility (optimized)
    const formattedPosts = posts.map(post => ({
      id: post.id.toString(),
      user: {
        id: post.user_id.toString(),
        username: post.username,
        profilePic: post.profile_pic || "/placeholder.svg"
      },
      mediaUrl: post.media_url,
      mediaType: post.media_type,
      caption: post.caption || "",
      deepfakeScore: post.deepfake_score,
      likes: post.likes_count,
      comments: post.comments_count,
      timestamp: istFormatter.format(new Date(post.created_at)),
      isLiked: post.is_liked,
      riskLevel: post.risk_level,
      aiConfirmed: post.ai_confirmed
    }))

    return NextResponse.json({
      success: true,
      posts: formattedPosts
    })

  } catch (error) {
    console.error('Error fetching posts:', error)
    return NextResponse.json(
      { error: "Failed to fetch posts" },
      { status: 500 }
    )
  }
}

export async function POST(request: NextRequest) {
  try {
    const contentType = request.headers.get('content-type')
    
    // Handle both form data and JSON requests
    let data: any
    if (contentType?.includes('multipart/form-data')) {
      const formData = await request.formData()
      data = {
        file: formData.get("file") as File,
        caption: formData.get("caption") as string,
        userId: formData.get("userId") as string,
        mediaUrl: formData.get("mediaUrl") as string,
        mediaType: formData.get("mediaType") as string,
        analysisResult: formData.get("analysisResult") ? JSON.parse(formData.get("analysisResult") as string) : null,
        aiConfirmed: formData.get("aiConfirmed") === 'true'
      }
    } else {
      data = await request.json()
    }

    const token = request.headers.get('authorization')?.replace('Bearer ', '')
    let userId: number
    let username = 'user' // Default fallback
    
    if (token && token.startsWith('demo-token-')) {
      userId = parseInt(token.replace('demo-token-', '')) || 1
      username = `user${userId}`
    } else if (token) {
      const decoded = verifyToken(token)
      if (!decoded) {
        return NextResponse.json({ error: "Invalid token" }, { status: 401 })
      }
      userId = decoded.userId
      username = `user${userId}` // Simple fallback for performance
    } else if (data.userId) {
      userId = parseInt(data.userId)
      username = data.username || `user${userId}`
    } else {
      return NextResponse.json({ error: "Authentication required" }, { status: 401 })
    }

    if (!data.mediaUrl && !data.file) {
      return NextResponse.json({ error: "Missing media" }, { status: 400 })
    }

    // Handle media URL efficiently
    let mediaUrl = data.mediaUrl || "/placeholder.svg?height=600&width=600"
    let mediaType = data.mediaType || 'image'

    // Create post in database
    const postId = await createPost(
      userId,
      mediaUrl,
      mediaType,
      data.caption,
      data.analysisResult,
      data.aiConfirmed || false
    )

    console.log(`Post created: ID ${postId} by User ${userId}`)

    // Pre-create IST formatter for consistent performance
    const istFormatter = new Intl.DateTimeFormat('en-IN', {
      timeZone: 'Asia/Kolkata',
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })

    // Calculate the correct deepfake score based on backend prediction
    const calculateDeepfakeScore = (analysisResult: any): number => {
      if (analysisResult?.backendInfo) {
        const confidence = analysisResult.backendInfo.confidence
        const prediction = analysisResult.backendInfo.prediction
        
        // If backend says "Real", then deepfake score is (1 - confidence)
        // If backend says "Deepfake", then deepfake score is confidence
        if (prediction === "Real") {
          return Math.round((1 - confidence) * 100)
        } else {
          return Math.round(confidence * 100)
        }
      }
      // Fallback to analysis confidence (already in percentage)
      return analysisResult?.confidence || 0
    }

    const deepfakeScore = calculateDeepfakeScore(data.analysisResult)

    const post = {
      id: (postId || Date.now()).toString(),
      userId: userId.toString(),
      mediaUrl,
      mediaType,
      caption: data.caption,
      deepfakeScore,
      likes: 0,
      comments: 0,
      createdAt: istFormatter.format(new Date()),
      riskLevel: data.analysisResult?.riskLevel || 'low',
      aiConfirmed: data.aiConfirmed || false,
      user: {
        id: userId.toString(),
        username,
        profilePic: "/placeholder.svg"
      }
    }

    return NextResponse.json({
      success: true,
      post,
      postId: postId || Date.now(),
      deepfakeScore,
      message: postId ? 'Post created successfully' : 'Post created in demo mode'
    })

  } catch (error) {
    console.error('Error creating post:', error)
    return NextResponse.json(
      { error: "Failed to create post: " + (error instanceof Error ? error.message : 'Unknown error') },
      { status: 500 }
    )
  }
}
