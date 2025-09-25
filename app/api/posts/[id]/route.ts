import { NextRequest, NextResponse } from "next/server"
import { verifyToken } from "@/lib/auth"
import { getDatabase } from "@/lib/database"

export async function DELETE(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const postId = params.id
    console.log('Attempting to delete post:', postId)
    
    const token = request.headers.get('authorization')?.replace('Bearer ', '')
    console.log('Token received:', token ? 'Yes' : 'No')
    
    if (!token) {
      return NextResponse.json(
        { error: "Authentication required" },
        { status: 401 }
      )
    }

    let userId: number
    
    if (token.startsWith('demo-token-')) {
      const demoUserId = token.replace('demo-token-', '')
      userId = parseInt(demoUserId) || 1
      console.log('Demo user ID:', userId)
    } else {
      const decoded = verifyToken(token)
      if (!decoded) {
        return NextResponse.json(
          { error: "Invalid token" },
          { status: 401 }
        )
      }
      userId = decoded.userId
      console.log('Authenticated user ID:', userId)
    }

    try {
      const db = await getDatabase()
      console.log('Database connected')
      
      // Check if the post exists first
      const postCheck = await new Promise((resolve, reject) => {
        db.get(
          'SELECT * FROM posts WHERE id = ?',
          [postId],
          (err: any, row: any) => {
            if (err) reject(err)
            else resolve(row)
          }
        )
      })

      console.log('Post found:', postCheck ? 'Yes' : 'No')
      
      if (!postCheck) {
        return NextResponse.json(
          { error: "Post not found" },
          { status: 404 }
        )
      }

      // Check if the post belongs to the user
      const userPost = await new Promise((resolve, reject) => {
        db.get(
          'SELECT * FROM posts WHERE id = ? AND user_id = ?',
          [postId, userId],
          (err: any, row: any) => {
            if (err) reject(err)
            else resolve(row)
          }
        )
      })

      console.log('User owns post:', userPost ? 'Yes' : 'No')

      if (!userPost) {
        return NextResponse.json(
          { error: "You don't have permission to delete this post" },
          { status: 403 }
        )
      }

      // Delete the post
      await new Promise<void>((resolve, reject) => {
        db.run(
          'DELETE FROM posts WHERE id = ? AND user_id = ?',
          [postId, userId],
          (err: any) => {
            if (err) {
              console.error('Delete error:', err)
              reject(err)
            } else {
              console.log('Post deleted successfully')
              resolve()
            }
          }
        )
      })

      return NextResponse.json({
        success: true,
        message: 'Post deleted successfully'
      })

    } catch (dbError) {
      console.error('Database error:', dbError)
      return NextResponse.json(
        { error: "Failed to delete post from database: " + (dbError instanceof Error ? dbError.message : 'Unknown error') },
        { status: 500 }
      )
    }

  } catch (error) {
    console.error('Error deleting post:', error)
    return NextResponse.json(
      { error: "Failed to delete post: " + (error instanceof Error ? error.message : 'Unknown error') },
      { status: 500 }
    )
  }
}
