import { NextRequest, NextResponse } from "next/server"
import { getDatabase } from "@/lib/database"

export async function GET(request: NextRequest) {
  try {
    const db = await getDatabase()
    
    // Get all tables data
    const users = await new Promise((resolve, reject) => {
      db.all("SELECT id, username, email, profile_pic, bio, created_at FROM users", (err: any, rows: any) => {
        if (err) reject(err)
        else resolve(rows)
      })
    })

    const posts = await new Promise((resolve, reject) => {
      db.all(`
        SELECT p.*, u.username as author_username 
        FROM posts p 
        LEFT JOIN users u ON p.user_id = u.id 
        ORDER BY p.created_at DESC
      `, (err: any, rows: any) => {
        if (err) reject(err)
        else resolve(rows)
      })
    })

    const stories = await new Promise((resolve, reject) => {
      db.all(`
        SELECT s.*, u.username as author_username 
        FROM stories s 
        LEFT JOIN users u ON s.user_id = u.id 
        ORDER BY s.created_at DESC
      `, (err: any, rows: any) => {
        if (err) reject(err)
        else resolve(rows)
      })
    })

    const comments = await new Promise((resolve, reject) => {
      db.all(`
        SELECT c.*, u.username as author_username, p.title as post_title
        FROM comments c 
        LEFT JOIN users u ON c.user_id = u.id 
        LEFT JOIN posts p ON c.post_id = p.id 
        ORDER BY c.created_at DESC
      `, (err: any, rows: any) => {
        if (err) reject(err)
        else resolve(rows)
      })
    })

    const likes = await new Promise((resolve, reject) => {
      db.all(`
        SELECT l.*, u.username as user_username, p.title as post_title
        FROM likes l 
        LEFT JOIN users u ON l.user_id = u.id 
        LEFT JOIN posts p ON l.post_id = p.id 
        ORDER BY l.created_at DESC
      `, (err: any, rows: any) => {
        if (err) reject(err)
        else resolve(rows)
      })
    })

    const sessions = await new Promise((resolve, reject) => {
      db.all(`
        SELECT s.*, u.username 
        FROM sessions s 
        LEFT JOIN users u ON s.user_id = u.id 
        ORDER BY s.created_at DESC
      `, (err: any, rows: any) => {
        if (err) reject(err)
        else resolve(rows)
      })
    })

    const aiAnalysis = await new Promise((resolve, reject) => {
      db.all(`
        SELECT a.*, u.username as user_username 
        FROM ai_analysis_logs a 
        LEFT JOIN users u ON a.user_id = u.id 
        ORDER BY a.created_at DESC
      `, (err: any, rows: any) => {
        if (err) reject(err)
        else resolve(rows)
      })
    })

    return NextResponse.json({
      success: true,
      database: {
        users,
        posts,
        stories,
        comments,
        likes,
        sessions,
        aiAnalysis
      },
      stats: {
        totalUsers: Array.isArray(users) ? users.length : 0,
        totalPosts: Array.isArray(posts) ? posts.length : 0,
        totalStories: Array.isArray(stories) ? stories.length : 0,
        totalComments: Array.isArray(comments) ? comments.length : 0,
        totalLikes: Array.isArray(likes) ? likes.length : 0,
        totalSessions: Array.isArray(sessions) ? sessions.length : 0,
        totalAiAnalysis: Array.isArray(aiAnalysis) ? aiAnalysis.length : 0
      }
    })

  } catch (error) {
    console.error('Database view error:', error)
    return NextResponse.json(
      { error: "Failed to fetch database data: " + (error instanceof Error ? error.message : 'Unknown error') },
      { status: 500 }
    )
  }
}
