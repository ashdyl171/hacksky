import { NextRequest, NextResponse } from "next/server"
import { getDatabase } from "@/lib/database"

export async function GET() {
  try {
    console.log('Testing database connection...')
    const db = await getDatabase()
    
    // Get all posts to see what we have
    const posts = await new Promise((resolve, reject) => {
      db.all('SELECT * FROM posts', (err: any, rows: any) => {
        if (err) {
          console.error('Database query error:', err)
          reject(err)
        } else {
          console.log('Posts found:', rows?.length || 0)
          resolve(rows)
        }
      })
    })

    return NextResponse.json({
      success: true,
      message: 'Database connection test successful',
      postsCount: Array.isArray(posts) ? posts.length : 0,
      posts: posts
    })

  } catch (error) {
    console.error('Database test error:', error)
    return NextResponse.json(
      { error: "Database test failed: " + (error instanceof Error ? error.message : 'Unknown error') },
      { status: 500 }
    )
  }
}
