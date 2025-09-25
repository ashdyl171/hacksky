import { NextRequest, NextResponse } from "next/server"
import { getDatabase } from "@/lib/database"

export async function POST() {
  try {
    const db = await getDatabase()
    
    // Clear all posts from database
    await db.run('DELETE FROM posts')
    await db.run('DELETE FROM likes')
    await db.run('DELETE FROM comments')
    
    // Reset auto-increment counters
    await db.run('DELETE FROM sqlite_sequence WHERE name IN ("posts", "likes", "comments")')
    
    const result = await db.get('SELECT COUNT(*) as count FROM posts')
    
    return NextResponse.json({
      success: true,
      message: "All posts cleared from database",
      postCount: result.count
    })
  } catch (error) {
    console.error('Database clear error:', error)
    return NextResponse.json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 })
  }
}
