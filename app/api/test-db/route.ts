import { NextRequest, NextResponse } from "next/server"
import { getDatabase } from "@/lib/database"

export async function GET() {
  try {
    const db = await getDatabase()
    
    // Test database connection
    const result = await db.get('SELECT COUNT(*) as count FROM posts')
    const userCount = await db.get('SELECT COUNT(*) as count FROM users')
    
    return NextResponse.json({
      success: true,
      message: "Database connected successfully",
      stats: {
        posts: result.count,
        users: userCount.count
      }
    })
  } catch (error) {
    console.error('Database test error:', error)
    return NextResponse.json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 })
  }
}

export async function POST(request: NextRequest) {
  try {
    const { username, email } = await request.json()
    const db = await getDatabase()
    
    // Create a test user
    const result = await db.run(
      'INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
      [username, email, 'test-hash']
    )
    
    return NextResponse.json({
      success: true,
      userId: result.lastID,
      message: "Test user created"
    })
  } catch (error) {
    console.error('Database test creation error:', error)
    return NextResponse.json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 })
  }
}
