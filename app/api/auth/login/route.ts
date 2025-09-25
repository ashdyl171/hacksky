import { type NextRequest, NextResponse } from "next/server"
import { authenticateUser, generateToken } from "@/lib/auth"
import { getDatabase } from "@/lib/database"

export async function POST(request: NextRequest) {
  try {
    const { email, password, username } = await request.json()

    // Initialize database
    const db = await getDatabase()
    console.log('Database initialized for login')

    // Support both email and username login
    let loginEmail = email

    // Basic validation
    if (!loginEmail && !username) {
      return NextResponse.json(
        { error: "Missing email or username" },
        { status: 400 }
      )
    }

    if (!password) {
      return NextResponse.json(
        { error: "Missing password" },
        { status: 400 }
      )
    }

    // If only username provided, try to find email in database
    if (!loginEmail && username) {
      try {
        const userRecord = await db.get(
          'SELECT email FROM users WHERE username = ?',
          [username]
        )
        if (userRecord) {
          loginEmail = userRecord.email
        } else {
          loginEmail = `${username}@demo.com` // Fallback for demo
        }
      } catch (dbError) {
        loginEmail = `${username}@demo.com` // Fallback for demo
      }
    }

    // Authenticate user
    const user = await authenticateUser(loginEmail, password)
    
    if (!user) {
      return NextResponse.json(
        { error: "Invalid credentials" },
        { status: 401 }
      )
    }

    // Generate JWT token
    const token = generateToken(user.id)

    console.log(`User logged in successfully: ${user.username} (ID: ${user.id})`)

    return NextResponse.json({
      success: true,
      user: {
        id: user.id.toString(),
        username: user.username,
        email: user.email,
        profilePic: user.profile_pic || "/placeholder.svg",
        bio: user.bio,
        createdAt: new Date().toISOString()
      },
      token
    })

  } catch (error) {
    console.error('Login error:', error)
    return NextResponse.json(
      { error: "Login failed: " + (error instanceof Error ? error.message : 'Unknown error') },
      { status: 500 }
    )
  }
}
