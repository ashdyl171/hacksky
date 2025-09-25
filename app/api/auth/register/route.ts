import { NextRequest, NextResponse } from "next/server"
import { createUser, generateToken } from "@/lib/auth"
import { getDatabase } from "@/lib/database"

export async function POST(request: NextRequest) {
  try {
    const { username, email, password } = await request.json()

    // Basic validation
    if (!username || !email || !password) {
      return NextResponse.json(
        { error: "Missing required fields" },
        { status: 400 }
      )
    }

    // Validate email format
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
    if (!emailRegex.test(email)) {
      return NextResponse.json(
        { error: "Invalid email format" },
        { status: 400 }
      )
    }

    // Validate password strength
    if (password.length < 3) {
      return NextResponse.json(
        { error: "Password must be at least 3 characters long" },
        { status: 400 }
      )
    }

    // Initialize database
    const db = await getDatabase()
    console.log('Database initialized for registration')

    // Create user in database
    const user = await createUser(username, email, password)
    
    if (!user) {
      return NextResponse.json(
        { error: "User already exists or registration failed" },
        { status: 409 }
      )
    }

    // Generate JWT token
    const token = generateToken(user.id)

    console.log(`User registered successfully: ${username} (ID: ${user.id})`)

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
    console.error('Registration error:', error)
    return NextResponse.json(
      { error: "Registration failed: " + (error instanceof Error ? error.message : 'Unknown error') },
      { status: 500 }
    )
  }
}
