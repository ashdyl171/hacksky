import bcrypt from 'bcryptjs'
import jwt from 'jsonwebtoken'
import { getDatabase } from './database'

const JWT_SECRET = process.env.JWT_SECRET || 'your-secret-key-change-in-production'

export interface AuthUser {
  id: number
  username: string
  email: string
  profile_pic?: string
  bio?: string
}

export async function hashPassword(password: string): Promise<string> {
  return await bcrypt.hash(password, 10)
}

export async function comparePassword(password: string, hash: string): Promise<boolean> {
  return await bcrypt.compare(password, hash)
}

export function generateToken(userId: number): string {
  return jwt.sign({ userId }, JWT_SECRET, { expiresIn: '7d' })
}

export function verifyToken(token: string): { userId: number } | null {
  try {
    const decoded = jwt.verify(token, JWT_SECRET) as { userId: number }
    return decoded
  } catch (error) {
    return null
  }
}

export async function createUser(
  username: string, 
  email: string, 
  password: string
): Promise<AuthUser | null> {
  try {
    const db = await getDatabase()
    const passwordHash = await hashPassword(password)
    
    const result = await db.run(
      'INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
      [username, email, passwordHash]
    )
    
    if (result.lastID) {
      const user = await db.get(
        'SELECT id, username, email, profile_pic, bio FROM users WHERE id = ?',
        [result.lastID]
      )
      return user as AuthUser
    }
    
    return null
  } catch (error) {
    console.error('Error creating user:', error)
    return null
  }
}

export async function authenticateUser(
  email: string, 
  password: string
): Promise<AuthUser | null> {
  try {
    const db = await getDatabase()
    const user = await db.get(
      'SELECT id, username, email, password_hash, profile_pic, bio FROM users WHERE email = ?',
      [email]
    )
    
    if (!user) return null
    
    const isValid = await comparePassword(password, user.password_hash)
    if (!isValid) return null
    
    return {
      id: user.id,
      username: user.username,
      email: user.email,
      profile_pic: user.profile_pic,
      bio: user.bio
    }
  } catch (error) {
    console.error('Error authenticating user:', error)
    return null
  }
}

export async function getUserById(id: number): Promise<AuthUser | null> {
  try {
    const db = await getDatabase()
    const user = await db.get(
      'SELECT id, username, email, profile_pic, bio FROM users WHERE id = ?',
      [id]
    )
    return user as AuthUser || null
  } catch (error) {
    console.error('Error getting user by ID:', error)
    return null
  }
}

export async function updateUserProfile(
  userId: number,
  updates: { username?: string; bio?: string; profile_pic?: string }
): Promise<AuthUser | null> {
  try {
    const db = await getDatabase()
    const setParts = []
    const values = []
    
    if (updates.username) {
      setParts.push('username = ?')
      values.push(updates.username)
    }
    if (updates.bio !== undefined) {
      setParts.push('bio = ?')
      values.push(updates.bio)
    }
    if (updates.profile_pic !== undefined) {
      setParts.push('profile_pic = ?')
      values.push(updates.profile_pic)
    }
    
    if (setParts.length === 0) return getUserById(userId)
    
    setParts.push('updated_at = CURRENT_TIMESTAMP')
    values.push(userId)
    
    await db.run(
      `UPDATE users SET ${setParts.join(', ')} WHERE id = ?`,
      values
    )
    
    return getUserById(userId)
  } catch (error) {
    console.error('Error updating user profile:', error)
    return null
  }
}
