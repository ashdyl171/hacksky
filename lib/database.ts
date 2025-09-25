import sqlite3 from 'sqlite3'
import { open, Database } from 'sqlite'
import path from 'path'

let db: Database<sqlite3.Database, sqlite3.Statement> | null = null

export async function getDatabase() {
  if (!db) {
    const dbPath = path.join(process.cwd(), 'socialguard.db')
    
    db = await open({
      filename: dbPath,
      driver: sqlite3.Database
    })

    // Initialize tables
    await initializeTables()
  }
  
  return db
}

async function initializeTables() {
  if (!db) return

  // Users table
  await db.exec(`
    CREATE TABLE IF NOT EXISTS users (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      username TEXT UNIQUE NOT NULL,
      email TEXT UNIQUE NOT NULL,
      password_hash TEXT NOT NULL,
      profile_pic TEXT,
      bio TEXT,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
  `)

  // Posts table
  await db.exec(`
    CREATE TABLE IF NOT EXISTS posts (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id INTEGER NOT NULL,
      media_url TEXT NOT NULL,
      media_type TEXT NOT NULL,
      caption TEXT,
      deepfake_score INTEGER DEFAULT 0,
      analysis_result TEXT,
      risk_level TEXT DEFAULT 'low',
      ai_confirmed BOOLEAN DEFAULT FALSE,
      likes_count INTEGER DEFAULT 0,
      comments_count INTEGER DEFAULT 0,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY (user_id) REFERENCES users (id)
    )
  `)

  // Stories table
  await db.exec(`
    CREATE TABLE IF NOT EXISTS stories (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id INTEGER NOT NULL,
      media_url TEXT NOT NULL,
      media_type TEXT NOT NULL,
      deepfake_score INTEGER DEFAULT 0,
      analysis_result TEXT,
      risk_level TEXT DEFAULT 'low',
      ai_confirmed BOOLEAN DEFAULT FALSE,
      expires_at DATETIME NOT NULL,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY (user_id) REFERENCES users (id)
    )
  `)

  // Comments table
  await db.exec(`
    CREATE TABLE IF NOT EXISTS comments (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      post_id INTEGER NOT NULL,
      user_id INTEGER NOT NULL,
      content TEXT NOT NULL,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY (post_id) REFERENCES posts (id),
      FOREIGN KEY (user_id) REFERENCES users (id)
    )
  `)

  // Likes table
  await db.exec(`
    CREATE TABLE IF NOT EXISTS likes (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      post_id INTEGER NOT NULL,
      user_id INTEGER NOT NULL,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      UNIQUE(post_id, user_id),
      FOREIGN KEY (post_id) REFERENCES posts (id),
      FOREIGN KEY (user_id) REFERENCES users (id)
    )
  `)

  // Story views table
  await db.exec(`
    CREATE TABLE IF NOT EXISTS story_views (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      story_id INTEGER NOT NULL,
      user_id INTEGER NOT NULL,
      viewed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      UNIQUE(story_id, user_id),
      FOREIGN KEY (story_id) REFERENCES stories (id),
      FOREIGN KEY (user_id) REFERENCES users (id)
    )
  `)

  // AI analysis logs table
  await db.exec(`
    CREATE TABLE IF NOT EXISTS ai_analysis_logs (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      content_id INTEGER NOT NULL,
      content_type TEXT NOT NULL,
      analysis_result TEXT NOT NULL,
      user_confirmed BOOLEAN DEFAULT FALSE,
      user_id INTEGER NOT NULL,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY (user_id) REFERENCES users (id)
    )
  `)

  // Sessions table
  await db.exec(`
    CREATE TABLE IF NOT EXISTS sessions (
      id TEXT PRIMARY KEY,
      user_id INTEGER NOT NULL,
      expires_at DATETIME NOT NULL,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY (user_id) REFERENCES users (id)
    )
  `)

  // Create performance indexes
  await db.exec(`
    CREATE INDEX IF NOT EXISTS idx_posts_created_at ON posts(created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_posts_user_id ON posts(user_id);
    CREATE INDEX IF NOT EXISTS idx_likes_post_user ON likes(post_id, user_id);
    CREATE INDEX IF NOT EXISTS idx_comments_post_id ON comments(post_id);
    CREATE INDEX IF NOT EXISTS idx_stories_expires_at ON stories(expires_at);
    CREATE INDEX IF NOT EXISTS idx_stories_user_id ON stories(user_id);
  `)

  console.log('Database tables and indexes initialized successfully')
}

export interface User {
  id: number
  username: string
  email: string
  password_hash: string
  profile_pic?: string
  bio?: string
  created_at: string
  updated_at: string
}

export interface Post {
  id: number
  user_id: number
  media_url: string
  media_type: string
  caption?: string
  deepfake_score: number
  analysis_result?: string
  risk_level: 'low' | 'medium' | 'high'
  ai_confirmed: boolean
  likes_count: number
  comments_count: number
  created_at: string
  updated_at: string
}

export interface Story {
  id: number
  user_id: number
  media_url: string
  media_type: string
  deepfake_score: number
  analysis_result?: string
  risk_level: 'low' | 'medium' | 'high'
  ai_confirmed: boolean
  expires_at: string
  created_at: string
}

export interface Comment {
  id: number
  post_id: number
  user_id: number
  content: string
  created_at: string
}

export interface Like {
  id: number
  post_id: number
  user_id: number
  created_at: string
}

export interface AIAnalysisLog {
  id: number
  content_id: number
  content_type: 'post' | 'story'
  analysis_result: string
  user_confirmed: boolean
  user_id: number
  created_at: string
}
