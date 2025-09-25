import { getDatabase } from './database'

export interface PostWithUser {
  id: number
  user_id: number
  username: string
  profile_pic?: string
  media_url: string
  media_type: string
  caption?: string
  deepfake_score: number
  analysis_result?: string
  risk_level: 'low' | 'medium' | 'high'
  ai_confirmed: boolean
  likes_count: number
  comments_count: number
  is_liked: boolean
  created_at: string
}

export interface StoryWithUser {
  id: number
  user_id: number
  username: string
  profile_pic?: string
  media_url: string
  media_type: string
  deepfake_score: number
  analysis_result?: string
  risk_level: 'low' | 'medium' | 'high'
  ai_confirmed: boolean
  viewed: boolean
  created_at: string
  expires_at: string
}

export async function createPost(
  userId: number,
  mediaUrl: string,
  mediaType: string,
  caption?: string,
  analysisResult?: any,
  aiConfirmed: boolean = false
): Promise<number | null> {
  try {
    const db = await getDatabase()
    
    // Calculate the correct deepfake score based on backend prediction
    let deepfakeScore = 0
    if (analysisResult?.backendInfo) {
      const confidence = analysisResult.backendInfo.confidence
      const prediction = analysisResult.backendInfo.prediction
      
      // If backend says "Real", then deepfake score is (1 - confidence)
      // If backend says "Deepfake", then deepfake score is confidence
      if (prediction === "Real") {
        deepfakeScore = Math.round((1 - confidence) * 100)
      } else {
        deepfakeScore = Math.round(confidence * 100)
      }
    } else {
      // Fallback to analysis confidence (already in percentage)
      deepfakeScore = analysisResult?.confidence || 0
    }
    
    const riskLevel = analysisResult?.riskLevel || 'low'
    const analysisJson = analysisResult ? JSON.stringify(analysisResult) : null
    
    const result = await db.run(
      `INSERT INTO posts 
       (user_id, media_url, media_type, caption, deepfake_score, analysis_result, risk_level, ai_confirmed) 
       VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
      [userId, mediaUrl, mediaType, caption, deepfakeScore, analysisJson, riskLevel, aiConfirmed]
    )
    
    return result.lastID || null
  } catch (error) {
    console.error('Error creating post:', error)
    return null
  }
}

export async function createStory(
  userId: number,
  mediaUrl: string,
  mediaType: string,
  analysisResult?: any,
  aiConfirmed: boolean = false
): Promise<number | null> {
  try {
    const db = await getDatabase()
    
    const deepfakeScore = analysisResult?.confidence || 0
    const riskLevel = analysisResult?.riskLevel || 'low'
    const analysisJson = analysisResult ? JSON.stringify(analysisResult) : null
    
    // Stories expire after 24 hours
    const expiresAt = new Date()
    expiresAt.setHours(expiresAt.getHours() + 24)
    
    const result = await db.run(
      `INSERT INTO stories 
       (user_id, media_url, media_type, deepfake_score, analysis_result, risk_level, ai_confirmed, expires_at) 
       VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
      [userId, mediaUrl, mediaType, deepfakeScore, analysisJson, riskLevel, aiConfirmed, expiresAt.toISOString()]
    )
    
    return result.lastID || null
  } catch (error) {
    console.error('Error creating story:', error)
    return null
  }
}

export async function getPosts(userId?: number, limit: number = 20, offset: number = 0): Promise<PostWithUser[]> {
  try {
    const db = await getDatabase()
    
    // Optimized query with proper indexing
    const posts = await db.all(`
      SELECT 
        p.id, p.user_id, p.media_url, p.media_type, p.caption,
        p.deepfake_score, p.analysis_result, p.risk_level, p.ai_confirmed,
        p.likes_count, p.comments_count, p.created_at,
        u.username, u.profile_pic,
        CASE WHEN l.id IS NOT NULL THEN 1 ELSE 0 END as is_liked
      FROM posts p
      INNER JOIN users u ON p.user_id = u.id
      LEFT JOIN likes l ON p.id = l.post_id AND l.user_id = ?
      ORDER BY p.created_at DESC
      LIMIT ? OFFSET ?
    `, [userId || null, limit, offset])
    
    return posts.map(post => ({
      ...post,
      is_liked: Boolean(post.is_liked),
      analysis_result: post.analysis_result ? JSON.parse(post.analysis_result) : null
    }))
  } catch (error) {
    console.error('Error getting posts:', error)
    return []
  }
}

export async function getStories(userId?: number): Promise<StoryWithUser[]> {
  try {
    const db = await getDatabase()
    
    const stories = await db.all(`
      SELECT 
        s.*,
        u.username,
        u.profile_pic,
        CASE WHEN sv.id IS NOT NULL THEN 1 ELSE 0 END as viewed
      FROM stories s
      JOIN users u ON s.user_id = u.id
      LEFT JOIN story_views sv ON s.id = sv.story_id AND sv.user_id = ?
      WHERE s.expires_at > datetime('now')
      ORDER BY s.created_at DESC
    `, [userId])
    
    return stories.map(story => ({
      ...story,
      viewed: Boolean(story.viewed),
      analysis_result: story.analysis_result ? JSON.parse(story.analysis_result) : null
    }))
  } catch (error) {
    console.error('Error getting stories:', error)
    return []
  }
}

export async function likePost(postId: number, userId: number): Promise<boolean> {
  try {
    const db = await getDatabase()
    
    // Check if already liked
    const existingLike = await db.get(
      'SELECT id FROM likes WHERE post_id = ? AND user_id = ?',
      [postId, userId]
    )
    
    if (existingLike) {
      // Unlike
      await db.run('DELETE FROM likes WHERE post_id = ? AND user_id = ?', [postId, userId])
      await db.run('UPDATE posts SET likes_count = likes_count - 1 WHERE id = ?', [postId])
    } else {
      // Like
      await db.run('INSERT INTO likes (post_id, user_id) VALUES (?, ?)', [postId, userId])
      await db.run('UPDATE posts SET likes_count = likes_count + 1 WHERE id = ?', [postId])
    }
    
    return true
  } catch (error) {
    console.error('Error toggling like:', error)
    return false
  }
}

export async function addComment(postId: number, userId: number, content: string): Promise<number | null> {
  try {
    const db = await getDatabase()
    
    const result = await db.run(
      'INSERT INTO comments (post_id, user_id, content) VALUES (?, ?, ?)',
      [postId, userId, content]
    )
    
    if (result.lastID) {
      await db.run('UPDATE posts SET comments_count = comments_count + 1 WHERE id = ?', [postId])
    }
    
    return result.lastID || null
  } catch (error) {
    console.error('Error adding comment:', error)
    return null
  }
}

export async function viewStory(storyId: number, userId: number): Promise<boolean> {
  try {
    const db = await getDatabase()
    
    await db.run(
      'INSERT OR IGNORE INTO story_views (story_id, user_id) VALUES (?, ?)',
      [storyId, userId]
    )
    
    return true
  } catch (error) {
    console.error('Error marking story as viewed:', error)
    return false
  }
}

export async function logAIAnalysis(
  contentId: number,
  contentType: 'post' | 'story',
  analysisResult: any,
  userConfirmed: boolean,
  userId: number
): Promise<boolean> {
  try {
    const db = await getDatabase()
    
    await db.run(
      `INSERT INTO ai_analysis_logs 
       (content_id, content_type, analysis_result, user_confirmed, user_id) 
       VALUES (?, ?, ?, ?, ?)`,
      [contentId, contentType, JSON.stringify(analysisResult), userConfirmed, userId]
    )
    
    return true
  } catch (error) {
    console.error('Error logging AI analysis:', error)
    return false
  }
}
