"""
Simple SQLite persistence layer for Trump Tweet Classifier.

This module provides a lightweight database interface without heavy ORM overhead,
designed to handle decent traffic while keeping things simple.
"""

import sqlite3
import threading
import os
from datetime import datetime
from contextlib import contextmanager
from typing import Optional, Dict, Any, List
import json


class Database:
    """
    Thread-safe SQLite database manager.
    """
    
    def __init__(self, db_path: str = "data/trump_classifier.db"):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._local = threading.local()
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database schema
        self._initialize_schema()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0
            )
            self._local.connection.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrency
            self._local.connection.execute("PRAGMA journal_mode=WAL")
            self._local.connection.execute("PRAGMA synchronous=NORMAL")
            self._local.connection.execute("PRAGMA cache_size=10000")
            
        return self._local.connection
    
    @contextmanager
    def get_cursor(self):
        """Context manager for database operations."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()
    
    def _initialize_schema(self):
        """Create database tables if they don't exist."""
        schema_sql = """
        -- Submissions table: log every classification request
        CREATE TABLE IF NOT EXISTS submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_ip TEXT NOT NULL,
            user_agent TEXT,
            text_content TEXT NOT NULL,
            text_length INTEGER NOT NULL,
            classification TEXT NOT NULL,
            confidence REAL NOT NULL,
            trump_level TEXT NOT NULL,
            trump_score INTEGER NOT NULL,
            processing_time_ms REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            session_id TEXT
        );
        
        -- Feedback table: user feedback on classifications
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            submission_id INTEGER,
            user_ip TEXT NOT NULL,
            agrees_with_rating BOOLEAN NOT NULL,
            feedback_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            session_id TEXT,
            FOREIGN KEY (submission_id) REFERENCES submissions(id)
        );
        
        -- Usage stats table: aggregate statistics
        CREATE TABLE IF NOT EXISTS usage_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            hour INTEGER NOT NULL,
            total_submissions INTEGER DEFAULT 0,
            total_feedback INTEGER DEFAULT 0,
            avg_trump_score REAL DEFAULT 0,
            unique_ips INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(date, hour)
        );
        
        -- Create indexes for better performance
        CREATE INDEX IF NOT EXISTS idx_submissions_created_at ON submissions(created_at);
        CREATE INDEX IF NOT EXISTS idx_submissions_user_ip ON submissions(user_ip);
        CREATE INDEX IF NOT EXISTS idx_submissions_trump_score ON submissions(trump_score);
        CREATE INDEX IF NOT EXISTS idx_feedback_created_at ON feedback(created_at);
        CREATE INDEX IF NOT EXISTS idx_feedback_submission_id ON feedback(submission_id);
        CREATE INDEX IF NOT EXISTS idx_usage_stats_date_hour ON usage_stats(date, hour);
        """
        
        with self.get_cursor() as cursor:
            cursor.executescript(schema_sql)
            
        # Apply any schema migrations
        self._apply_migrations()
    
    def _apply_migrations(self):
        """Apply database schema migrations."""
        with self.get_cursor() as cursor:
            # Check if sharing columns exist
            cursor.execute("PRAGMA table_info(submissions)")
            columns = [column[1] for column in cursor.fetchall()]
            
            # Add sharing columns if they don't exist
            if 'is_public' not in columns:
                cursor.execute("ALTER TABLE submissions ADD COLUMN is_public BOOLEAN DEFAULT 0")
                
            if 'share_hash' not in columns:
                cursor.execute("ALTER TABLE submissions ADD COLUMN share_hash TEXT")
                
            if 'share_title' not in columns:
                cursor.execute("ALTER TABLE submissions ADD COLUMN share_title TEXT")
                
            if 'share_description' not in columns:
                cursor.execute("ALTER TABLE submissions ADD COLUMN share_description TEXT")
            
            # Add unique constraint to share_hash if it doesn't exist
            try:
                cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_share_hash ON submissions(share_hash)")
            except Exception:
                pass  # Index might already exist
    
    def log_submission(self, 
                      user_ip: str,
                      user_agent: str,
                      text_content: str,
                      classification: str,
                      confidence: float,
                      trump_level: str,
                      trump_score: int,
                      processing_time_ms: float,
                      session_id: Optional[str] = None) -> int:
        """
        Log a classification submission.
        
        Returns:
            submission_id: The ID of the logged submission
        """
        with self.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO submissions (
                    user_ip, user_agent, text_content, text_length,
                    classification, confidence, trump_level, trump_score,
                    processing_time_ms, session_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user_ip, user_agent, text_content, len(text_content),
                classification, confidence, trump_level, trump_score,
                processing_time_ms, session_id
            ))
            return cursor.lastrowid
    
    def log_feedback(self,
                    submission_id: Optional[int],
                    user_ip: str,
                    agrees_with_rating: bool,
                    feedback_message: Optional[str] = None,
                    session_id: Optional[str] = None) -> int:
        """
        Log user feedback.
        
        Returns:
            feedback_id: The ID of the logged feedback
        """
        with self.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO feedback (
                    submission_id, user_ip, agrees_with_rating,
                    feedback_message, session_id
                ) VALUES (?, ?, ?, ?, ?)
            """, (submission_id, user_ip, agrees_with_rating, feedback_message, session_id))
            return cursor.lastrowid
    
    def make_submission_shareable(self, 
                                submission_id: int, 
                                share_hash: str,
                                share_title: str = None,
                                share_description: str = None) -> bool:
        """
        Make a submission publicly shareable.
        
        Returns:
            bool: True if successful, False otherwise
        """
        with self.get_cursor() as cursor:
            cursor.execute("""
                UPDATE submissions 
                SET is_public = 1, share_hash = ?, share_title = ?, share_description = ?
                WHERE id = ?
            """, (share_hash, share_title, share_description, submission_id))
            return cursor.rowcount > 0
    
    def get_submission_by_share_hash(self, share_hash: str) -> Optional[Dict[str, Any]]:
        """
        Get a public submission by its share hash.
        
        Returns:
            Dict with submission data or None if not found/not public
        """
        with self.get_cursor() as cursor:
            cursor.execute("""
                SELECT id, text_content, text_length, classification, confidence,
                       trump_level, trump_score, created_at, share_title, share_description
                FROM submissions 
                WHERE share_hash = ? AND is_public = 1
            """, (share_hash,))
            
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_sharing_stats(self) -> Dict[str, Any]:
        """Get statistics about shared submissions."""
        with self.get_cursor() as cursor:
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_shared,
                    AVG(trump_score) as avg_shared_score,
                    COUNT(CASE WHEN trump_score >= 70 THEN 1 END) as high_score_shares
                FROM submissions 
                WHERE is_public = 1
            """)
            
            row = cursor.fetchone()
            return dict(row) if row else {}
    
    def get_submission_stats(self, days: int = 7) -> Dict[str, Any]:
        """Get submission statistics for the last N days."""
        with self.get_cursor() as cursor:
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_submissions,
                    COUNT(DISTINCT user_ip) as unique_users,
                    AVG(trump_score) as avg_trump_score,
                    AVG(confidence) as avg_confidence,
                    AVG(processing_time_ms) as avg_processing_time,
                    COUNT(CASE WHEN trump_score >= 70 THEN 1 END) as high_scores,
                    AVG(text_length) as avg_text_length
                FROM submissions 
                WHERE created_at >= datetime('now', '-{} days')
            """.format(days))
            
            row = cursor.fetchone()
            return dict(row) if row else {}
    
    def get_feedback_stats(self, days: int = 7) -> Dict[str, Any]:
        """Get feedback statistics for the last N days."""
        with self.get_cursor() as cursor:
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_feedback,
                    COUNT(CASE WHEN agrees_with_rating = 1 THEN 1 END) as positive_feedback,
                    COUNT(CASE WHEN agrees_with_rating = 0 THEN 1 END) as negative_feedback,
                    COUNT(CASE WHEN feedback_message IS NOT NULL THEN 1 END) as has_message
                FROM feedback 
                WHERE created_at >= datetime('now', '-{} days')
            """.format(days))
            
            row = cursor.fetchone()
            return dict(row) if row else {}
    
    def get_recent_submissions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent submissions (for admin/debugging)."""
        with self.get_cursor() as cursor:
            cursor.execute("""
                SELECT id, user_ip, text_content, classification, 
                       confidence, trump_level, trump_score, created_at
                FROM submissions 
                ORDER BY created_at DESC 
                LIMIT ?
            """, (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def cleanup_old_data(self, days: int = 90):
        """Clean up old data to keep database size manageable."""
        with self.get_cursor() as cursor:
            # Delete old submissions and feedback
            cursor.execute("""
                DELETE FROM submissions 
                WHERE created_at < datetime('now', '-{} days')
            """.format(days))
            
            cursor.execute("""
                DELETE FROM feedback 
                WHERE created_at < datetime('now', '-{} days')
            """.format(days))
            
            # Vacuum to reclaim space
            cursor.execute("VACUUM")
    
    def close(self):
        """Close database connections."""
        if hasattr(self._local, 'connection'):
            self._local.connection.close()


# Global database instance
db = Database()
