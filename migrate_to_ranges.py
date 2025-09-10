#!/usr/bin/env python3
"""
Migration script to update the estimative_responses table to support ranges.
This will backup existing data, update the schema, and provide a clean migration.
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = "data/trump_classifier.db"
BACKUP_PATH = f"data/trump_classifier_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"

def backup_database():
    """Create a backup of the current database."""
    print("Creating database backup...")
    if os.path.exists(DB_PATH):
        import shutil
        shutil.copy2(DB_PATH, BACKUP_PATH)
        print(f"✓ Backup created: {BACKUP_PATH}")
    else:
        print("No existing database found - will create new one")

def migrate_schema():
    """Migrate the estimative_responses table to support ranges."""
    print("Migrating database schema...")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Check if we need to migrate
        cursor.execute("PRAGMA table_info(estimative_responses)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'probability_low' in columns:
            print("✓ Database already has range columns - no migration needed")
            return
        
        # Check if old table exists and has data
        cursor.execute("SELECT COUNT(*) FROM estimative_responses")
        old_count = cursor.fetchone()[0]
        print(f"Found {old_count} existing responses")
        
        if old_count > 0:
            # Backup old data
            cursor.execute("""
                CREATE TABLE estimative_responses_old AS 
                SELECT * FROM estimative_responses
            """)
            print("✓ Backed up existing response data")
        
        # Drop the old table
        cursor.execute("DROP TABLE estimative_responses")
        print("✓ Dropped old table")
        
        # Create new table with range support
        cursor.execute("""
            CREATE TABLE estimative_responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                word_id INTEGER NOT NULL,
                user_id TEXT NOT NULL,
                probability_low REAL NOT NULL CHECK (probability_low >= 0 AND probability_low <= 100),
                probability_high REAL NOT NULL CHECK (probability_high >= 0 AND probability_high <= 100),
                probability_midpoint REAL NOT NULL CHECK (probability_midpoint >= 0 AND probability_midpoint <= 100),
                response_time_ms INTEGER,
                user_agent TEXT,
                user_ip TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (word_id) REFERENCES estimative_words(id),
                CHECK (probability_low <= probability_midpoint),
                CHECK (probability_midpoint <= probability_high)
            )
        """)
        print("✓ Created new range-based table")
        
        # Migrate old data if it exists
        if old_count > 0:
            print("Migrating old single-value responses to ranges...")
            cursor.execute("""
                INSERT INTO estimative_responses (
                    word_id, user_id, probability_low, probability_high, probability_midpoint,
                    response_time_ms, user_agent, user_ip, created_at
                )
                SELECT 
                    word_id, user_id, 
                    CASE 
                        WHEN probability <= 10 THEN 0
                        ELSE probability - 10 
                    END as probability_low,
                    CASE 
                        WHEN probability >= 90 THEN 100
                        ELSE probability + 10 
                    END as probability_high,
                    probability as probability_midpoint,
                    response_time_ms, user_agent, user_ip, created_at
                FROM estimative_responses_old
            """)
            migrated_count = cursor.rowcount
            print(f"✓ Migrated {migrated_count} responses with synthetic ranges (±10% around original value)")
            
            # Drop the backup table
            cursor.execute("DROP TABLE estimative_responses_old")
        
        # Recreate indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_estimative_responses_word_id ON estimative_responses(word_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_estimative_responses_user_id ON estimative_responses(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_estimative_responses_created_at ON estimative_responses(created_at)")
        print("✓ Recreated indexes")
        
        conn.commit()
        print("✅ Migration completed successfully!")
        
    except Exception as e:
        conn.rollback()
        print(f"❌ Migration failed: {e}")
        raise
    finally:
        conn.close()

def test_new_schema():
    """Test the new schema by inserting a test record."""
    print("\nTesting new schema...")
    
    from database import db
    
    try:
        # Test getting a word
        word = db.get_random_estimative_word()
        if not word:
            print("❌ No words available for testing")
            return
            
        print(f"Using test word: '{word['word_phrase']}'")
        
        # Test inserting a range response
        response_id = db.log_estimative_response(
            word_id=word['id'],
            user_id="migration_test_user",
            probability_low=75.0,
            probability_high=95.0,
            probability_midpoint=85.0,
            response_time_ms=3000,
            user_agent="Migration Test",
            user_ip="127.0.0.1"
        )
        
        print(f"✓ Successfully inserted test response with ID: {response_id}")
        
        # Test stats
        stats = db.get_estimative_stats()
        print(f"✓ Stats retrieved: {stats.get('total_responses', 0)} total responses")
        
        print("✅ New schema is working correctly!")
        
    except Exception as e:
        print(f"❌ Schema test failed: {e}")

if __name__ == "__main__":
    print("🔄 Estimative Probability Database Migration")
    print("=" * 50)
    
    backup_database()
    migrate_schema()
    test_new_schema()
    
    print("\n🎉 Migration complete! The Estimative Probability system now supports probability ranges.")
    print("You can now test the updated system at: http://localhost:8001/estimative")
