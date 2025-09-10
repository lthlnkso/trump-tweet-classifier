#!/usr/bin/env python3
"""
Script to add new estimative probability words to the database.
This will add the specific words requested by the user.
"""

from database import db

# First set of words - professional/analytical terms
professional_words = [
    "Almost Certain",
    "Highly Likely", 
    "Very Good Chance",
    "We Believe",
    "Likely",
    "Probable",
    "Probably",
    "Better than Even",
    "About Even",
    "Probably Not",
    "We Doubt",
    "Unlikely",
    "Improbable",
    "Chances are Slight",
    "Little Chance",
    "Highly Unlikely",
    "Almost No Chance",
]

# Second set of words - standard probability expressions
standard_words = [
    "Definite",
    "Almost certain",
    "Highly probable",
    "A good chance",
    "Likely",
    "Quite likely",
    "Better than even",
    "Probable",
    "Possible",
    "Improbable",
    "Highly unlikely",
    "Unlikely",
    "Seldom",
    "Impossible",
    "Rare",
]

def add_word_sets():
    """Add the new word sets to the database."""
    print("Adding new estimative probability words...")
    
    added_count = 0
    duplicate_count = 0
    
    # Add professional words
    print("\n=== Adding Professional/Analytical Words ===")
    for word in professional_words:
        try:
            word_id = db.add_estimative_word(word, "professional", 2)
            print(f"✓ Added: '{word}' (ID: {word_id})")
            added_count += 1
        except Exception as e:
            print(f"⚠ Skipped '{word}': {e}")
            duplicate_count += 1
    
    # Add standard words
    print("\n=== Adding Standard Probability Words ===")
    for word in standard_words:
        try:
            word_id = db.add_estimative_word(word, "standard", 1)
            print(f"✓ Added: '{word}' (ID: {word_id})")
            added_count += 1
        except Exception as e:
            print(f"⚠ Skipped '{word}': {e}")
            duplicate_count += 1
    
    print(f"\n=== Summary ===")
    print(f"✓ Successfully added: {added_count} words")
    print(f"⚠ Skipped (duplicates): {duplicate_count} words")
    
    # Test the system
    print(f"\n=== Testing Random Word Retrieval ===")
    for i in range(3):
        test_word = db.get_random_estimative_word()
        if test_word:
            print(f"Random word {i+1}: '{test_word['word_phrase']}' (Category: {test_word['category']})")
        else:
            print(f"❌ Error getting random word {i+1}")

def show_all_words():
    """Display all words currently in the database."""
    print("\n=== All Words in Database ===")
    try:
        # Get all words by querying directly
        import sqlite3
        with db.get_cursor() as cursor:
            cursor.execute("""
                SELECT id, word_phrase, category, difficulty_level, active
                FROM estimative_words 
                ORDER BY category, word_phrase
            """)
            words = cursor.fetchall()
            
            current_category = None
            for word in words:
                if word[2] != current_category:  # category column
                    current_category = word[2]
                    print(f"\n--- {current_category.upper()} ---")
                
                status = "✓" if word[4] else "✗"  # active column
                print(f"{status} {word[0]:2d}. {word[1]} (Level {word[3]})")
                
        print(f"\nTotal words: {len(words)}")
        
    except Exception as e:
        print(f"❌ Error retrieving words: {e}")

if __name__ == "__main__":
    print("🎯 Estimative Probability Word Manager")
    print("=====================================")
    
    add_word_sets()
    show_all_words()
    
    print("\n🎉 Word addition complete! The API should now work with the new words.")
    print("💡 You can test at: http://localhost:8001/estimative")
