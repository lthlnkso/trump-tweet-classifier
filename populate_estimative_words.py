#!/usr/bin/env python3
"""
Script to populate the estimative_words table with initial data.
Run this once to seed the database with estimative probability phrases.
"""

from database import db

# Common estimative probability words and phrases
estimative_words = [
    # CIA Intelligence Community Standard Phrases
    ("almost no chance", "cia_standard", 1),
    ("very unlikely", "cia_standard", 1),
    ("unlikely", "cia_standard", 1),
    ("probably not", "cia_standard", 1),
    ("even chance", "cia_standard", 1),
    ("possible", "cia_standard", 1),
    ("probably", "cia_standard", 1),
    ("likely", "cia_standard", 1),
    ("very likely", "cia_standard", 1),
    ("almost certain", "cia_standard", 1),
    
    # Common variations
    ("highly unlikely", "common", 1),
    ("quite certain", "common", 1),
    ("very probable", "common", 1),
    ("improbable", "common", 1),
    ("doubtful", "common", 1),
    ("conceivable", "common", 1),
    ("plausible", "common", 1),
    ("credible", "common", 1),
    ("feasible", "common", 1),
    ("remote chance", "common", 2),
    ("slim chance", "common", 2),
    ("good chance", "common", 1),
    ("fair chance", "common", 1),
    ("decent chance", "common", 1),
    ("strong chance", "common", 1),
    ("excellent chance", "common", 1),
    
    # More challenging phrases
    ("virtually certain", "challenging", 3),
    ("virtually impossible", "challenging", 3),
    ("extremely unlikely", "challenging", 2),
    ("highly probable", "challenging", 2),
    ("marginally likely", "challenging", 3),
    ("moderately likely", "challenging", 2),
    ("reasonably certain", "challenging", 2),
    ("somewhat likely", "challenging", 2),
    ("somewhat unlikely", "challenging", 2),
    ("highly improbable", "challenging", 3),
    ("exceptionally unlikely", "challenging", 3),
    ("overwhelmingly likely", "challenging", 3),
    
    # Informal expressions
    ("no way", "informal", 2),
    ("fat chance", "informal", 2),
    ("sure thing", "informal", 1),
    ("long shot", "informal", 2),
    ("safe bet", "informal", 1),
    ("coin flip", "informal", 1),
    ("pretty sure", "informal", 1),
    ("pretty unlikely", "informal", 1),
    ("dead certain", "informal", 2),
    ("zero chance", "informal", 1),
]

def populate_words():
    """Add estimative words to the database."""
    print("Populating estimative words database...")
    
    count = 0
    for word_phrase, category, difficulty in estimative_words:
        try:
            word_id = db.add_estimative_word(word_phrase, category, difficulty)
            print(f"Added: '{word_phrase}' (ID: {word_id})")
            count += 1
        except Exception as e:
            print(f"Skipped '{word_phrase}': {e}")
    
    print(f"\nSuccessfully added {count} estimative words to the database!")
    
    # Test getting a random word
    print("\nTesting random word retrieval:")
    test_word = db.get_random_estimative_word()
    if test_word:
        print(f"Random word: '{test_word['word_phrase']}' (Category: {test_word['category']})")
    else:
        print("Error: No words found!")

if __name__ == "__main__":
    populate_words()
