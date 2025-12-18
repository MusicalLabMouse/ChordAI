"""
Check which songs from songs folder are still unmatched.
Uses the same matching logic as organize_dataset.py
"""

import csv
import re
from pathlib import Path
from collections import defaultdict


def normalize_text(text):
    """Normalize text for matching."""
    text = text.lower()
    text = re.sub(r'\.(mp3|wav|mp4)$', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[_\-/\\]', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_song_artist_from_filename(filename):
    """Extract song name and artist from filename."""
    name = Path(filename).stem

    if ' - ' in name:
        parts = name.split(' - ')
        song_name = parts[0].strip()
        artist = ' - '.join(parts[1:]).strip()

        # Remove version info from artist
        version_patterns = [
            r'\s*-?\s*remaster(ed)?\s*\d*',
            r'\s*-?\s*live\s*.*',
            r'\s*-?\s*\d+\s*remaster',
            r'\s*-?\s*single\s*version',
            r'\s*-?\s*mono(\s*version)?',
            r'\s*-?\s*digital(ly)?\s*remaster(ed)?',
            r'\s*-?\s*\d{4}\s*remaster',
        ]

        for pattern in version_patterns:
            artist = re.sub(pattern, '', artist, flags=re.IGNORECASE).strip()
    else:
        song_name = name
        artist = ""

    return song_name, artist


def load_csv_data(csv_path):
    """Load CSV and create lookup structures."""
    lookup_by_song = defaultdict(list)

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            song_id = row['song_id']
            song_name = row['song_name']
            artist = row['artist']

            norm_song = normalize_text(song_name)
            lookup_by_song[norm_song].append((song_id, song_name, artist))

    return lookup_by_song


def check_unmatched(csv_path, songs_dir, training_dir):
    """Check which songs are still unmatched."""

    print("=" * 80)
    print("CHECKING UNMATCHED SONGS")
    print("=" * 80)

    # Load CSV data
    lookup_by_song = load_csv_data(csv_path)

    # Get all MP3s in songs folder
    songs_path = Path(songs_dir)
    mp3_files = sorted(list(songs_path.glob('*.mp3')))

    # Get all song_ids that have folders in training_data
    training_path = Path(training_dir)
    matched_song_ids = set()
    for folder in training_path.iterdir():
        if folder.is_dir():
            matched_song_ids.add(folder.name)

    print(f"\nTotal MP3s in songs folder: {len(mp3_files)}")
    print(f"Total folders in training_data: {len(matched_song_ids)}")

    # Check each MP3
    unmatched = []
    matched_count = 0

    for mp3_path in mp3_files:
        mp3_filename = mp3_path.name
        song_name, artist = extract_song_artist_from_filename(mp3_filename)
        norm_song = normalize_text(song_name)

        if norm_song in lookup_by_song:
            matched_count += 1
        else:
            unmatched.append(mp3_filename)

    print(f"\n" + "=" * 80)
    if unmatched:
        print(f"UNMATCHED SONGS: {len(unmatched)}")
        print("=" * 80)
        print("\nThese songs could not be matched to any song in the CSV:\n")
        for filename in unmatched:
            print(f"  - {filename}")
    else:
        print("ALL SONGS MATCHED!")
        print("=" * 80)

    print(f"\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total songs in songs folder: {len(mp3_files)}")
    print(f"Matched to CSV: {matched_count}")
    print(f"Unmatched (not in CSV): {len(unmatched)}")
    print(f"Folders in training_data: {len(matched_song_ids)}")
    print("=" * 80)

    if len(unmatched) == 0:
        print("\n[SUCCESS] All songs have been matched!")
    else:
        print(f"\n[INFO] {len(unmatched)} songs are not in the CSV and cannot be matched.")

    return unmatched


if __name__ == '__main__':
    csv_path = '../mcgill_billboard_spotify.csv'
    songs_dir = '../songs'
    training_dir = '../training_data'

    check_unmatched(csv_path, songs_dir, training_dir)
