"""
Verify that all songs in the songs folder have been matched to training_data.
"""

import csv
from pathlib import Path
from collections import defaultdict


def normalize_text(text):
    """Normalize text for matching."""
    import re
    text = text.lower()
    text = re.sub(r'\.(mp3|wav|mp4)$', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[_\-/\\]', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_song_from_filename(filename):
    """Extract song name from filename."""
    name = Path(filename).stem
    if ' - ' in name:
        parts = name.split(' - ')
        song_name = parts[0].strip()
    else:
        song_name = name
    return song_name


def load_training_data_songs(training_dir):
    """Load all songs currently in training_data."""
    training_path = Path(training_dir)
    matched_songs = set()

    for folder in training_path.iterdir():
        if not folder.is_dir():
            continue

        mp3_files = list(folder.glob('*.mp3'))
        if mp3_files:
            # Get the original filename (we'll normalize it for comparison)
            for mp3 in mp3_files:
                song_name = extract_song_from_filename(mp3.name)
                matched_songs.add(normalize_text(song_name))

    return matched_songs


def verify_all_matched(songs_dir, training_dir):
    """Verify all songs from songs folder are in training_data."""

    print("=" * 80)
    print("VERIFICATION: All Songs Matched?")
    print("=" * 80)

    # Get all MP3s in songs folder
    songs_path = Path(songs_dir)
    all_mp3s = sorted(list(songs_path.glob('*.mp3')))

    # Get all songs in training_data
    matched_songs = load_training_data_songs(training_dir)

    print(f"\nTotal MP3s in songs folder: {len(all_mp3s)}")
    print(f"Total folders in training_data: {len(list(Path(training_dir).iterdir()))}")
    print(f"Matched songs in training_data: {len(matched_songs)}")

    # Check each song
    unmatched = []
    for mp3_path in all_mp3s:
        song_name = extract_song_from_filename(mp3_path.name)
        norm_song = normalize_text(song_name)

        if norm_song not in matched_songs:
            unmatched.append(mp3_path.name)

    print("\n" + "=" * 80)
    if unmatched:
        print(f"UNMATCHED SONGS: {len(unmatched)}")
        print("=" * 80)
        for filename in unmatched:
            print(f"  - {filename}")
    else:
        print("ALL SONGS MATCHED!")
        print("=" * 80)
        print(f"\nAll {len(all_mp3s)} songs from the songs folder have been successfully")
        print("matched and organized into training_data folders.")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total songs in songs folder: {len(all_mp3s)}")
    print(f"Matched to training_data: {len(all_mp3s) - len(unmatched)}")
    print(f"Still unmatched: {len(unmatched)}")

    return len(unmatched) == 0


if __name__ == '__main__':
    songs_dir = '../songs'
    training_dir = '../training_data'

    all_matched = verify_all_matched(songs_dir, training_dir)

    if all_matched:
        print("\n[SUCCESS] Dataset is ready for feature extraction!")
    else:
        print("\n[WARNING] Some songs still need to be matched.")
