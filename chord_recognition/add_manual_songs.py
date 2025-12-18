"""
Manually add specific songs to their designated song_id folders.
"""

import csv
import shutil
from pathlib import Path


def sanitize_filename(filename):
    """
    Remove illegal characters from Windows filenames.
    Illegal characters: < > : " / \ | ? *
    """
    illegal_chars = '<>:"/\\|?*'
    for char in illegal_chars:
        filename = filename.replace(char, '')
    return filename


def load_csv_lookup(csv_path):
    """Load CSV and create song_id lookup."""
    lookup = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            lookup[row['song_id']] = (row['song_name'], row['artist'])
    return lookup


def add_manual_songs(songs_dir, training_dir, csv_path):
    """Add manually specified songs to their folders."""

    # Manual mappings: filename -> song_id
    manual_mappings = {
        "Kicks (feat. Mark Lindsay) - Single Version - Paul Revere & The Raiders, Mark Lindsay.mp3": "1181",
        "Make Me Smile Now More Than Ever - 2002 Remaster - Chicago.mp3": "1085",
        "Walk Like A Man (You Can Call Me Your Man) - Remastered 1999 - Grand Funk Railroad.mp3": "0762",
        "Wendy (Stereo) - The Beach Boys.mp3": "0479"
    }

    # Load CSV data
    print("Loading CSV data...")
    csv_lookup = load_csv_lookup(csv_path)

    songs_path = Path(songs_dir)
    training_path = Path(training_dir)

    print("\n" + "=" * 80)
    print("MANUALLY ADDING SONGS")
    print("=" * 80)

    added_count = 0

    for mp3_filename, song_id in manual_mappings.items():
        mp3_path = songs_path / mp3_filename

        if not mp3_path.exists():
            print(f"\n[ERROR] File not found: {mp3_filename}")
            continue

        if song_id not in csv_lookup:
            print(f"\n[ERROR] song_id {song_id} not found in CSV")
            continue

        csv_song_name, csv_artist = csv_lookup[song_id]

        # Create folder
        song_folder = training_path / song_id
        song_folder.mkdir(parents=True, exist_ok=True)

        # Create standardized filename (sanitized for Windows)
        new_filename = f"{csv_song_name} - {csv_artist}.mp3"
        new_filename = sanitize_filename(new_filename)
        dest_path = song_folder / new_filename

        # Copy file
        shutil.copy2(mp3_path, dest_path)
        added_count += 1

        print(f"\n[OK] Added song_id={song_id}")
        print(f"  Source: {mp3_filename}")
        print(f"  Dest:   {song_id}/{new_filename}")
        print(f"  CSV:    {csv_song_name} - {csv_artist}")

    print("\n" + "=" * 80)
    print(f"COMPLETE: Added {added_count}/4 songs")
    print("=" * 80)


if __name__ == '__main__':
    csv_path = '../mcgill_billboard_spotify.csv'
    songs_dir = '../songs'
    training_dir = '../training_data'

    add_manual_songs(songs_dir, training_dir, csv_path)
