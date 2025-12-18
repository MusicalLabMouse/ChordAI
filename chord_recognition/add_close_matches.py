"""
Add the 23 close-matched songs to their designated song_id folders.
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


def add_close_matches(songs_dir, training_dir, csv_path):
    """Add close-matched songs to their folders."""

    # Manual mappings: filename -> song_id
    close_matches = {
        "A Love Song - Live - Anne Murray.mp3": "0917",
        "After The Love Is Gone - Earth, Wind & Fire.mp3": "1056",
        "Big Yellow Taxi (2007) - Joni Mitchell.mp3": "0668",
        "Chicago We Can Change the World - 2018 Remaster - Graham Nash.mp3": "0248",
        "Crystal Blue Persuasions (Single Version) - Tommy James & The Shondells.mp3": "0528",
        "Doo Doo Doo Doo (Heartbreaker) - 2005 Digital Remaster - The Rolling Stones.mp3": "0345",
        "El Condor Pasa (If I Could) - Daniel Alomía Robles, Simon & Garfunkel.mp3": "1261",
        "Get Up I Feel Like Being Like A Sex Machine, Pts. 1 & 2 - James Brown, The Original J.B.s.mp3": "0857",
        "I Got You - 1964 Smash Version - James Brown.mp3": "0914",
        "I've Been Loving You Too Long - To Stop Now - Otis Redding.mp3": "0502",
        "Indiana Wants Me (Rerecorded) - R. Dean Taylor.mp3": "0030",
        "Living It Down ( [en espanol]) - Freddy Fender.mp3": "0346",
        "Old Time Rock And Roll - Remastered 2011 - Bob Seger.mp3": "0249",
        "Reelin' And Rockin' - Chuck Berry.mp3": "1110",
        "Running Up That Hill (A Deal With God) - 2018 Remaster - Kate Bush.mp3": "0531",
        "Somebody's Watching Me - Single Version - Rockwell.mp3": "0334",
        "That's Old Fashioned (The Way Love Should Be) - The Everly Brothers.mp3": "1248",
        "The Anaheim, Azusa And Cucamonga Sewing Circle, Book Review And Timing Association - Jan & Dean.mp3": "0689",
        "The Best of My Love - 2013 Remaster - Eagles.mp3": "1061",
        "The Sound of Silence - Electric Version - Simon & Garfunkel.mp3": "1153",
        "Willie And The Hand Jive Get Ready - Live At Long Beach Arena, California 1974 - Eric Clapton.mp3": "0194",
        "You Can't Judge A Book By Its Cover - Bo Diddley.mp3": "0179",
        "You've Got Another Thing Coming - Judas Priest.mp3": "0861"
    }

    # Load CSV data
    print("Loading CSV data...")
    csv_lookup = load_csv_lookup(csv_path)

    songs_path = Path(songs_dir)
    training_path = Path(training_dir)

    print("\n" + "=" * 80)
    print("ADDING CLOSE-MATCHED SONGS")
    print("=" * 80)

    added_count = 0
    errors = []

    for mp3_filename, song_id in close_matches.items():
        mp3_path = songs_path / mp3_filename

        if not mp3_path.exists():
            errors.append(f"File not found: {mp3_filename}")
            continue

        if song_id not in csv_lookup:
            errors.append(f"song_id {song_id} not found in CSV")
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

        print(f"\n[{added_count}/23] Added song_id={song_id}")
        print(f"  MP3: {mp3_filename}")
        print(f"  CSV: {csv_song_name} - {csv_artist}")

    print("\n" + "=" * 80)
    print(f"COMPLETE: Added {added_count}/23 close-matched songs")
    print("=" * 80)

    if errors:
        print("\nErrors encountered:")
        for error in errors:
            print(f"  - {error}")


if __name__ == '__main__':
    csv_path = '../mcgill_billboard_spotify.csv'
    songs_dir = '../songs'
    training_dir = '../training_data'

    add_close_matches(songs_dir, training_dir, csv_path)
