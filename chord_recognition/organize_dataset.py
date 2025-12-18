"""
Organize dataset by matching MP3 files to song_ids from CSV.
Creates proper folder structure in training_data.
"""

import os
import csv
import shutil
import re
from pathlib import Path
from collections import defaultdict


def normalize_text(text):
    """
    Normalize text for matching.
    Removes special characters, extra spaces, converts to lowercase.
    """
    # Convert to lowercase
    text = text.lower()

    # Remove file extensions
    text = re.sub(r'\.(mp3|wav|mp4)$', '', text, flags=re.IGNORECASE)

    # Replace common separators with space
    text = re.sub(r'[_\-/\\]', ' ', text)

    # Remove special characters but keep alphanumeric and spaces
    text = re.sub(r'[^\w\s]', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def normalize_artist(artist):
    """
    Normalize artist name - take only the first artist.
    Examples:
        "Rob Base & DJ EZ Rock" -> "rob base"
        "Daryl Hall & John Oates" -> "daryl hall"
        "Marvin Gaye, Kim Weston" -> "marvin gaye"
    """
    artist = normalize_text(artist)

    # Split by common delimiters and take first artist
    # Order matters - check most specific patterns first
    delimiters = [
        ' feat ',
        ' featuring ',
        ' ft ',
        ' with ',
        ' and ',
        ' & ',
        ', ',
    ]

    for delimiter in delimiters:
        if delimiter in artist:
            artist = artist.split(delimiter)[0].strip()
            break

    return artist


def extract_song_artist_from_filename(filename):
    """
    Extract song name and artist from filename.
    Expected format: "Song Title - Artist.mp3"
    """
    # Remove extension
    name = Path(filename).stem

    # Remove common suffixes like "Live", "Remastered", etc. before splitting
    # We'll keep them for now and strip later if needed

    # Split by ' - ' to separate song and artist
    if ' - ' in name:
        parts = name.split(' - ')

        # First part is always the song
        song_name = parts[0].strip()

        # Everything after first ' - ' is artist (may contain more ' - ')
        artist = ' - '.join(parts[1:]).strip()

        # Remove version info from artist (like "Remastered", "Live", etc.)
        # Common patterns at the end of artist names
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
        # If no separator, treat whole name as song title
        song_name = name
        artist = ""

    return song_name, artist


def load_csv_data(csv_path):
    """
    Load CSV and create lookup structures.

    Returns:
        song_id_map: {song_id: (song_name, artist)}
        lookup_by_song: {normalized_song: [(song_id, song_name, artist, normalized_artist)]}
    """
    song_id_map = {}
    lookup_by_song = defaultdict(list)

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            song_id = row['song_id']
            song_name = row['song_name']
            artist = row['artist']

            song_id_map[song_id] = (song_name, artist)

            # Create lookup by normalized song name
            norm_song = normalize_text(song_name)
            norm_artist = normalize_artist(artist)

            lookup_by_song[norm_song].append((song_id, song_name, artist, norm_artist))

    return song_id_map, lookup_by_song


def find_best_match(mp3_song, mp3_artist, candidates):
    """
    Find the best matching song_id from candidates.

    Args:
        mp3_song: normalized song name from MP3 filename
        mp3_artist: normalized artist from MP3 filename
        candidates: list of (song_id, song_name, artist, norm_artist) tuples

    Returns:
        best_song_id or None
    """
    norm_mp3_artist = normalize_artist(mp3_artist)

    # If only one candidate, return it
    if len(candidates) == 1:
        return candidates[0][0]

    # Try to match by artist
    for song_id, song_name, artist, norm_artist in candidates:
        # Exact match on normalized artist
        if norm_mp3_artist == norm_artist:
            return song_id

        # Partial match - artist contains or is contained
        if norm_mp3_artist and norm_artist:
            if norm_mp3_artist in norm_artist or norm_artist in norm_mp3_artist:
                return song_id

    # If no artist match, return first candidate (alphabetically by song_id)
    candidates_sorted = sorted(candidates, key=lambda x: x[0])
    return candidates_sorted[0][0]


def sanitize_filename(filename):
    """
    Remove illegal characters from Windows filenames.
    Illegal characters: < > : " / \ | ? *
    """
    illegal_chars = '<>:"/\\|?*'
    for char in illegal_chars:
        filename = filename.replace(char, '')
    return filename


def organize_dataset(csv_path, songs_dir, training_dir):
    """
    Organize songs from songs_dir into training_dir based on CSV mappings.
    """
    print("=" * 80)
    print("Dataset Organization Script")
    print("=" * 80)

    # Load CSV data
    print("\n[1/5] Loading CSV data...")
    song_id_map, lookup_by_song = load_csv_data(csv_path)
    print(f"    Loaded {len(song_id_map)} songs from CSV")

    # Find all MP3 files
    print(f"\n[2/5] Scanning {songs_dir} for MP3 files...")
    songs_path = Path(songs_dir)
    mp3_files = sorted(list(songs_path.glob('*.mp3')))
    print(f"    Found {len(mp3_files)} MP3 files")

    # Match MP3s to song_ids
    print("\n[3/5] Matching MP3s to song_ids...")
    matches = {}  # {song_id: (mp3_path, mp3_filename, song_name, artist)}
    unmatched = []
    duplicate_songs = defaultdict(list)  # Track songs with same name

    for mp3_path in mp3_files:
        mp3_filename = mp3_path.name
        song_name, artist = extract_song_artist_from_filename(mp3_filename)

        norm_song = normalize_text(song_name)

        if norm_song in lookup_by_song:
            candidates = lookup_by_song[norm_song]

            # Track duplicate song names
            if len(candidates) > 1:
                duplicate_songs[norm_song].append({
                    'mp3_file': mp3_filename,
                    'mp3_artist': artist,
                    'candidates': candidates
                })

            # Find best match
            song_id = find_best_match(norm_song, artist, candidates)

            if song_id:
                csv_song_name, csv_artist = song_id_map[song_id]
                matches[song_id] = (mp3_path, mp3_filename, csv_song_name, csv_artist)
        else:
            unmatched.append((mp3_path, mp3_filename, song_name, artist))

    print(f"    Matched: {len(matches)} MP3s")
    print(f"    Unmatched: {len(unmatched)} MP3s")
    print(f"    Songs with duplicates: {len(duplicate_songs)}")

    # Create training_data folders and copy files
    print(f"\n[4/5] Creating folders and copying files...")
    training_path = Path(training_dir)
    training_path.mkdir(parents=True, exist_ok=True)

    copied_count = 0
    for song_id, (mp3_path, mp3_filename, csv_song_name, csv_artist) in sorted(matches.items()):
        # Create folder
        song_folder = training_path / song_id
        song_folder.mkdir(parents=True, exist_ok=True)

        # Create standardized filename (sanitized for Windows)
        new_filename = f"{csv_song_name} - {csv_artist}.mp3"
        new_filename = sanitize_filename(new_filename)
        dest_path = song_folder / new_filename

        # Copy file
        shutil.copy2(mp3_path, dest_path)
        copied_count += 1

        if copied_count % 50 == 0:
            print(f"    Copied {copied_count}/{len(matches)} files...")

    print(f"    Copied all {copied_count} files")

    # Generate report
    print("\n[5/5] Generating report...")

    report = []
    report.append("=" * 80)
    report.append("DATASET ORGANIZATION REPORT")
    report.append("=" * 80)
    report.append(f"\nTotal MP3 files found: {len(mp3_files)}")
    report.append(f"Successfully matched: {len(matches)}")
    report.append(f"Unmatched: {len(unmatched)}")
    report.append(f"\nSongs in CSV: {len(song_id_map)}")
    report.append(f"Folders created: {len(matches)}")
    report.append(f"Missing songs: {len(song_id_map) - len(matches)}")

    # Report on duplicate song names
    if duplicate_songs:
        report.append("\n" + "=" * 80)
        report.append("DUPLICATE SONG NAMES (Same song title, different artists)")
        report.append("=" * 80)

        for norm_song, duplicates in sorted(duplicate_songs.items()):
            report.append(f"\n{'-' * 80}")
            report.append(f"Song: {duplicates[0]['candidates'][0][1]}")  # Original song name
            report.append(f"{'-' * 80}")

            for dup in duplicates:
                mp3_file = dup['mp3_file']
                mp3_artist = dup['mp3_artist']
                candidates = dup['candidates']

                # Find which one was matched
                matched_id = find_best_match(norm_song, mp3_artist, candidates)
                matched_csv = [c for c in candidates if c[0] == matched_id][0]

                report.append(f"\nMP3 File: {mp3_file}")
                report.append(f"  Extracted Artist: {mp3_artist}")
                report.append(f"  MATCHED TO => song_id={matched_id}: {matched_csv[2]}")
                report.append(f"\n  All candidates for this song:")
                for song_id, song_name, artist, norm_artist in sorted(candidates, key=lambda x: x[0]):
                    marker = "[X] SELECTED" if song_id == matched_id else "[ ]"
                    report.append(f"    {marker} song_id={song_id}: {artist}")

    # Report unmatched files
    if unmatched:
        report.append("\n" + "=" * 80)
        report.append("UNMATCHED FILES (Not found in CSV)")
        report.append("=" * 80)
        for mp3_path, mp3_filename, song_name, artist in unmatched:
            report.append(f"\n  {mp3_filename}")
            report.append(f"    Song: {song_name}")
            report.append(f"    Artist: {artist}")

    # List missing songs
    matched_ids = set(matches.keys())
    missing_ids = sorted([sid for sid in song_id_map.keys() if sid not in matched_ids])

    if missing_ids:
        report.append("\n" + "=" * 80)
        report.append(f"MISSING SONGS (In CSV but no MP3 found) - {len(missing_ids)} songs")
        report.append("=" * 80)
        for song_id in missing_ids[:20]:  # Show first 20
            song_name, artist = song_id_map[song_id]
            report.append(f"  song_id={song_id}: {song_name} - {artist}")
        if len(missing_ids) > 20:
            report.append(f"  ... and {len(missing_ids) - 20} more")

    report.append("\n" + "=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)

    # Print and save report
    report_text = "\n".join(report)
    print(report_text)

    # Save to file
    report_path = Path('dataset_organization_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\nReport saved to: {report_path.absolute()}")

    return {
        'matched': len(matches),
        'unmatched': len(unmatched),
        'duplicates': len(duplicate_songs),
        'missing': len(missing_ids)
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Organize dataset from CSV')
    parser.add_argument('--csv', type=str, default='../mcgill_billboard_spotify.csv',
                        help='Path to CSV file')
    parser.add_argument('--songs_dir', type=str, default='../McGill Billboard - Training Songs',
                        help='Path to folder with MP3 files')
    parser.add_argument('--training_dir', type=str, default='../training_data',
                        help='Path to training_data directory')
    args = parser.parse_args()

    stats = organize_dataset(args.csv, args.songs_dir, args.training_dir)

    print(f"\n[OK] Organization complete!")
    print(f"  Matched: {stats['matched']}")
    print(f"  Unmatched: {stats['unmatched']}")
    print(f"  Duplicates: {stats['duplicates']}")
    print(f"  Missing: {stats['missing']}")


if __name__ == '__main__':
    main()
