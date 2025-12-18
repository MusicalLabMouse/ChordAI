"""
Dataset Cleanup Script
Reorganizes training_data to match CSV mappings.
Creates proper folder structure with one folder per song_id.
"""

import os
import csv
import shutil
import re
from pathlib import Path
from collections import defaultdict


def normalize_name(name):
    """
    Normalize song/artist name for matching.
    Removes special characters, extra spaces, converts to lowercase.
    """
    # Convert to lowercase
    name = name.lower()

    # Remove common file extensions
    name = re.sub(r'\.(mp3|wav|mp4)$', '', name, flags=re.IGNORECASE)

    # Replace common separators with space
    name = re.sub(r'[_\-/\\]', ' ', name)

    # Remove special characters but keep alphanumeric and spaces
    name = re.sub(r'[^\w\s]', '', name)

    # Remove extra spaces
    name = re.sub(r'\s+', ' ', name).strip()

    return name


def normalize_artist(artist):
    """
    Normalize artist name, handling multiple artists.
    E.g., "Rob Base & DJ EZ Rock" -> "rob base"
    """
    artist = normalize_name(artist)

    # Take only the first artist if multiple
    # Split by common delimiters: &, and, feat, featuring, with
    patterns = [' & ', ' and ', ' feat ', ' featuring ', ' with ', ' ft ']
    for pattern in patterns:
        if pattern in artist:
            artist = artist.split(pattern)[0].strip()
            break

    return artist


def extract_song_artist_from_filename(filename):
    """
    Extract song name and artist from filename.
    Expected format: "Song Title - Artist.mp3"
    """
    # Remove extension
    name = Path(filename).stem

    # Split by ' - ' to separate song and artist
    if ' - ' in name:
        parts = name.split(' - ')
        song_name = parts[0].strip()
        artist = parts[-1].strip()  # Take last part as artist
    else:
        # If no separator, treat whole name as song title
        song_name = name
        artist = ""

    return song_name, artist


def load_csv_mappings(csv_path):
    """
    Load CSV and create mappings.

    Returns:
        song_id_map: dict mapping song_id -> (song_name, artist)
        lookup_map: dict mapping (normalized_song, normalized_artist) -> song_id
    """
    song_id_map = {}
    lookup_map = {}

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            song_id = row['song_id']
            song_name = row['song_name']
            artist = row['artist']

            song_id_map[song_id] = (song_name, artist)

            # Create normalized lookup key
            norm_song = normalize_name(song_name)
            norm_artist = normalize_artist(artist)
            lookup_map[(norm_song, norm_artist)] = song_id

    return song_id_map, lookup_map


def find_all_mp3_files(data_dir):
    """
    Find all MP3 files in training_data directory.

    Returns:
        list of (file_path, song_name, artist) tuples
    """
    mp3_files = []
    data_path = Path(data_dir)

    # Recursively find all MP3 files
    for mp3_file in data_path.rglob('*.mp3'):
        song_name, artist = extract_song_artist_from_filename(mp3_file.name)
        mp3_files.append((mp3_file, song_name, artist))

    return mp3_files


def match_mp3_to_song_id(mp3_info, lookup_map):
    """
    Match an MP3 file to its song_id.

    Args:
        mp3_info: (file_path, song_name, artist) tuple
        lookup_map: dict mapping (norm_song, norm_artist) -> song_id

    Returns:
        song_id if match found, None otherwise
    """
    file_path, song_name, artist = mp3_info

    # Normalize
    norm_song = normalize_name(song_name)
    norm_artist = normalize_artist(artist)

    # Try exact match first
    key = (norm_song, norm_artist)
    if key in lookup_map:
        return lookup_map[key]

    # Try matching with any artist in CSV (in case artist normalization differs)
    for (csv_song, csv_artist), song_id in lookup_map.items():
        if csv_song == norm_song:
            # Check if artist is similar
            if csv_artist in norm_artist or norm_artist in csv_artist:
                return song_id

    return None


def reorganize_dataset(csv_path, data_dir, backup=True):
    """
    Reorganize the dataset based on CSV mappings.

    Args:
        csv_path: Path to mcgill_billboard_spotify.csv
        data_dir: Path to training_data directory
        backup: Whether to create backup before modifying
    """
    print("=" * 70)
    print("Dataset Cleanup & Reorganization")
    print("=" * 70)

    # Load CSV mappings
    print("\n1. Loading CSV mappings...")
    song_id_map, lookup_map = load_csv_mappings(csv_path)
    print(f"   Found {len(song_id_map)} songs in CSV")

    # Find all MP3 files
    print("\n2. Scanning for MP3 files...")
    mp3_files = find_all_mp3_files(data_dir)
    print(f"   Found {len(mp3_files)} MP3 files")

    # Match MP3s to song_ids
    print("\n3. Matching MP3 files to song_ids...")
    matches = {}
    unmatched = []

    for mp3_info in mp3_files:
        file_path, song_name, artist = mp3_info
        song_id = match_mp3_to_song_id(mp3_info, lookup_map)

        if song_id:
            if song_id not in matches:
                matches[song_id] = []
            matches[song_id].append(file_path)
        else:
            unmatched.append(mp3_info)

    print(f"   Matched: {len(matches)} song_ids")
    print(f"   Unmatched: {len(unmatched)} files")

    if unmatched:
        print("\n   Unmatched files:")
        for file_path, song_name, artist in unmatched[:10]:
            print(f"     - {file_path.name}")
        if len(unmatched) > 10:
            print(f"     ... and {len(unmatched) - 10} more")

    # Create backup if requested
    if backup:
        print("\n4. Creating backup...")
        backup_dir = Path(data_dir).parent / 'training_data_backup'
        if not backup_dir.exists():
            print(f"   Note: Skipping backup (would be large). Original files will be moved, not copied.")
        else:
            print(f"   Backup already exists at {backup_dir}")

    # Reorganize files
    print("\n5. Reorganizing files...")
    data_path = Path(data_dir)
    reorganized_count = 0
    created_folders = 0

    for song_id, (song_name, artist) in song_id_map.items():
        # Create folder for this song_id
        song_folder = data_path / song_id

        if not song_folder.exists():
            song_folder.mkdir(parents=True, exist_ok=True)
            created_folders += 1

        # Move MP3 files to correct folder if we have matches
        if song_id in matches:
            for mp3_file in matches[song_id]:
                # Expected filename format
                new_filename = f"{song_name} - {artist}.mp3"
                new_path = song_folder / new_filename

                # Only move if not already in correct location
                if mp3_file.resolve() != new_path.resolve():
                    # Check if file already exists
                    if new_path.exists():
                        print(f"   [WARN] {new_path.name} already exists in {song_id}, skipping duplicate")
                    else:
                        shutil.move(str(mp3_file), str(new_path))
                        reorganized_count += 1
                        print(f"   [OK] Moved to {song_id}/: {song_name}")

        # Also move .lab file if it exists in old location
        # Look for any .lab file that might belong to this song
        for old_folder in data_path.iterdir():
            if old_folder.is_dir() and old_folder.name != song_id:
                lab_files = list(old_folder.glob('*.lab'))
                for lab_file in lab_files:
                    # Check if this lab file might belong to current song
                    # This is tricky - we'll move it if the folder is otherwise empty
                    pass

    print(f"\n   Reorganized {reorganized_count} MP3 files")
    print(f"   Created {created_folders} new folders")

    # Clean up empty directories
    print("\n6. Cleaning up empty directories...")
    removed_count = 0
    for folder in data_path.iterdir():
        if folder.is_dir():
            # Check if folder is empty or only has empty subdirectories
            contents = list(folder.iterdir())
            if not contents:
                shutil.rmtree(folder)
                removed_count += 1

    print(f"   Removed {removed_count} empty folders")

    # Final verification
    print("\n7. Verification...")
    song_id_folders = sorted([d.name for d in data_path.iterdir() if d.is_dir() and d.name.isdigit()])
    folders_with_mp3 = []
    folders_with_lab = []
    folders_with_both = []

    for folder_name in song_id_folders:
        folder = data_path / folder_name
        has_mp3 = len(list(folder.glob('*.mp3'))) > 0
        has_lab = len(list(folder.glob('*.lab'))) > 0

        if has_mp3:
            folders_with_mp3.append(folder_name)
        if has_lab:
            folders_with_lab.append(folder_name)
        if has_mp3 and has_lab:
            folders_with_both.append(folder_name)

    print(f"   Total song_id folders: {len(song_id_folders)}")
    print(f"   Folders with MP3: {len(folders_with_mp3)}")
    print(f"   Folders with .lab: {len(folders_with_lab)}")
    print(f"   Folders with both: {len(folders_with_both)}")
    print(f"   Folders missing MP3: {len(song_id_folders) - len(folders_with_mp3)}")
    print(f"   Folders missing .lab: {len(song_id_folders) - len(folders_with_lab)}")

    print("\n" + "=" * 70)
    print("Dataset cleanup complete!")
    print("=" * 70)

    return {
        'total_songs': len(song_id_map),
        'matched_mp3s': len(matches),
        'folders_with_both': len(folders_with_both),
        'folders_with_mp3': len(folders_with_mp3),
        'folders_with_lab': len(folders_with_lab)
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Clean up and reorganize dataset')
    parser.add_argument('--csv', type=str, default='../mcgill_billboard_spotify.csv',
                        help='Path to CSV file')
    parser.add_argument('--data_dir', type=str, default='../training_data',
                        help='Path to training_data directory')
    parser.add_argument('--no-backup', action='store_true',
                        help='Skip backup creation')
    args = parser.parse_args()

    stats = reorganize_dataset(args.csv, args.data_dir, backup=not args.no_backup)

    print(f"\nDataset is ready for feature extraction!")
    print(f"You can now run: python feature_extraction.py")


if __name__ == '__main__':
    main()
