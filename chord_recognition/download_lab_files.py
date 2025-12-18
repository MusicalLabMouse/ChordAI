"""
Download .lab files from McGill Billboard GitHub repository.
Downloads only for song_ids that exist in training_data.
"""

import urllib.request
import time
from pathlib import Path


def download_lab_files(training_dir):
    """Download .lab files for all song_ids in training_data."""

    training_path = Path(training_dir)

    # Get all song_ids from training_data folders
    song_ids = sorted([f.name for f in training_path.iterdir() if f.is_dir()])

    print("=" * 80)
    print("DOWNLOADING LAB FILES FROM GITHUB")
    print("=" * 80)
    print(f"\nFound {len(song_ids)} song_ids in training_data")
    print("Downloading corresponding .lab files from GitHub...")
    print()

    base_url = "https://raw.githubusercontent.com/boomerr1/The-McGill-Billboard-Project/master/billboard-2.0.1-lab"

    downloaded = 0
    skipped = 0
    errors = []

    for i, song_id in enumerate(song_ids, 1):
        song_folder = training_path / song_id
        lab_file = song_folder / "full.lab"

        # Skip if already exists
        if lab_file.exists():
            skipped += 1
            if i % 50 == 0:
                print(f"[{i}/{len(song_ids)}] Checked {i} songs...")
            continue

        # Download from GitHub
        url = f"{base_url}/{song_id}/full.lab"

        try:
            urllib.request.urlretrieve(url, lab_file)
            downloaded += 1

            if downloaded % 10 == 0:
                print(f"[{i}/{len(song_ids)}] Downloaded {downloaded} files...")

            # Small delay to avoid hammering the server
            time.sleep(0.1)

        except Exception as e:
            errors.append({
                'song_id': song_id,
                'url': url,
                'error': str(e)
            })

    print("\n" + "=" * 80)
    print("DOWNLOAD COMPLETE")
    print("=" * 80)
    print(f"\nTotal song_ids: {len(song_ids)}")
    print(f"Downloaded: {downloaded}")
    print(f"Already existed: {skipped}")
    print(f"Errors: {len(errors)}")

    if errors:
        print("\n" + "=" * 80)
        print("ERRORS")
        print("=" * 80)
        for err in errors[:10]:  # Show first 10 errors
            print(f"\nsong_id={err['song_id']}")
            print(f"  URL: {err['url']}")
            print(f"  Error: {err['error']}")

        if len(errors) > 10:
            print(f"\n... and {len(errors) - 10} more errors")

    print("\n" + "=" * 80)

    # Verify how many folders now have .lab files
    folders_with_lab = 0
    for song_id in song_ids:
        lab_file = training_path / song_id / "full.lab"
        if lab_file.exists():
            folders_with_lab += 1

    print(f"\nFolders with .lab files: {folders_with_lab}/{len(song_ids)}")
    print(f"Missing .lab files: {len(song_ids) - folders_with_lab}")

    if folders_with_lab == len(song_ids):
        print("\n[SUCCESS] All folders now have .lab files!")
    else:
        print(f"\n[WARNING] {len(song_ids) - folders_with_lab} folders are missing .lab files")

    print("=" * 80)


if __name__ == '__main__':
    training_dir = '../training_data'
    download_lab_files(training_dir)
