"""
Check for folders with multiple MP3s (duplicate song_ids).
"""

from pathlib import Path


def check_duplicates(training_dir):
    """Check for folders with multiple MP3 files."""

    training_path = Path(training_dir)

    duplicates = []
    total_mp3s = 0

    for folder in sorted(training_path.iterdir()):
        if not folder.is_dir():
            continue

        mp3_files = list(folder.glob('*.mp3'))
        total_mp3s += len(mp3_files)

        if len(mp3_files) > 1:
            duplicates.append({
                'song_id': folder.name,
                'count': len(mp3_files),
                'files': [f.name for f in mp3_files]
            })

    print("=" * 80)
    print("CHECKING FOR DUPLICATE SONG_IDs")
    print("=" * 80)

    print(f"\nTotal folders: {len(list(training_path.iterdir()))}")
    print(f"Total MP3s in all folders: {total_mp3s}")

    if duplicates:
        print(f"\nFolders with multiple MP3s: {len(duplicates)}")
        print("=" * 80)

        for dup in duplicates:
            print(f"\nsong_id={dup['song_id']} has {dup['count']} MP3s:")
            for filename in dup['files']:
                print(f"  - {filename}")
    else:
        print("\nNo duplicate song_ids found!")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    training_dir = '../training_data'
    check_duplicates(training_dir)
