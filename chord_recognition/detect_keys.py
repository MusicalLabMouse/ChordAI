"""
Standalone Key Detection Script
Uses Krumhansl-Schmuckler algorithm to detect musical keys from audio files.

Based on: https://github.com/Corentin-Lcs/music-key-finder

Usage:
    python detect_keys.py --input ../training_datasets --output keys.csv --limit 100
    python detect_keys.py --input ../training_datasets --output keys.csv  # All songs
    python detect_keys.py --file song.mp3  # Single file mode
    python detect_keys.py --input ../training_datasets --update-labs  # Add keys to .lab files

Output format (CSV):
    filepath,key,confidence
    /path/to/song1.mp3,G:maj,0.8234
    /path/to/song2.mp3,A:min,0.7891

Lab file format (when using --update-labs):
    # key: G:maj
    # confidence: 0.8234
    0.0  0.5  N
    0.5  2.3  C:maj
    ...
"""

import argparse
import warnings
import numpy as np
from pathlib import Path
import csv

# Suppress librosa warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

try:
    import librosa
except ImportError:
    print("Error: librosa is required. Install with: pip install librosa")
    exit(1)

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Note: tqdm not installed. Progress bars disabled. Install with: pip install tqdm")


# =============================================================================
# Krumhansl-Schmuckler Key Profiles
# =============================================================================
# These are empirically-derived profiles representing the "ideal" distribution
# of pitch classes in major and minor keys (Krumhansl & Kessler, 1982)

MAJOR_PROFILE = np.array([
    6.35,  # C  (tonic)
    2.23,  # C#
    3.48,  # D  (supertonic)
    2.33,  # D#
    4.38,  # E  (mediant)
    4.09,  # F  (subdominant)
    2.52,  # F#
    5.19,  # G  (dominant)
    2.39,  # G#
    3.66,  # A  (submediant)
    2.29,  # A#
    2.88,  # B  (leading tone)
])

MINOR_PROFILE = np.array([
    6.33,  # C  (tonic)
    2.68,  # C#
    3.52,  # D  (supertonic)
    5.38,  # D#/Eb (minor third)
    2.60,  # E
    3.53,  # F  (subdominant)
    2.54,  # F#
    4.75,  # G  (dominant)
    3.98,  # G#/Ab (minor sixth)
    2.69,  # A
    3.34,  # A#/Bb (minor seventh)
    3.17,  # B
])

# Key names (using sharps for consistency)
KEY_NAMES_SHARP = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Alternative names with flats for common flat keys
KEY_NAMES_FLAT = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

# Use flats for these keys (more common in music notation)
FLAT_KEY_INDICES = {1, 3, 6, 8, 10}  # Db, Eb, Gb, Ab, Bb


def get_key_name(pitch_class_idx, use_flats=True):
    """Get the key name for a pitch class index."""
    if use_flats and pitch_class_idx in FLAT_KEY_INDICES:
        return KEY_NAMES_FLAT[pitch_class_idx]
    return KEY_NAMES_SHARP[pitch_class_idx]


def detect_key(audio_path, sr=None, segment_duration=10.0):
    """
    Detect the musical key of an audio file using the Krumhansl-Schmuckler algorithm.

    Based on: https://github.com/Corentin-Lcs/music-key-finder

    The algorithm works by:
    1. Extracting a chromagram (pitch class distribution) from the audio
    2. Correlating it with ideal major/minor key profiles
    3. Returning the key with the highest correlation

    Args:
        audio_path: Path to the audio file (MP3, WAV, etc.)
        sr: Sample rate for analysis (None = use original sample rate)
        segment_duration: Duration in seconds for each analysis segment

    Returns:
        key: String like "G:maj" or "A:min"
        confidence: Pearson correlation coefficient (-1.0 to 1.0, higher = more confident)
        all_correlations: Dict with correlation values for all 24 keys (for debugging)
    """
    # Load audio (sr=None keeps original sample rate, matching the reference implementation)
    y, sr = librosa.load(audio_path, sr=sr, mono=True)

    # Trim silence from beginning and end
    y, _ = librosa.effects.trim(y)

    # Check if audio is too short
    if len(y) < sr * 2:  # Less than 2 seconds
        raise ValueError(f"Audio too short: {len(y)/sr:.1f} seconds")

    # Split into 10-second segments for more detailed analysis
    segment_length = int(sr * segment_duration)
    num_segments = max(1, len(y) // segment_length)
    chroma_mean_total = np.zeros(12)

    for i in range(num_segments):
        start = i * segment_length
        end = (i + 1) * segment_length
        segment = y[start:end]

        # Chromogram calculation (matching reference implementation)
        chroma = librosa.feature.chroma_cqt(y=segment, sr=sr)
        chroma_mean_total += np.mean(chroma, axis=1)

    # Average chromagram values across segments
    chroma_mean_total /= num_segments

    # Normalize using L2 norm (matching reference implementation)
    chroma_mean_total /= np.linalg.norm(chroma_mean_total)

    # Calculate correlations with major and minor key profiles
    # Note: np.roll(profile, i) shifts the profile, comparing against each possible key
    major_correlations = [np.corrcoef(np.roll(MAJOR_PROFILE, i), chroma_mean_total)[0, 1] for i in range(12)]
    minor_correlations = [np.corrcoef(np.roll(MINOR_PROFILE, i), chroma_mean_total)[0, 1] for i in range(12)]

    # Build correlations dict for all 24 keys
    correlations = {}
    for i in range(12):
        key_name = get_key_name(i)
        correlations[f"{key_name}:maj"] = major_correlations[i]
        correlations[f"{key_name}:min"] = minor_correlations[i]

    # Determine the best key (matching reference implementation)
    major_max = max(major_correlations)
    minor_max = max(minor_correlations)
    major_key_idx = np.argmax(major_correlations)
    minor_key_idx = np.argmax(minor_correlations)

    if major_max > minor_max:
        best_key = f"{get_key_name(major_key_idx)}:maj"
        best_corr = major_max
    else:
        best_key = f"{get_key_name(minor_key_idx)}:min"
        best_corr = minor_max

    return best_key, best_corr, correlations


def detect_key_simple(audio_path, sr=22050):
    """
    Simplified key detection returning just key and confidence.

    Args:
        audio_path: Path to audio file
        sr: Sample rate

    Returns:
        key: String like "G:maj" or "A:min"
        confidence: Float from 0 to 1
    """
    key, confidence, _ = detect_key(audio_path, sr)
    return key, confidence


def print_detailed_results(key, confidence, correlations):
    """Print detailed analysis results."""
    print(f"\nDetected Key: {key}")
    print(f"Confidence: {confidence:.4f}")
    print("\nAll key correlations:")
    print("-" * 40)

    # Sort by correlation value
    sorted_keys = sorted(correlations.items(), key=lambda x: x[1], reverse=True)

    for i, (k, v) in enumerate(sorted_keys[:10]):
        marker = " <-- BEST" if i == 0 else ""
        print(f"  {k:8s}: {v:+.4f}{marker}")
    print("  ...")


def find_audio_files(input_path, extensions=('.mp3', '.wav', '.flac', '.ogg', '.m4a')):
    """Find all audio files in directory (recursively)."""
    input_path = Path(input_path)

    if input_path.is_file():
        return [input_path]

    audio_files = []
    for ext in extensions:
        audio_files.extend(input_path.rglob(f'*{ext}'))

    return sorted(audio_files)


def find_lab_files(input_path):
    """Find all .lab files in directory (recursively)."""
    input_path = Path(input_path)
    return sorted(input_path.rglob('*.lab'))


def find_audio_for_lab(lab_path, extensions=('.mp3', '.wav', '.flac', '.ogg', '.m4a')):
    """Find the audio file in the same directory as a .lab file."""
    lab_dir = lab_path.parent

    for ext in extensions:
        audio_files = list(lab_dir.glob(f'*{ext}'))
        if audio_files:
            return audio_files[0]  # Return first match

    return None


def update_lab_with_key(lab_path, key, confidence):
    """
    Add key and confidence to the top of a .lab file.

    If the file already has key/confidence comments, they are replaced.
    """
    # Read existing content
    with open(lab_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Remove any existing key/confidence comments at the top
    while lines and lines[0].startswith('# key:'):
        lines.pop(0)
    while lines and lines[0].startswith('# confidence:'):
        lines.pop(0)

    # Prepend new key/confidence
    header = f"# key: {key}\n# confidence: {confidence:.4f}\n"

    # Write back
    with open(lab_path, 'w', encoding='utf-8') as f:
        f.write(header)
        f.writelines(lines)


def process_labs(input_path, limit=None, sr=None):
    """
    Find all .lab files, detect keys from corresponding audio, and update the .lab files.

    Args:
        input_path: Directory to search for .lab files
        limit: Maximum number of files to process (None = all)
        sr: Sample rate for analysis (None = use original)

    Returns:
        results: List of dicts with lab_path, audio_path, key, confidence
        errors: List of (path, error_message) tuples
    """
    import sys

    lab_files = find_lab_files(input_path)

    if not lab_files:
        print("No .lab files found!", file=sys.stderr)
        return [], []

    if limit:
        lab_files = lab_files[:limit]

    results = []
    errors = []

    # Use tqdm if available
    if TQDM_AVAILABLE:
        iterator = tqdm(lab_files, desc='Processing .lab files')
    else:
        iterator = lab_files
        print(f"Processing {len(lab_files)} .lab files...")

    for lab_path in iterator:
        # Find corresponding audio file
        audio_path = find_audio_for_lab(lab_path)

        if audio_path is None:
            error_msg = "No audio file found in directory"
            errors.append((lab_path, error_msg))
            print(f"{lab_path}\tERROR\t{error_msg}")
            continue

        try:
            # Detect key
            key, confidence, _ = detect_key(audio_path, sr=sr)

            # Update .lab file
            update_lab_with_key(lab_path, key, confidence)

            result = {
                'lab_path': str(lab_path),
                'audio_path': str(audio_path),
                'key': key,
                'confidence': confidence
            }
            results.append(result)

            # Print to console
            print(f"{lab_path}\t{key}\t{confidence:.4f}")

        except Exception as e:
            error_msg = str(e)[:100]
            errors.append((lab_path, error_msg))
            print(f"{lab_path}\tERROR\t{error_msg}")

    return results, errors


def process_batch(audio_files, output_path=None, sr=22050):
    """Process multiple audio files and output results to console (and optionally CSV)."""
    results = []
    errors = []

    # Print header (tab-separated for clickable paths)
    print("filepath\tkey\tconfidence")

    for audio_path in audio_files:
        try:
            key, confidence = detect_key_simple(audio_path, sr)
            result = {
                'filepath': str(audio_path.resolve()),  # Full absolute path
                'key': key,
                'confidence': f'{confidence:.4f}'
            }
            results.append(result)

            # Print to console (tab-separated so filepath is clickable)
            print(f"{result['filepath']}\t{result['key']}\t{result['confidence']}")

        except Exception as e:
            error_msg = str(e)[:100]  # Truncate long errors
            result = {
                'filepath': str(audio_path.resolve()),
                'key': 'ERROR',
                'confidence': error_msg
            }
            results.append(result)
            errors.append((audio_path, error_msg))

            # Print error to console
            print(f"{result['filepath']}\tERROR\t{error_msg}")

    # Optionally write to CSV file
    if output_path:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['filepath', 'key', 'confidence'])
            writer.writeheader()
            writer.writerows(results)

    return results, errors


def main():
    parser = argparse.ArgumentParser(
        description='Detect musical keys from audio files using Krumhansl-Schmuckler algorithm',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single file
  python detect_keys.py --file song.mp3

  # Analyze a single file with detailed output
  python detect_keys.py --file song.mp3 --detailed

  # Batch process a directory (first 10 songs) - outputs to console
  python detect_keys.py --input ./training_datasets --limit 10

  # Process all songs, output to console
  python detect_keys.py --input ./training_datasets

  # Process all songs, also save to CSV file
  python detect_keys.py --input ./training_datasets --output keys.csv

  # Add keys to .lab files (first 5)
  python detect_keys.py --input ./training_datasets --update-labs --limit 5

  # Add keys to all .lab files
  python detect_keys.py --input ./training_datasets --update-labs
        """
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--file', type=str,
                            help='Single audio file to analyze')
    input_group.add_argument('--input', type=str,
                            help='Input directory containing audio files')

    # Output options
    parser.add_argument('--output', type=str, default=None,
                       help='Optional: also save results to CSV file')
    parser.add_argument('--limit', type=int, default=None,
                       help='Maximum number of songs to process')
    parser.add_argument('--update-labs', action='store_true',
                       help='Find .lab files and add detected key to each one')

    # Analysis options
    parser.add_argument('--sr', type=int, default=None,
                       help='Sample rate for analysis (default: original sample rate)')
    parser.add_argument('--detailed', action='store_true',
                       help='Show detailed correlation values (single file mode only)')

    args = parser.parse_args()

    if args.file:
        # Single file mode
        audio_path = Path(args.file)
        if not audio_path.exists():
            print(f"Error: File not found: {audio_path}", file=__import__('sys').stderr)
            return 1

        try:
            key, confidence, correlations = detect_key(audio_path, sr=args.sr)

            if args.detailed:
                print_detailed_results(key, confidence, correlations)
            else:
                # Output: full_path  key  confidence (tab-separated for clickable paths)
                print(f"{audio_path.resolve()}\t{key}\t{confidence:.4f}")

        except Exception as e:
            print(f"Error: {e}", file=__import__('sys').stderr)
            return 1

    else:
        # Batch mode
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Directory not found: {input_path}", file=__import__('sys').stderr)
            return 1

        if args.update_labs:
            # Update .lab files mode
            results, errors = process_labs(input_path, limit=args.limit, sr=args.sr)

            # Print summary
            import sys
            print(f"\nProcessed {len(results)} .lab files", file=sys.stderr)
            if errors:
                print(f"Errors: {len(errors)}", file=sys.stderr)

        else:
            # Standard audio file mode
            audio_files = find_audio_files(input_path)

            if not audio_files:
                print("No audio files found!", file=__import__('sys').stderr)
                return 1

            # Apply limit if specified
            if args.limit:
                audio_files = audio_files[:args.limit]

            # Process files - outputs to console
            sr = args.sr if args.sr else 22050  # Default to 22050 for batch mode
            results, errors = process_batch(audio_files, args.output, sr=sr)

            # Print summary to stderr (so it doesn't interfere with CSV output on stdout)
            if args.output:
                import sys
                print(f"\nAlso saved to: {args.output}", file=sys.stderr)

    return 0


if __name__ == '__main__':
    exit(main())
