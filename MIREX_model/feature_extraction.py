"""
Feature Extraction for MIREX Chord Estimation Model
Extracts CQT features with 36 bins/octave and converts chords to scale degrees.
"""

import os
import json
import argparse
import random
import numpy as np
import librosa
import warnings
import re
from tqdm import tqdm
from pathlib import Path
from contextlib import contextmanager

import config

warnings.filterwarnings('ignore', category=UserWarning, module='librosa')
warnings.filterwarnings('ignore', category=FutureWarning)


@contextmanager
def suppress_stderr():
    """Suppress stderr output."""
    null_fd = os.open(os.devnull, os.O_RDWR)
    save_stderr = os.dup(2)
    os.dup2(null_fd, 2)
    try:
        yield
    finally:
        os.dup2(save_stderr, 2)
        os.close(null_fd)


# Note name to semitone mapping
NOTE_TO_SEMITONE = {
    'C': 0, 'Db': 1, 'D': 2, 'Eb': 3, 'E': 4, 'F': 5,
    'F#': 6, 'G': 7, 'Ab': 8, 'A': 9, 'Bb': 10, 'B': 11
}

SEMITONE_TO_NOTE = {v: k for k, v in NOTE_TO_SEMITONE.items()}


def normalize_note_name(note):
    """Normalize note name to standard form (prefer flats except F#)."""
    if note in config.ENHARMONIC_MAP:
        return config.ENHARMONIC_MAP[note]
    return note


def note_to_semitone(note):
    """Convert note name to semitone (0-11)."""
    note = normalize_note_name(note)
    return NOTE_TO_SEMITONE.get(note, 0)


def get_scale_degree(root_semitone, key_semitone):
    """
    Get scale degree (0-11) from root and key semitones.
    0 = I, 1 = bII, 2 = II, etc.
    """
    return (root_semitone - key_semitone) % 12


def parse_salami_key(salami_path):
    """
    Parse the key (tonic) from salami_chords.txt file.

    Returns:
        key: Note name (e.g., 'C', 'F#') or None if not found
    """
    try:
        with open(salami_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('# tonic:'):
                    key = line.split(':')[1].strip()
                    return normalize_note_name(key)
    except Exception:
        pass
    return None


def normalize_chord(chord_label):
    """
    Normalize chord label to standard form.

    Returns:
        dict with 'root', 'quality', 'bass' or None for N/silence
    """
    if chord_label in ['N', 'X', 'silence', '*']:
        return None

    # Handle slash chords (bass note)
    bass = None
    if '/' in chord_label and not chord_label.startswith('/'):
        parts = chord_label.split('/')
        chord_label = parts[0]
        bass = normalize_note_name(parts[1]) if len(parts) > 1 else None

    # Split root and quality
    if ':' not in chord_label:
        root = chord_label
        quality = 'maj'
    else:
        parts = chord_label.split(':')
        root = parts[0]
        quality = parts[1] if len(parts) > 1 else 'maj'

    root = normalize_note_name(root)

    # Normalize quality to major/minor
    quality_lower = quality.lower()
    if 'min' in quality_lower or 'dim' in quality_lower:
        quality = 'min'
    else:
        quality = 'maj'

    if bass is None:
        bass = root

    return {'root': root, 'quality': quality, 'bass': bass}


def chord_to_scale_degree_representation(chord_info, key):
    """
    Convert chord to scale degree representation.

    Args:
        chord_info: dict from normalize_chord()
        key: Key note name

    Returns:
        dict with 'key_idx', 'degree_idx', 'bass_idx', 'pitch_abs', 'pitch_root', 'pitch_bass'
    """
    if chord_info is None or key is None:
        # No chord or no key
        return {
            'key_idx': 0,  # N
            'degree_idx': 0,  # N
            'bass_idx': 0,  # N
            'pitch_abs': np.zeros(12, dtype=np.float32),
            'pitch_root': np.zeros(12, dtype=np.float32),
            'pitch_bass': np.zeros(12, dtype=np.float32)
        }

    key_semitone = note_to_semitone(key)
    root_semitone = note_to_semitone(chord_info['root'])
    bass_semitone = note_to_semitone(chord_info['bass'])

    # Key index (1-12, 0 is N)
    key_idx = key_semitone + 1

    # Scale degree index (1-12, 0 is N)
    degree = get_scale_degree(root_semitone, key_semitone)
    degree_idx = degree + 1

    # Bass index (1-12, 0 is N)
    bass_idx = bass_semitone + 1

    # Pitch presence vectors
    pitch_abs = np.zeros(12, dtype=np.float32)
    pitch_root = np.zeros(12, dtype=np.float32)
    pitch_bass = np.zeros(12, dtype=np.float32)

    # Build chord pitches based on quality
    if chord_info['quality'] == 'maj':
        intervals = [0, 4, 7]  # Root, major 3rd, 5th
    else:
        intervals = [0, 3, 7]  # Root, minor 3rd, 5th

    for interval in intervals:
        # Absolute pitch
        pitch = (root_semitone + interval) % 12
        pitch_abs[pitch] = 1.0

        # Interval from root
        pitch_root[interval] = 1.0

        # Interval from bass
        bass_interval = (root_semitone + interval - bass_semitone) % 12
        pitch_bass[bass_interval] = 1.0

    return {
        'key_idx': key_idx,
        'degree_idx': degree_idx,
        'bass_idx': bass_idx,
        'pitch_abs': pitch_abs,
        'pitch_root': pitch_root,
        'pitch_bass': pitch_bass
    }


def time_to_frame(time_seconds, sr=config.SAMPLE_RATE, hop_length=config.HOP_LENGTH):
    """Convert time in seconds to frame index."""
    return int(time_seconds * sr / hop_length)


def extract_cqt_features(audio_path):
    """
    Extract CQT features with 36 bins per octave.

    Returns:
        features: [n_frames, 252] CQT in dB scale
    """
    try:
        with warnings.catch_warnings(), suppress_stderr():
            warnings.simplefilter("ignore")
            y, sr = librosa.load(audio_path, sr=config.SAMPLE_RATE, mono=True)
    except Exception as e:
        raise ValueError(f"Failed to load audio: {e}")

    if len(y) < config.HOP_LENGTH * 2:
        raise ValueError(f"Audio too short: {len(y)} samples")

    # Compute CQT
    CQT = librosa.cqt(
        y,
        sr=sr,
        hop_length=config.HOP_LENGTH,
        n_bins=config.N_BINS,
        bins_per_octave=config.BINS_PER_OCTAVE,
        fmin=librosa.note_to_hz(config.FMIN_NOTE)
    )

    # Convert to dB scale (as in ChordFormer)
    CQT_db = librosa.amplitude_to_db(np.abs(CQT), ref=np.max)

    # Transpose to [n_frames, n_bins]
    features = CQT_db.T.astype(np.float32)

    return features


def parse_lab_file(lab_path, n_frames, key):
    """
    Parse .lab file and create frame-level scale degree labels.

    Returns:
        labels: dict with arrays for each output head
    """
    # Initialize label arrays
    key_labels = np.zeros(n_frames, dtype=np.int64)
    degree_labels = np.zeros(n_frames, dtype=np.int64)
    bass_labels = np.zeros(n_frames, dtype=np.int64)
    pitch_abs_labels = np.zeros((n_frames, 12), dtype=np.float32)
    pitch_root_labels = np.zeros((n_frames, 12), dtype=np.float32)
    pitch_bass_labels = np.zeros((n_frames, 12), dtype=np.float32)

    # Read annotations
    annotations = []
    with open(lab_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 3:
                start_time = float(parts[0])
                end_time = float(parts[1])
                chord_label = parts[2].strip()
                annotations.append((start_time, end_time, chord_label))

    # Convert annotations to frame-level labels
    for start_time, end_time, chord_label in annotations:
        start_frame = time_to_frame(start_time)
        end_frame = time_to_frame(end_time)

        start_frame = max(0, min(start_frame, n_frames - 1))
        end_frame = max(0, min(end_frame, n_frames))

        # Parse chord and convert to scale degree representation
        chord_info = normalize_chord(chord_label)
        repr_dict = chord_to_scale_degree_representation(chord_info, key)

        # Fill frames
        key_labels[start_frame:end_frame] = repr_dict['key_idx']
        degree_labels[start_frame:end_frame] = repr_dict['degree_idx']
        bass_labels[start_frame:end_frame] = repr_dict['bass_idx']
        pitch_abs_labels[start_frame:end_frame] = repr_dict['pitch_abs']
        pitch_root_labels[start_frame:end_frame] = repr_dict['pitch_root']
        pitch_bass_labels[start_frame:end_frame] = repr_dict['pitch_bass']

    return {
        'key': key_labels,
        'degree': degree_labels,
        'bass': bass_labels,
        'pitch_abs': pitch_abs_labels,
        'pitch_root': pitch_root_labels,
        'pitch_bass': pitch_bass_labels
    }


def process_song(song_dir, output_dir):
    """
    Process a single song: extract features and scale degree labels.

    Returns:
        True if successful, False otherwise
    """
    song_dir = Path(song_dir)
    output_dir = Path(output_dir)

    # Find audio file
    audio_files = list(song_dir.glob('*.mp3'))
    if not audio_files:
        tqdm.write(f"Warning: No MP3 in {song_dir.name}")
        return False

    audio_path = audio_files[0]

    # Find .lab file
    lab_files = list(song_dir.glob('*.lab'))
    if not lab_files:
        tqdm.write(f"Warning: No .lab in {song_dir.name}")
        return False

    lab_path = lab_files[0]

    # Find salami_chords.txt for key
    salami_path = song_dir / 'salami_chords.txt'
    key = None
    if salami_path.exists():
        key = parse_salami_key(salami_path)

    if key is None:
        tqdm.write(f"Warning: No key found for {song_dir.name}, skipping")
        return False

    try:
        # Extract features
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            features = extract_cqt_features(audio_path)

        n_frames = features.shape[0]

        # Parse labels
        labels = parse_lab_file(lab_path, n_frames, key)

        # Save features and labels
        np.save(output_dir / 'features.npy', features)
        np.save(output_dir / 'key_labels.npy', labels['key'])
        np.save(output_dir / 'degree_labels.npy', labels['degree'])
        np.save(output_dir / 'bass_labels.npy', labels['bass'])
        np.save(output_dir / 'pitch_abs_labels.npy', labels['pitch_abs'])
        np.save(output_dir / 'pitch_root_labels.npy', labels['pitch_root'])
        np.save(output_dir / 'pitch_bass_labels.npy', labels['pitch_bass'])

        # Save key info
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump({'key': key, 'n_frames': n_frames}, f)

        return True

    except Exception as e:
        tqdm.write(f"[ERROR] {song_dir.name}: {str(e)[:80]}")
        return False


def compute_normalization_stats(features_dir, song_ids):
    """Compute mean and std from training set."""
    print("Computing normalization statistics...")

    all_features = []
    for song_id in tqdm(song_ids, desc="Loading features"):
        feature_path = Path(features_dir) / f"{song_id:04d}" / 'features.npy'
        if feature_path.exists():
            features = np.load(feature_path)
            all_features.append(features)

    all_features = np.concatenate(all_features, axis=0)

    mean = np.mean(all_features, axis=0).astype(np.float32)
    std = np.std(all_features, axis=0).astype(np.float32)
    std = np.maximum(std, 1e-6)

    return mean, std


def main():
    parser = argparse.ArgumentParser(description='Extract features for MIREX model')
    parser.add_argument('--data_dirs', type=str, nargs='+',
                        default=config.DEFAULT_DATA_DIRS,
                        help='Paths to training data directories')
    parser.add_argument('--output_dir', type=str, default=config.DEFAULT_FEATURES_DIR,
                        help='Path to output features directory')
    args = parser.parse_args()

    data_dirs = [Path(d) for d in args.data_dirs]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all songs
    all_song_dirs = []
    for data_dir in data_dirs:
        if data_dir.exists():
            song_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
            all_song_dirs.extend(song_dirs)
            print(f"Found {len(song_dirs)} songs in {data_dir}")

    print(f"\nTotal songs: {len(all_song_dirs)}")

    # Process songs with sequential numbering
    successful = 0
    processed_ids = []

    for new_id, song_dir in enumerate(tqdm(all_song_dirs, desc="Extracting features"), start=1):
        song_output_dir = output_dir / f"{new_id:04d}"
        song_output_dir.mkdir(parents=True, exist_ok=True)

        if process_song(song_dir, song_output_dir):
            successful += 1
            processed_ids.append(new_id)

    print(f"\nSuccessfully processed {successful}/{len(all_song_dirs)} songs")

    # Create data split
    random.seed(42)
    shuffled_ids = processed_ids.copy()
    random.shuffle(shuffled_ids)

    total = len(shuffled_ids)
    train_split = int(0.8 * total)
    val_split = int(0.9 * total)

    train_ids = sorted(shuffled_ids[:train_split])
    val_ids = sorted(shuffled_ids[train_split:val_split])
    test_ids = sorted(shuffled_ids[val_split:])

    data_split = {'train': train_ids, 'val': val_ids, 'test': test_ids}

    with open(output_dir / 'data_split.json', 'w') as f:
        json.dump(data_split, f, indent=2)

    print(f"Data split: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")

    # Compute normalization
    mean, std = compute_normalization_stats(output_dir, train_ids)

    normalization = {'mean': mean.tolist(), 'std': std.tolist()}
    with open(output_dir / 'normalization.json', 'w') as f:
        json.dump(normalization, f)

    print("Feature extraction complete!")


if __name__ == '__main__':
    main()
