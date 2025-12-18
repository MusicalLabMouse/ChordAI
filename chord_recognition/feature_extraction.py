"""
Feature Extraction for Chord Recognition
Extracts CQT chromagram features from MP3 files and aligns with chord annotations.
Supports both legacy (84-bin) and ChordFormer (252-bin) feature extraction.
"""

import os
import sys
import json
import argparse
import random
import re
import numpy as np
import librosa
import warnings
from tqdm import tqdm
from pathlib import Path
from contextlib import contextmanager
import torch

# Try to import nnAudio for GPU-accelerated CQT
try:
    from nnAudio.features import CQT
    NNAUDIO_AVAILABLE = True
except ImportError:
    NNAUDIO_AVAILABLE = False
    print("Note: nnAudio not installed. Using librosa (CPU). Install with: pip install nnAudio")

# Suppress audioread warnings for corrupted MP3s
warnings.filterwarnings('ignore', category=UserWarning, module='librosa')
warnings.filterwarnings('ignore', category=FutureWarning)


@contextmanager
def suppress_stderr():
    """Suppress stderr output (for mpg123 error messages)."""
    null_fd = os.open(os.devnull, os.O_RDWR)
    save_stderr = os.dup(2)
    os.dup2(null_fd, 2)
    try:
        yield
    finally:
        os.dup2(save_stderr, 2)
        os.close(null_fd)


# Audio processing parameters
SAMPLE_RATE = 22050
HOP_LENGTH = 512  # ChordFormer uses 512, legacy used 2048
FMIN = librosa.note_to_hz('C1')  # C1 = 32.7 Hz

# Legacy parameters (for backward compatibility)
N_BINS_LEGACY = 84
BINS_PER_OCTAVE_LEGACY = 12

# ChordFormer parameters
N_BINS_CHORDFORMER = 252
BINS_PER_OCTAVE_CHORDFORMER = 36
N_OCTAVES = 7

# Chord component mappings for structured representation
ROOT_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
ROOT_TO_IDX = {root: idx for idx, root in enumerate(ROOT_NAMES)}

# Enharmonic normalization (prefer sharps for consistency)
ENHARMONIC_MAP = {
    'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#',
    'Cb': 'B', 'Fb': 'E', 'B#': 'C', 'E#': 'F'
}

# Triad types (index 0 = N for "no chord")
TRIAD_TYPES = ['maj', 'min', 'sus4', 'sus2', 'dim', 'aug']
TRIAD_TO_IDX = {t: idx for idx, t in enumerate(TRIAD_TYPES)}

# Extension mappings
SEVENTH_MAP = {'N': 0, '7': 1, 'b7': 2, 'bb7': 3, 'maj7': 1, 'min7': 2, '6': 3}
NINTH_MAP = {'N': 0, '9': 1, '#9': 2, 'b9': 3}
ELEVENTH_MAP = {'N': 0, '11': 1, '#11': 2}
THIRTEENTH_MAP = {'N': 0, '13': 1, 'b13': 2}


def normalize_root(root):
    """Normalize root note to sharp spelling (C, C#, D, D#, ...)."""
    if root in ENHARMONIC_MAP:
        return ENHARMONIC_MAP[root]
    return root


def parse_chord_structured(chord_label):
    """
    Parse a chord label into 6-component structured representation.

    Returns:
        Dictionary with keys: root_triad, bass, 7th, 9th, 11th, 13th
        All values are integer indices.

    root_triad encoding:
        0 = N (no chord)
        1-84 = root (0-11) * 7 + triad_type (0-6) + 1
            triad_type: 0=maj, 1=min, 2=sus4, 3=sus2, 4=dim, 5=aug, 6=N-quality
    """
    result = {
        'root_triad': 0,  # N
        'bass': 0,        # N
        '7th': 0,         # N
        '9th': 0,         # N
        '11th': 0,        # N
        '13th': 0         # N
    }

    # Handle no-chord
    if chord_label in ('N', 'X', ''):
        return result

    # Handle bass note (slash chord)
    bass_note = None
    if '/' in chord_label:
        parts = chord_label.split('/')
        chord_label = parts[0]
        bass_note = parts[1] if len(parts) > 1 else None

    # Parse root and quality
    if ':' in chord_label:
        root, quality = chord_label.split(':', 1)
    else:
        root = chord_label
        quality = 'maj'

    # Normalize root
    root = normalize_root(root)
    if root not in ROOT_TO_IDX:
        return result  # Unknown root, return N

    root_idx = ROOT_TO_IDX[root]

    # Parse quality to get triad type
    quality_lower = quality.lower()

    # Determine triad type
    if quality_lower.startswith('min') or quality_lower.startswith('m') and not quality_lower.startswith('maj'):
        triad_idx = 1  # min
    elif 'sus4' in quality_lower:
        triad_idx = 2  # sus4
    elif 'sus2' in quality_lower:
        triad_idx = 3  # sus2
    elif 'dim' in quality_lower or 'hdim' in quality_lower:
        triad_idx = 4  # dim
    elif 'aug' in quality_lower:
        triad_idx = 5  # aug
    elif quality_lower.startswith('maj') or quality_lower == '' or quality_lower in ('7', '9', '11', '13'):
        triad_idx = 0  # maj
    else:
        triad_idx = 0  # default to maj

    # Compute root_triad index: 1 + root * 7 + triad
    result['root_triad'] = 1 + root_idx * 7 + triad_idx

    # Parse bass note
    if bass_note:
        bass_note = normalize_root(bass_note)
        if bass_note in ROOT_TO_IDX:
            result['bass'] = ROOT_TO_IDX[bass_note] + 1  # 1-indexed (0 = N)
    else:
        # Bass is same as root
        result['bass'] = root_idx + 1

    # Parse extensions from quality string
    # 7th
    if 'maj7' in quality_lower:
        result['7th'] = 1  # maj7
    elif 'min7' in quality_lower or 'm7' in quality_lower:
        result['7th'] = 2  # b7 (minor 7th)
    elif 'dim7' in quality_lower:
        result['7th'] = 3  # bb7 (diminished 7th)
    elif '7' in quality:  # Check original case for dominant 7
        result['7th'] = 2  # b7 (dominant 7th)

    # 9th
    if '#9' in quality or 'sharp9' in quality_lower:
        result['9th'] = 2
    elif 'b9' in quality or 'flat9' in quality_lower:
        result['9th'] = 3
    elif '9' in quality:
        result['9th'] = 1

    # 11th
    if '#11' in quality or 'sharp11' in quality_lower:
        result['11th'] = 2
    elif '11' in quality:
        result['11th'] = 1

    # 13th
    if 'b13' in quality or 'flat13' in quality_lower:
        result['13th'] = 2
    elif '13' in quality:
        result['13th'] = 1

    return result


def time_to_frame(time_seconds, sr=SAMPLE_RATE, hop_length=HOP_LENGTH):
    """Convert time in seconds to frame index."""
    return int(time_seconds * sr / hop_length)


def normalize_chord(chord_label):
    """
    Normalize chord to 25 major/minor classes.

    Examples:
        C:maj, C:maj7, C:maj9 -> C:maj
        C:min, C:min7, C:minmaj7 -> C:min
        C#:maj -> Db:maj (enharmonic)
        N -> N
        X -> N (unknown)

    Args:
        chord_label: Original chord label

    Returns:
        Normalized chord label (root:maj or root:min or N)
    """
    # No chord or unknown
    if chord_label == 'N' or chord_label == 'X':
        return 'N'

    # Split into root and quality
    if ':' not in chord_label:
        # Just a root note, assume major
        root = chord_label
        quality = 'maj'
    else:
        parts = chord_label.split(':')
        root = parts[0]
        quality = parts[1] if len(parts) > 1 else 'maj'

    # Normalize enharmonic equivalents to preferred spelling
    # Based on dataset frequency: use Bb, Db, Eb, F#, Ab
    enharmonic_map = {
        'A#': 'Bb',
        'C#': 'Db',
        'D#': 'Eb',
        'Gb': 'F#',
        'G#': 'Ab',
        'Cb': 'B',
        'Fb': 'E',
    }

    if root in enharmonic_map:
        root = enharmonic_map[root]

    # Determine if minor or major
    # Minor indicators: min, dim, hdim
    # NOTE: Don't use startswith('m') because it catches 'maj'!
    quality_lower = quality.lower()

    if 'min' in quality_lower or 'dim' in quality_lower:
        return f"{root}:min"
    else:
        # Everything else is major (maj, sus, aug, 7, 9, etc.)
        return f"{root}:maj"


def build_chord_vocabulary(data_dirs):
    """
    Scan all .lab files to build chord vocabulary.

    Args:
        data_dirs: List of paths to training data directories

    Returns:
        chord_to_idx: Dictionary mapping chord labels to indices
        idx_to_chord: Dictionary mapping indices to chord labels
    """
    print("Building chord vocabulary...")
    chord_set = set()

    # Scan all data directories
    for data_dir in data_dirs:
        data_path = Path(data_dir)
        if not data_path.exists():
            print(f"Warning: {data_dir} does not exist, skipping")
            continue

        for song_dir in sorted(data_path.iterdir()):
            if not song_dir.is_dir():
                continue

            # Find .lab file
            lab_files = list(song_dir.glob('*.lab'))
            if not lab_files:
                continue

            lab_file = lab_files[0]

            # Read chord labels
            with open(lab_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split('\t')
                    if len(parts) >= 3:
                        chord_label = parts[2].strip()
                        # Normalize to 25 classes (maj/min)
                        normalized_chord = normalize_chord(chord_label)
                        chord_set.add(normalized_chord)

    # Sort chords and create mappings (ensure 'N' is index 0)
    chords = sorted(list(chord_set))
    if 'N' in chords:
        chords.remove('N')
        chords = ['N'] + chords

    chord_to_idx = {chord: idx for idx, chord in enumerate(chords)}
    idx_to_chord = {idx: chord for chord, idx in chord_to_idx.items()}

    print(f"Found {len(chords)} unique chords")
    print(f"Sample chords: {chords[:10]}")

    return chord_to_idx, idx_to_chord


# Global GPU CQT transform (initialized once for efficiency)
_gpu_cqt_transform = None
_gpu_device = None


def get_gpu_cqt_transform(n_bins, bins_per_octave, device):
    """Get or create GPU CQT transform (cached for reuse)."""
    global _gpu_cqt_transform, _gpu_device

    if _gpu_cqt_transform is None or _gpu_device != device:
        _gpu_cqt_transform = CQT(
            sr=SAMPLE_RATE,
            hop_length=HOP_LENGTH,
            fmin=FMIN,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            output_format='Magnitude'
        ).to(device)
        _gpu_device = device

    return _gpu_cqt_transform


def extract_cqt_features(audio_path, model_type='chordformer', device=None):
    """
    Extract CQT chromagram features from audio file.

    Args:
        audio_path: Path to audio file (MP3)
        model_type: 'chordformer' (252 bins) or 'legacy' (84 bins)
        device: torch device for GPU acceleration (None = auto-detect)

    Returns:
        features: CQT features, shape [n_frames, n_bins]
    """
    # Select parameters based on model type
    if model_type == 'chordformer':
        n_bins = N_BINS_CHORDFORMER
        bins_per_octave = BINS_PER_OCTAVE_CHORDFORMER
    else:  # legacy
        n_bins = N_BINS_LEGACY
        bins_per_octave = BINS_PER_OCTAVE_LEGACY

    # Load audio file
    try:
        with warnings.catch_warnings(), suppress_stderr():
            warnings.simplefilter("ignore")
            y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        raise ValueError(f"Failed to load audio: {e}")

    # Check if audio is too short or empty
    if len(y) < HOP_LENGTH * 2:
        raise ValueError(f"Audio too short: {len(y)} samples")

    # Use GPU acceleration if available
    if NNAUDIO_AVAILABLE and device is not None and device.type == 'cuda':
        # GPU-accelerated CQT using nnAudio
        cqt_transform = get_gpu_cqt_transform(n_bins, bins_per_octave, device)

        # Convert to tensor and compute CQT
        y_tensor = torch.from_numpy(y).float().unsqueeze(0).to(device)
        with torch.no_grad():
            CQT_mag = cqt_transform(y_tensor)  # [1, n_bins, n_frames]

        # Convert to dB scale (like paper's amplitude_to_db with ref=np.max)
        CQT_mag_np = CQT_mag.squeeze(0).cpu().numpy()  # [n_bins, n_frames]
        features = librosa.amplitude_to_db(CQT_mag_np, ref=np.max)

        # Transpose to [n_frames, n_bins]
        features = features.T.astype(np.float32)
    else:
        # CPU fallback using librosa
        CQT = librosa.cqt(
            y,
            sr=sr,
            hop_length=HOP_LENGTH,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            fmin=FMIN
        )

        # Convert to dB scale (like paper's amplitude_to_db with ref=np.max)
        features = librosa.amplitude_to_db(np.abs(CQT), ref=np.max)

        # Transpose to shape [n_frames, n_bins]
        features = features.T.astype(np.float32)

    return features


def parse_lab_file(lab_path, n_frames, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
                   chord_to_idx=None, model_type='chordformer'):
    """
    Parse .lab file and create frame-level chord labels.

    Args:
        lab_path: Path to .lab annotation file
        n_frames: Number of frames in the audio
        sr: Sample rate
        hop_length: Hop length for frame alignment
        chord_to_idx: Dictionary mapping chord labels to indices (for legacy mode)
        model_type: 'chordformer' for 6-head labels, 'legacy' for single class

    Returns:
        For legacy mode: labels array, shape [n_frames]
        For chordformer mode: dict of 6 label arrays
    """
    if model_type == 'chordformer':
        # Initialize 6-head structured labels
        labels = {
            'root_triad': np.zeros(n_frames, dtype=np.int64),
            'bass': np.zeros(n_frames, dtype=np.int64),
            '7th': np.zeros(n_frames, dtype=np.int64),
            '9th': np.zeros(n_frames, dtype=np.int64),
            '11th': np.zeros(n_frames, dtype=np.int64),
            '13th': np.zeros(n_frames, dtype=np.int64)
        }
    else:
        labels = np.zeros(n_frames, dtype=np.int64)

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
        start_frame = time_to_frame(start_time, sr, hop_length)
        end_frame = time_to_frame(end_time, sr, hop_length)

        # Clip to valid range
        start_frame = max(0, min(start_frame, n_frames - 1))
        end_frame = max(0, min(end_frame, n_frames))

        if model_type == 'chordformer':
            # Parse chord into 6-component representation
            chord_components = parse_chord_structured(chord_label)

            # Fill frames for each head
            for head_name, head_value in chord_components.items():
                labels[head_name][start_frame:end_frame] = head_value
        else:
            # Legacy: normalize chord to 25 classes
            normalized_chord = normalize_chord(chord_label)

            # Get chord index
            if chord_to_idx and normalized_chord in chord_to_idx:
                chord_idx = chord_to_idx[normalized_chord]
            else:
                chord_idx = 0  # Default to 'N' if unknown chord

            # Fill frames
            labels[start_frame:end_frame] = chord_idx

    return labels


def process_song(song_dir, output_dir, chord_to_idx):
    """
    Process a single song: extract features and labels.

    Args:
        song_dir: Path to song directory
        output_dir: Path to output directory
        chord_to_idx: Chord vocabulary mapping

    Returns:
        True if successful, False otherwise
    """
    song_dir = Path(song_dir)
    song_id = song_dir.name

    # Find audio file (MP3)
    audio_files = list(song_dir.glob('*.mp3'))
    if not audio_files:
        print(f"Warning: No MP3 file found in {song_id}")
        return False

    audio_path = audio_files[0]

    # Find .lab file
    lab_files = list(song_dir.glob('*.lab'))
    if not lab_files:
        print(f"Warning: No .lab file found in {song_id}")
        return False

    lab_path = lab_files[0]

    try:
        # Extract CQT features
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            features = extract_cqt_features(audio_path)

        n_frames = features.shape[0]

        # Parse labels
        labels = parse_lab_file(lab_path, n_frames, chord_to_idx=chord_to_idx)

        # Create output directory
        song_output_dir = Path(output_dir) / song_id
        song_output_dir.mkdir(parents=True, exist_ok=True)

        # Save features and labels
        np.save(song_output_dir / 'features.npy', features)
        np.save(song_output_dir / 'labels.npy', labels)

        return True

    except Exception as e:
        # Log error but continue processing - show filename and error
        tqdm.write(f"\n[ERROR] song_id={song_id}")
        tqdm.write(f"  MP3: {audio_path.name}")
        tqdm.write(f"  Issue: {str(e)[:80]}")
        return False


def process_song_to_dir(song_dir, output_dir, chord_to_idx=None, model_type='chordformer', device=None):
    """
    Process a single song and save to specified output directory.

    Args:
        song_dir: Path to song directory (contains MP3 and .lab files)
        output_dir: Path to output directory (already created)
        chord_to_idx: Chord vocabulary mapping (for legacy mode)
        model_type: 'chordformer' or 'legacy'
        device: torch device for GPU acceleration

    Returns:
        True if successful, False otherwise
    """
    song_dir = Path(song_dir)
    output_dir = Path(output_dir)

    # Find audio file (MP3)
    audio_files = list(song_dir.glob('*.mp3'))
    if not audio_files:
        tqdm.write(f"Warning: No MP3 file found in {song_dir.name}")
        return False

    audio_path = audio_files[0]

    # Find .lab file
    lab_files = list(song_dir.glob('*.lab'))
    if not lab_files:
        tqdm.write(f"Warning: No .lab file found in {song_dir.name}")
        return False

    lab_path = lab_files[0]

    try:
        # Extract CQT features
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            features = extract_cqt_features(audio_path, model_type=model_type, device=device)

        n_frames = features.shape[0]

        # Parse labels
        labels = parse_lab_file(lab_path, n_frames, chord_to_idx=chord_to_idx,
                               model_type=model_type)

        # Save features
        np.save(output_dir / 'features.npy', features)

        # Save labels (different format for each mode)
        if model_type == 'chordformer':
            # Save each head's labels separately for efficient loading
            for head_name, head_labels in labels.items():
                np.save(output_dir / f'labels_{head_name}.npy', head_labels)
            # Also save combined for compatibility
            np.savez(output_dir / 'labels.npz', **labels)
        else:
            np.save(output_dir / 'labels.npy', labels)

        return True

    except Exception as e:
        # Log error but continue processing
        tqdm.write(f"\n[ERROR] {song_dir.name}: {str(e)[:80]}")
        return False


def compute_normalization_stats(features_dir, train_song_ids):
    """
    Compute mean and std from training set for z-score normalization.

    Args:
        features_dir: Path to features directory
        train_song_ids: List of training song IDs

    Returns:
        mean: Mean vector, shape [84]
        std: Std vector, shape [84]
    """
    print("Computing normalization statistics from training set...")

    all_features = []
    for song_id in tqdm(train_song_ids, desc="Loading training features"):
        feature_path = Path(features_dir) / f"{song_id:04d}" / 'features.npy'
        if feature_path.exists():
            features = np.load(feature_path)
            all_features.append(features)

    # Concatenate all features
    all_features = np.concatenate(all_features, axis=0)

    # Compute statistics
    mean = np.mean(all_features, axis=0).astype(np.float32)
    std = np.std(all_features, axis=0).astype(np.float32)

    # Avoid division by zero
    std = np.maximum(std, 1e-6)

    print(f"Mean shape: {mean.shape}, Std shape: {std.shape}")

    return mean, std


def main():
    parser = argparse.ArgumentParser(description='Extract CQT features for chord recognition')
    parser.add_argument('--data_dirs', type=str, nargs='+',
                        default=['../training_datasets/training_data', '../training_datasets/training_data_set2'],
                        help='Paths to training data directories')
    parser.add_argument('--output_dir', type=str, default='../features',
                        help='Path to output features directory')
    parser.add_argument('--model_type', type=str, default='chordformer',
                        choices=['chordformer', 'legacy'],
                        help='Model type: chordformer (252 bins, 6-head) or legacy (84 bins)')
    args = parser.parse_args()

    model_type = args.model_type
    data_dirs = [Path(d) for d in args.data_dirs]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model type: {model_type}")
    if model_type == 'chordformer':
        print(f"  CQT bins: {N_BINS_CHORDFORMER} ({BINS_PER_OCTAVE_CHORDFORMER} bins/octave)")
        print(f"  Labels: 6-head structured (root_triad, bass, 7th, 9th, 11th, 13th)")
    else:
        print(f"  CQT bins: {N_BINS_LEGACY} ({BINS_PER_OCTAVE_LEGACY} bins/octave)")
        print(f"  Labels: single class (25 maj/min chords)")

    # Save model type metadata
    metadata = {
        'model_type': model_type,
        'n_bins': N_BINS_CHORDFORMER if model_type == 'chordformer' else N_BINS_LEGACY,
        'bins_per_octave': BINS_PER_OCTAVE_CHORDFORMER if model_type == 'chordformer' else BINS_PER_OCTAVE_LEGACY,
        'hop_length': HOP_LENGTH,
        'sample_rate': SAMPLE_RATE
    }
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # Step 1: Build chord vocabulary (only needed for legacy mode)
    chord_to_idx = None
    if model_type == 'legacy':
        chord_to_idx, idx_to_chord = build_chord_vocabulary(data_dirs)

        # Save vocabulary
        with open(output_dir / 'chord_to_idx.json', 'w') as f:
            json.dump(chord_to_idx, f, indent=2)

        with open(output_dir / 'idx_to_chord.json', 'w') as f:
            json.dump(idx_to_chord, f, indent=2)

        print(f"Saved chord vocabulary to {output_dir}")

    # Step 2: Collect all songs from all directories
    all_song_dirs = []
    for data_dir in data_dirs:
        if data_dir.exists():
            song_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
            all_song_dirs.extend(song_dirs)
            print(f"Found {len(song_dirs)} songs in {data_dir}")

    print(f"\nTotal songs to process: {len(all_song_dirs)}")

    # Step 3: Set up device for GPU acceleration
    device = None
    if NNAUDIO_AVAILABLE and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"\nUsing GPU acceleration: {torch.cuda.get_device_name(0)}")
        # Pre-initialize the CQT transform
        if model_type == 'chordformer':
            _ = get_gpu_cqt_transform(N_BINS_CHORDFORMER, BINS_PER_OCTAVE_CHORDFORMER, device)
        else:
            _ = get_gpu_cqt_transform(N_BINS_LEGACY, BINS_PER_OCTAVE_LEGACY, device)
        print("CQT transform initialized on GPU")
    else:
        if not NNAUDIO_AVAILABLE:
            print("\nUsing CPU (install nnAudio for GPU acceleration: pip install nnAudio)")
        elif not torch.cuda.is_available():
            print("\nUsing CPU (no CUDA GPU available)")

    # Step 4: Process all songs with sequential numbering
    successful = 0
    processed_ids = []

    for new_id, song_dir in enumerate(tqdm(all_song_dirs, desc="Extracting features"), start=1):
        # Create output directory with new sequential ID
        song_output_dir = output_dir / f"{new_id:04d}"
        song_output_dir.mkdir(parents=True, exist_ok=True)

        # Process the song
        if process_song_to_dir(song_dir, song_output_dir, chord_to_idx=chord_to_idx,
                               model_type=model_type, device=device):
            successful += 1
            processed_ids.append(new_id)

    print(f"\nSuccessfully processed {successful}/{len(all_song_dirs)} songs")

    # Step 5: Create data split (80/10/10) using shuffled IDs for better distribution
    random.seed(42)  # For reproducibility
    shuffled_ids = processed_ids.copy()
    random.shuffle(shuffled_ids)

    total_songs = len(shuffled_ids)
    train_split = int(0.8 * total_songs)
    val_split = int(0.9 * total_songs)

    train_ids = sorted(shuffled_ids[:train_split])
    val_ids = sorted(shuffled_ids[train_split:val_split])
    test_ids = sorted(shuffled_ids[val_split:])

    data_split = {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids
    }

    with open(output_dir / 'data_split.json', 'w') as f:
        json.dump(data_split, f, indent=2)

    print(f"\nData split: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")

    # Step 6: Compute normalization statistics from training set
    mean, std = compute_normalization_stats(output_dir, train_ids)

    normalization = {
        'mean': mean.tolist(),
        'std': std.tolist()
    }

    with open(output_dir / 'normalization.json', 'w') as f:
        json.dump(normalization, f)

    print(f"Saved normalization statistics to {output_dir}/normalization.json")
    print("\nFeature extraction complete!")


if __name__ == '__main__':
    main()
