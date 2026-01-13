"""
PyTorch Dataset for Chord Recognition
Loads pre-computed CQT features and frame-level labels.
Supports both legacy (single-class) and ChordFormer (6-head) label formats.
Includes data augmentation for improved generalization.
"""

import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

# ChordFormer label heads
CHORDFORMER_HEADS = ['root_triad', 'bass', '7th', '9th', '11th', '13th']

# MIREX label heads (for degree-based model)
MIREX_HEADS_CATEGORICAL = ['key', 'degree', 'bass']
MIREX_HEADS_BINARY = ['pitches_abs', 'intervals_root', 'intervals_bass']
MIREX_HEADS = MIREX_HEADS_CATEGORICAL + MIREX_HEADS_BINARY


def shift_chord_labels(labels, semitone_shift, num_classes):
    """
    Shift chord labels to match pitch-shifted features.

    The chord vocabulary is organized as:
    - Index 0: N (no chord)
    - Indices 1-36: 12 roots × 3 qualities (maj, min, sus)
      (A:maj=1, A:min=2, A:sus=3, Ab:maj=4, Ab:min=5, Ab:sus=6, ...)

    Each root occupies 3 indices (maj, min, sus), so there are 12 roots.

    Args:
        labels: numpy array of chord indices
        semitone_shift: number of semitones to shift (+/-)
        num_classes: total number of classes (37)

    Returns:
        shifted_labels: numpy array with shifted chord indices
    """
    shifted = labels.copy()

    # Only shift non-N chords (index > 0)
    mask = labels > 0

    if mask.any():
        # Convert label index to (root_idx, quality)
        # Labels 1-36 map to 12 roots × 3 qualities
        # root_idx = (label - 1) // 3  (0-11)
        # quality = (label - 1) % 3    (0=maj, 1=min, 2=sus)

        valid_labels = labels[mask]
        root_idx = (valid_labels - 1) // 3  # 0-11
        quality = (valid_labels - 1) % 3     # 0=maj, 1=min, 2=sus

        # Shift root by semitones (mod 12)
        new_root_idx = (root_idx + semitone_shift) % 12

        # Convert back to label index
        new_labels = new_root_idx * 3 + quality + 1

        shifted[mask] = new_labels

    return shifted


class ChordDataset(Dataset):
    """
    Dataset class for chord recognition.
    Loads pre-computed CQT features and applies z-score normalization.
    Supports on-the-fly data augmentation for training.
    """

    def __init__(self, features_dir, song_ids, normalization_path=None,
                 augment=False, num_classes=25):
        """
        Args:
            features_dir: Path to features directory
            song_ids: List of song IDs to include
            normalization_path: Path to normalization.json (optional)
            augment: Whether to apply data augmentation (for training only)
            num_classes: Number of chord classes (for pitch shift augmentation)
        """
        self.features_dir = Path(features_dir)
        self.song_ids = song_ids
        self.augment = augment
        self.num_classes = num_classes

        # Load normalization statistics
        self.mean = None
        self.std = None
        if normalization_path and Path(normalization_path).exists():
            with open(normalization_path, 'r') as f:
                norm_data = json.load(f)
                self.mean = np.array(norm_data['mean'], dtype=np.float32)
                self.std = np.array(norm_data['std'], dtype=np.float32)

    def __len__(self):
        return len(self.song_ids)

    def __getitem__(self, idx):
        """
        Returns:
            features: CQT features, shape [n_frames, 84]
            labels: Frame-level labels, shape [n_frames]
            length: Sequence length (number of frames)
        """
        song_id = self.song_ids[idx]
        song_dir = self.features_dir / f"{song_id:04d}"

        # Load features and labels
        features = np.load(song_dir / 'features.npy')
        labels = np.load(song_dir / 'labels.npy')

        # Apply data augmentation (training only)
        if self.augment:
            features, labels = self._apply_augmentation(features, labels)

        # Apply z-score normalization (after augmentation)
        if self.mean is not None and self.std is not None:
            features = (features - self.mean) / self.std

        # Convert to tensors
        features = torch.from_numpy(features).float()  # [n_frames, 84]
        labels = torch.from_numpy(labels).long()       # [n_frames]
        length = features.shape[0]

        return features, labels, length

    def _apply_augmentation(self, features, labels):
        """
        Apply data augmentation to features and labels.

        Augmentations applied:
        1. Pitch shift (50% prob): Roll CQT bins by ±1-2 semitones
        2. Time masking (30% prob): Zero out 1-3 consecutive frames
        3. Gaussian noise (30% prob): Add small random noise

        Args:
            features: CQT features [n_frames, 84]
            labels: Frame-level labels [n_frames]

        Returns:
            augmented_features, augmented_labels
        """
        features = features.copy()  # Don't modify original
        labels = labels.copy()

        # 1. Pitch shift - DISABLED for now
        # The chord vocabulary is in alphabetical order, not chromatic order,
        # so label shifting doesn't work correctly. Would need a proper
        # root-to-semitone mapping to fix this.
        # TODO: Implement proper chromatic pitch shifting with vocabulary lookup

        # 2. Time masking (SpecAugment-style)
        if random.random() < 0.3:
            n_frames = len(features)
            if n_frames > 5:
                mask_len = random.randint(1, 3)
                mask_start = random.randint(0, n_frames - mask_len - 1)
                features[mask_start:mask_start + mask_len] = 0

        # 3. Gaussian noise
        if random.random() < 0.3:
            noise_std = 0.1
            noise = np.random.normal(0, noise_std, features.shape).astype(np.float32)
            features = features + noise

        return features, labels


class ChordFormerDataset(Dataset):
    """
    Dataset class for ChordFormer model with 6-head structured chord labels.
    Loads pre-computed CQT features (252 bins) and multi-head labels.
    """

    def __init__(self, features_dir, song_ids, normalization_path=None,
                 augment=False, sequence_length=None):
        """
        Args:
            features_dir: Path to features directory
            song_ids: List of song IDs to include
            normalization_path: Path to normalization.json (optional)
            augment: Whether to apply data augmentation (for training only)
            sequence_length: If set, extract random windows of this length (training)
        """
        self.features_dir = Path(features_dir)
        self.song_ids = song_ids
        self.augment = augment
        self.sequence_length = sequence_length

        # Load normalization statistics
        self.mean = None
        self.std = None
        if normalization_path and Path(normalization_path).exists():
            with open(normalization_path, 'r') as f:
                norm_data = json.load(f)
                self.mean = np.array(norm_data['mean'], dtype=np.float32)
                self.std = np.array(norm_data['std'], dtype=np.float32)

    def __len__(self):
        return len(self.song_ids)

    def __getitem__(self, idx):
        """
        Returns:
            features: CQT features, shape [n_frames, 252]
            labels: Dict of 6 label arrays, each shape [n_frames]
            length: Sequence length (number of frames)
        """
        song_id = self.song_ids[idx]
        song_dir = self.features_dir / f"{song_id:04d}"

        # Load features
        features = np.load(song_dir / 'features.npy')

        # Load multi-head labels
        labels = {}
        for head in CHORDFORMER_HEADS:
            label_path = song_dir / f'labels_{head}.npy'
            if label_path.exists():
                labels[head] = np.load(label_path)
            else:
                # Fallback to combined file
                labels_combined = np.load(song_dir / 'labels.npz')
                labels[head] = labels_combined[head]

        # Extract random window during training
        if self.sequence_length and len(features) > self.sequence_length:
            start_idx = random.randint(0, len(features) - self.sequence_length)
            end_idx = start_idx + self.sequence_length
            features = features[start_idx:end_idx]
            for head in CHORDFORMER_HEADS:
                labels[head] = labels[head][start_idx:end_idx]

        # Apply data augmentation (training only)
        if self.augment:
            features, labels = self._apply_augmentation(features, labels)

        # Apply z-score normalization (after augmentation)
        if self.mean is not None and self.std is not None:
            features = (features - self.mean) / self.std

        # Convert to tensors
        features = torch.from_numpy(features).float()  # [n_frames, 252]
        labels_tensors = {
            head: torch.from_numpy(labels[head]).long()
            for head in CHORDFORMER_HEADS
        }
        length = features.shape[0]

        return features, labels_tensors, length

    def _apply_augmentation(self, features, labels):
        """
        Apply data augmentation to features and labels.

        Augmentations:
        1. Pitch shift (50% prob): Roll CQT bins and shift chord roots
        2. Time masking (30% prob): Zero out 1-3 consecutive frames
        3. Gaussian noise (30% prob): Add small random noise

        Args:
            features: CQT features [n_frames, 252]
            labels: Dict of label arrays

        Returns:
            augmented_features, augmented_labels
        """
        features = features.copy()
        labels = {head: arr.copy() for head, arr in labels.items()}
        n_bins = features.shape[1]
        bins_per_octave = 36  # ChordFormer uses 36 bins/octave

        # 1. Pitch shift (-5 to +6 semitones as per ChordFormer paper)
        # Paper applies pitch augmentation to ALL training data, so we use high probability
        if random.random() < 0.9:  # 90% to always get pitch variation
            semitone_shift = random.randint(-5, 6)
            bin_shift = semitone_shift * (bins_per_octave // 12)  # 3 bins per semitone

            if bin_shift != 0:
                # Roll features along frequency axis
                features = np.roll(features, bin_shift, axis=1)
                # Zero out wrapped bins (they would wrap around, which is incorrect)
                if bin_shift > 0:
                    features[:, :bin_shift] = 0  # Zero low frequencies
                else:
                    features[:, bin_shift:] = 0  # Zero high frequencies

                # Shift root-based labels to match transposed audio
                labels = self._shift_chord_labels(labels, semitone_shift)

        # 2. Time masking (SpecAugment-style)
        if random.random() < 0.3:
            n_frames = len(features)
            if n_frames > 5:
                mask_len = random.randint(1, 3)
                mask_start = random.randint(0, n_frames - mask_len - 1)
                features[mask_start:mask_start + mask_len] = 0

        # 3. Gaussian noise
        if random.random() < 0.3:
            noise_std = 0.1
            noise = np.random.normal(0, noise_std, features.shape).astype(np.float32)
            features = features + noise

        return features, labels

    def _shift_chord_labels(self, labels, semitone_shift):
        """
        Shift chord labels to match pitch-shifted features.

        For root_triad: shift root by semitones
        For bass: shift bass note by semitones
        Extensions (7th, 9th, 11th, 13th) are not affected by pitch shift.
        """
        shifted = {head: arr.copy() for head, arr in labels.items()}

        # Shift root_triad
        # root_triad: 0=N, 1-84 = root*7 + triad + 1
        root_triad = labels['root_triad']
        mask = root_triad > 0
        if mask.any():
            valid = root_triad[mask]
            root_idx = (valid - 1) // 7  # 0-11
            triad_idx = (valid - 1) % 7  # 0-6
            new_root = (root_idx + semitone_shift) % 12
            shifted['root_triad'][mask] = new_root * 7 + triad_idx + 1

        # Shift bass
        # bass: 0=N, 1-12 = bass note index
        bass = labels['bass']
        mask = bass > 0
        if mask.any():
            valid = bass[mask]
            new_bass = (valid - 1 + semitone_shift) % 12 + 1
            shifted['bass'][mask] = new_bass

        return shifted


# =============================================================================
# MIREX 2025 Dataset with Time-Stretching Augmentation
# =============================================================================

def apply_time_stretch(features, labels, rate):
    """
    Apply time-stretching to CQT features and adjust labels.

    Uses scipy interpolation for efficiency on pre-computed features.

    Args:
        features: CQT features [n_frames, n_bins]
        labels: Dict of label arrays (categorical: [n_frames], binary: [n_frames, 12])
        rate: Stretch rate (0.8 = slower/longer, 1.2 = faster/shorter)

    Returns:
        stretched_features, stretched_labels
    """
    import scipy.ndimage

    n_frames_orig = features.shape[0]

    # Time-stretch features (interpolation along time axis)
    # rate > 1 = faster (fewer frames), rate < 1 = slower (more frames)
    stretched_features = scipy.ndimage.zoom(features, (1/rate, 1), order=1)

    n_frames_new = stretched_features.shape[0]

    # Stretch labels to match new length
    stretched_labels = {}
    for head, arr in labels.items():
        if arr.ndim == 1:
            # Categorical labels - use nearest-neighbor interpolation
            indices = np.round(np.linspace(0, len(arr) - 1, n_frames_new)).astype(int)
            stretched_labels[head] = arr[indices]
        else:
            # Binary labels [n_frames, 12] - use nearest-neighbor interpolation
            indices = np.round(np.linspace(0, arr.shape[0] - 1, n_frames_new)).astype(int)
            stretched_labels[head] = arr[indices]

    return stretched_features.astype(np.float32), stretched_labels


class MIREXChordFormerDataset(Dataset):
    """
    Dataset class for MIREX 2025 degree-based chord recognition.

    Loads pre-computed CQT features (252 bins) and MIREX-format labels:
    - Categorical: key, degree, bass
    - Binary: pitches_abs, intervals_root, intervals_bass

    Includes data augmentation per MIREX paper Section 3.2:
    - Pitch shift (-5 to +6 semitones)
    - Time-stretching (0.8x to 1.2x, 50% probability)
    - Gaussian noise (50% probability)
    """

    def __init__(self, features_dir, song_ids, normalization_path=None,
                 augment=False, sequence_length=None,
                 noise_prob=0.5, time_stretch_prob=0.5,
                 time_stretch_range=(0.8, 1.2)):
        """
        Args:
            features_dir: Path to features directory
            song_ids: List of song IDs to include
            normalization_path: Path to normalization.json (optional)
            augment: Whether to apply data augmentation (for training only)
            sequence_length: If set, extract random windows of this length
            noise_prob: Probability of applying Gaussian noise (default 0.5 per paper)
            time_stretch_prob: Probability of time-stretching (default 0.5 per paper)
            time_stretch_range: Time stretch range (default 0.8-1.2 per paper)
        """
        self.features_dir = Path(features_dir)
        self.augment = augment
        self.sequence_length = sequence_length
        self.noise_prob = noise_prob
        self.time_stretch_prob = time_stretch_prob
        self.time_stretch_range = time_stretch_range

        # Filter song_ids to only include those with MIREX labels
        # This handles the case where features were extracted without --mirex flag
        valid_song_ids = []
        skipped = 0
        for song_id in song_ids:
            song_dir = self.features_dir / f"{song_id:04d}"
            mirex_labels_path = song_dir / 'labels_mirex.npz'
            mirex_key_path = song_dir / 'labels_mirex_key.npy'
            
            # Check if MIREX labels exist
            if mirex_labels_path.exists() or mirex_key_path.exists():
                valid_song_ids.append(song_id)
            else:
                skipped += 1
        
        if skipped > 0:
            print(f"Warning: Skipped {skipped} songs without MIREX labels")
            print(f"  (Run feature extraction with --mirex flag to generate MIREX labels)")
        
        self.song_ids = valid_song_ids

        # Load normalization statistics
        self.mean = None
        self.std = None
        if normalization_path and Path(normalization_path).exists():
            with open(normalization_path, 'r') as f:
                norm_data = json.load(f)
                self.mean = np.array(norm_data['mean'], dtype=np.float32)
                self.std = np.array(norm_data['std'], dtype=np.float32)

    def __len__(self):
        return len(self.song_ids)

    def __getitem__(self, idx):
        """
        Returns:
            features: CQT features, shape [n_frames, 252]
            labels: Dict with:
                - Categorical: key, degree, bass [n_frames]
                - Binary: pitches_abs, intervals_root, intervals_bass [n_frames, 12]
            length: Sequence length (number of frames)
        """
        song_id = self.song_ids[idx]
        song_dir = self.features_dir / f"{song_id:04d}"

        # Load features
        features = np.load(song_dir / 'features.npy')

        # Load MIREX labels
        labels = {}

        # Try to load MIREX-format labels first
        mirex_labels_path = song_dir / 'labels_mirex.npz'
        if mirex_labels_path.exists():
            mirex_data = np.load(mirex_labels_path)
            for head in MIREX_HEADS:
                labels[head] = mirex_data[head]
        else:
            # Fallback: try to load individual label files
            for head in MIREX_HEADS_CATEGORICAL:
                label_path = song_dir / f'labels_mirex_{head}.npy'
                if label_path.exists():
                    labels[head] = np.load(label_path)
                else:
                    # Create zero labels if not found
                    labels[head] = np.zeros(len(features), dtype=np.int64)

            for head in MIREX_HEADS_BINARY:
                label_path = song_dir / f'labels_mirex_{head}.npy'
                if label_path.exists():
                    labels[head] = np.load(label_path)
                else:
                    # Create zero labels if not found
                    labels[head] = np.zeros((len(features), 12), dtype=np.int64)

        # Extract random window during training
        if self.sequence_length and len(features) > self.sequence_length:
            start_idx = random.randint(0, len(features) - self.sequence_length)
            end_idx = start_idx + self.sequence_length
            features = features[start_idx:end_idx]
            for head in MIREX_HEADS:
                labels[head] = labels[head][start_idx:end_idx]

        # Apply data augmentation (training only)
        if self.augment:
            features, labels = self._apply_augmentation(features, labels)

        # Apply z-score normalization (after augmentation)
        if self.mean is not None and self.std is not None:
            features = (features - self.mean) / self.std

        # Convert to tensors
        features = torch.from_numpy(features).float()
        labels_tensors = {}
        for head in MIREX_HEADS_CATEGORICAL:
            labels_tensors[head] = torch.from_numpy(labels[head]).long()
        for head in MIREX_HEADS_BINARY:
            labels_tensors[head] = torch.from_numpy(labels[head]).float()

        length = features.shape[0]

        return features, labels_tensors, length

    def _apply_augmentation(self, features, labels):
        """
        Apply data augmentation per MIREX paper Section 3.2.

        Augmentations:
        1. Pitch shift (-5 to +6 semitones) - 90% probability
        2. Time-stretching (0.8x to 1.2x) - 50% probability per paper
        3. Gaussian noise - 50% probability per paper
        """
        features = features.copy()
        labels = {head: arr.copy() for head, arr in labels.items()}
        n_bins = features.shape[1]
        bins_per_octave = 36  # ChordFormer uses 36 bins/octave

        # 1. Pitch shift (-5 to +6 semitones)
        if random.random() < 0.9:
            semitone_shift = random.randint(-5, 6)
            bin_shift = semitone_shift * (bins_per_octave // 12)  # 3 bins per semitone

            if bin_shift != 0:
                # Roll features along frequency axis
                features = np.roll(features, bin_shift, axis=1)
                # Zero out wrapped bins
                if bin_shift > 0:
                    features[:, :bin_shift] = 0
                else:
                    features[:, bin_shift:] = 0

                # Shift pitch-based labels
                labels = self._shift_mirex_labels(labels, semitone_shift)

        # 2. Time-stretching (0.8x to 1.2x) - per paper Section 3.2
        if random.random() < self.time_stretch_prob:
            stretch_rate = random.uniform(*self.time_stretch_range)
            features, labels = apply_time_stretch(features, labels, stretch_rate)

        # 3. Gaussian noise - 50% probability per paper Section 3.2
        if random.random() < self.noise_prob:
            noise_std = 0.1
            noise = np.random.normal(0, noise_std, features.shape).astype(np.float32)
            features = features + noise

        return features, labels

    def _shift_mirex_labels(self, labels, semitone_shift):
        """
        Shift MIREX labels to match pitch-shifted features.

        For degree-based representation when pitch shifting:
        - key: MUST shift (if C major shifted +2, now D major)
        - degree: stays the same (chord's relationship to key is preserved)
        - bass: MUST shift (absolute pitch class)
        - pitches_abs: MUST rotate (absolute pitch classes)
        - intervals_root/bass: stay the same (relative intervals don't change)

        Example: G chord (V) in C major, shift +2 semitones:
        - key: C (idx 1) -> D (idx 3)
        - degree: V (idx 11) -> V (idx 11) [unchanged]
        - bass: G (idx 8) -> A (idx 10)
        - pitches_abs: [G,B,D] -> [A,C#,E] (rotated)
        """
        shifted = {head: arr.copy() for head, arr in labels.items()}

        # Shift key (key: 0=N, 1-12 = C through B)
        key = labels['key']
        key_mask = key > 0
        if key_mask.any():
            valid_keys = key[key_mask]
            # Shift within 1-12 range (pitch classes)
            new_keys = (valid_keys - 1 + semitone_shift) % 12 + 1
            shifted['key'][key_mask] = new_keys

        # Shift bass notes (bass: 0=N, 1-12 = pitch classes)
        bass = labels['bass']
        bass_mask = bass > 0
        if bass_mask.any():
            valid_bass = bass[bass_mask]
            new_bass = (valid_bass - 1 + semitone_shift) % 12 + 1
            shifted['bass'][bass_mask] = new_bass

        # Rotate absolute pitch vectors (these are absolute pitch classes)
        shifted['pitches_abs'] = np.roll(labels['pitches_abs'], semitone_shift, axis=1)

        # intervals_root and intervals_bass are RELATIVE intervals, don't shift
        # (the interval from root to other notes doesn't change when transposing)

        return shifted


def collate_fn_mirex(batch):
    """
    Custom collate function for MIREX dataset with mixed categorical/binary labels.

    Args:
        batch: List of (features, labels_dict, length) tuples

    Returns:
        padded_features: Tensor of shape [batch, max_len, 252]
        padded_labels: Dict of tensors
            - Categorical (key, degree, bass): [batch, max_len]
            - Binary (pitches_abs, intervals_root, intervals_bass): [batch, max_len, 12]
        lengths: Tensor of sequence lengths [batch]
    """
    # Sort batch by length (descending)
    batch = sorted(batch, key=lambda x: x[2], reverse=True)

    features_list = [item[0] for item in batch]
    labels_list = [item[1] for item in batch]
    lengths = [item[2] for item in batch]

    # Get max length and feature dimension
    max_len = max(lengths)
    n_bins = features_list[0].shape[1]

    # Pad features
    batch_size = len(batch)
    padded_features = torch.zeros(batch_size, max_len, n_bins, dtype=torch.float32)

    # Initialize padded labels
    padded_labels = {}
    for head in MIREX_HEADS_CATEGORICAL:
        padded_labels[head] = torch.full((batch_size, max_len), -1, dtype=torch.long)
    for head in MIREX_HEADS_BINARY:
        padded_labels[head] = torch.zeros((batch_size, max_len, 12), dtype=torch.float32)

    for i, (features, labels, length) in enumerate(zip(features_list, labels_list, lengths)):
        padded_features[i, :length, :] = features
        for head in MIREX_HEADS_CATEGORICAL:
            padded_labels[head][i, :length] = labels[head]
        for head in MIREX_HEADS_BINARY:
            padded_labels[head][i, :length, :] = labels[head]

    lengths = torch.tensor(lengths, dtype=torch.long)

    return padded_features, padded_labels, lengths


def collate_fn(batch):
    """
    Custom collate function for variable-length sequences.
    Pads sequences to max length in batch and sorts by length (descending).

    Args:
        batch: List of (features, labels, length) tuples

    Returns:
        padded_features: Tensor of shape [batch, max_len, 84]
        padded_labels: Tensor of shape [batch, max_len]
        lengths: Tensor of sequence lengths [batch]
        sorted_indices: Original indices (for unsorting if needed)
    """
    # Sort batch by length (descending) for pack_padded_sequence
    batch = sorted(batch, key=lambda x: x[2], reverse=True)

    features_list = [item[0] for item in batch]
    labels_list = [item[1] for item in batch]
    lengths = [item[2] for item in batch]

    # Get max length
    max_len = max(lengths)

    # Pad features and labels
    batch_size = len(batch)
    padded_features = torch.zeros(batch_size, max_len, 84, dtype=torch.float32)
    padded_labels = torch.full((batch_size, max_len), -1, dtype=torch.long)  # -1 for padding

    for i, (features, labels, length) in enumerate(zip(features_list, labels_list, lengths)):
        padded_features[i, :length, :] = features
        padded_labels[i, :length] = labels

    lengths = torch.tensor(lengths, dtype=torch.long)

    return padded_features, padded_labels, lengths


def collate_fn_chordformer(batch):
    """
    Custom collate function for ChordFormer with multi-head labels.
    Pads sequences to max length in batch.

    Args:
        batch: List of (features, labels_dict, length) tuples

    Returns:
        padded_features: Tensor of shape [batch, max_len, 252]
        padded_labels: Dict of tensors, each shape [batch, max_len]
        lengths: Tensor of sequence lengths [batch]
    """
    # Sort batch by length (descending) for potential pack_padded_sequence use
    batch = sorted(batch, key=lambda x: x[2], reverse=True)

    features_list = [item[0] for item in batch]
    labels_list = [item[1] for item in batch]
    lengths = [item[2] for item in batch]

    # Get max length and feature dimension
    max_len = max(lengths)
    n_bins = features_list[0].shape[1]  # 252 for ChordFormer

    # Pad features
    batch_size = len(batch)
    padded_features = torch.zeros(batch_size, max_len, n_bins, dtype=torch.float32)

    # Initialize padded labels for each head
    padded_labels = {
        head: torch.full((batch_size, max_len), -1, dtype=torch.long)
        for head in CHORDFORMER_HEADS
    }

    for i, (features, labels, length) in enumerate(zip(features_list, labels_list, lengths)):
        padded_features[i, :length, :] = features
        for head in CHORDFORMER_HEADS:
            padded_labels[head][i, :length] = labels[head]

    lengths = torch.tensor(lengths, dtype=torch.long)

    return padded_features, padded_labels, lengths


def load_data_splits(features_dir):
    """
    Load train/val/test splits from data_split.json.

    Args:
        features_dir: Path to features directory

    Returns:
        train_ids, val_ids, test_ids: Lists of song IDs
    """
    split_path = Path(features_dir) / 'data_split.json'
    with open(split_path, 'r') as f:
        splits = json.load(f)

    return splits['train'], splits['val'], splits['test']


def get_dataloaders(features_dir, batch_size=8, num_workers=0):
    """
    Create train, validation, and test dataloaders.

    Args:
        features_dir: Path to features directory
        batch_size: Batch size
        num_workers: Number of workers for data loading

    Returns:
        train_loader, val_loader, test_loader: DataLoaders
        num_classes: Number of chord classes
    """
    from torch.utils.data import DataLoader

    # Load data splits
    train_ids, val_ids, test_ids = load_data_splits(features_dir)

    # Load chord vocabulary to get num_classes
    chord_to_idx_path = Path(features_dir) / 'chord_to_idx.json'
    with open(chord_to_idx_path, 'r') as f:
        chord_to_idx = json.load(f)
    num_classes = len(chord_to_idx)

    # Normalization path
    norm_path = Path(features_dir) / 'normalization.json'

    # Create datasets (augmentation disabled until we confirm baseline works)
    train_dataset = ChordDataset(
        features_dir, train_ids, norm_path,
        augment=False, num_classes=num_classes
    )
    val_dataset = ChordDataset(
        features_dir, val_ids, norm_path,
        augment=False, num_classes=num_classes
    )
    test_dataset = ChordDataset(
        features_dir, test_ids, norm_path,
        augment=False, num_classes=num_classes
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Dataset loaded: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")
    print(f"Number of classes: {num_classes}")

    return train_loader, val_loader, test_loader, num_classes


def get_chordformer_dataloaders(features_dir, batch_size=16, num_workers=0,
                                 sequence_length=1000, augment_train=True):
    """
    Create train, validation, and test dataloaders for ChordFormer.

    Args:
        features_dir: Path to features directory
        batch_size: Batch size
        num_workers: Number of workers for data loading
        sequence_length: Length of training sequences (1000 frames = ~23s)
        augment_train: Whether to apply augmentation to training data

    Returns:
        train_loader, val_loader, test_loader: DataLoaders
        head_sizes: Dict of output sizes for each head
    """
    from torch.utils.data import DataLoader

    # Load data splits
    train_ids, val_ids, test_ids = load_data_splits(features_dir)

    # Normalization path
    norm_path = Path(features_dir) / 'normalization.json'

    # Create datasets
    # Note: Using sequence_length for val/test too due to memory constraints
    # with relative positional encoding (O(n^2) memory for attention)
    train_dataset = ChordFormerDataset(
        features_dir, train_ids, norm_path,
        augment=augment_train, sequence_length=sequence_length
    )
    val_dataset = ChordFormerDataset(
        features_dir, val_ids, norm_path,
        augment=False, sequence_length=sequence_length  # Same length for memory
    )
    test_dataset = ChordFormerDataset(
        features_dir, test_ids, norm_path,
        augment=False, sequence_length=sequence_length  # Same length for memory
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_chordformer,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_chordformer,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_chordformer,
        num_workers=num_workers,
        pin_memory=True
    )

    # ChordFormer head output sizes
    head_sizes = {
        'root_triad': 85,  # N + 12 roots × 7 triads
        'bass': 13,        # N + 12 bass notes
        '7th': 4,          # N, 7, b7, bb7
        '9th': 4,          # N, 9, #9, b9
        '11th': 3,         # N, 11, #11
        '13th': 3          # N, 13, b13
    }

    print(f"ChordFormer Dataset loaded: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")
    print(f"Training sequence length: {sequence_length} frames")
    print(f"Output heads: {list(head_sizes.keys())}")

    return train_loader, val_loader, test_loader, head_sizes


def get_mirex_dataloaders(features_dir, batch_size=16, num_workers=0,
                          sequence_length=1000, augment_train=True,
                          noise_prob=0.5, time_stretch_prob=0.5):
    """
    Create train, validation, and test dataloaders for MIREX model.

    Args:
        features_dir: Path to features directory
        batch_size: Batch size
        num_workers: Number of workers for data loading
        sequence_length: Length of training sequences (1000 frames = ~23s)
        augment_train: Whether to apply augmentation to training data
        noise_prob: Noise augmentation probability (default 0.5 per paper)
        time_stretch_prob: Time-stretch augmentation probability (default 0.5 per paper)

    Returns:
        train_loader, val_loader, test_loader: DataLoaders
        head_sizes: Dict of output sizes for each head
    """
    from torch.utils.data import DataLoader

    features_dir = Path(features_dir)
    
    # Check metadata to verify features were extracted with MIREX mode
    metadata_path = features_dir / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        if not metadata.get('mirex_mode', False):
            print("WARNING: Features were NOT extracted with --mirex flag!")
            print("  Key confidence filtering may not have been applied.")
            print("  Re-run feature extraction with --mirex flag for best results.")
            print()

    # Load data splits
    train_ids, val_ids, test_ids = load_data_splits(features_dir)

    # Normalization path
    norm_path = Path(features_dir) / 'normalization.json'

    # Create datasets
    train_dataset = MIREXChordFormerDataset(
        features_dir, train_ids, norm_path,
        augment=augment_train, sequence_length=sequence_length,
        noise_prob=noise_prob, time_stretch_prob=time_stretch_prob
    )
    val_dataset = MIREXChordFormerDataset(
        features_dir, val_ids, norm_path,
        augment=False, sequence_length=sequence_length
    )
    test_dataset = MIREXChordFormerDataset(
        features_dir, test_ids, norm_path,
        augment=False, sequence_length=sequence_length
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_mirex,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_mirex,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_mirex,
        num_workers=num_workers,
        pin_memory=True
    )

    # MIREX head output sizes
    head_sizes = {
        'key': 13,          # N + 12 keys
        'degree': 18,       # N + 17 degrees with enharmonic distinction
        'bass': 13,         # N + 12 bass notes
        'pitches_abs': 12,  # 12 pitch classes
        'intervals_root': 12,
        'intervals_bass': 12
    }

    print(f"MIREX Dataset loaded: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")
    print(f"Training sequence length: {sequence_length} frames")
    print(f"Augmentation: noise={noise_prob}, time_stretch={time_stretch_prob}")
    print(f"Output heads: {list(head_sizes.keys())}")

    return train_loader, val_loader, test_loader, head_sizes


if __name__ == '__main__':
    # Test the dataset
    features_dir = '../features'

    print("Testing dataset loading...")
    train_loader, val_loader, test_loader, num_classes = get_dataloaders(
        features_dir,
        batch_size=4
    )

    # Test a batch
    for features, labels, lengths in train_loader:
        print(f"Features shape: {features.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Lengths: {lengths}")
        print(f"Number of classes: {num_classes}")
        break

    print("Dataset test passed!")
