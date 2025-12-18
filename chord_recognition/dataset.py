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
    train_dataset = ChordFormerDataset(
        features_dir, train_ids, norm_path,
        augment=augment_train, sequence_length=sequence_length
    )
    val_dataset = ChordFormerDataset(
        features_dir, val_ids, norm_path,
        augment=False, sequence_length=None  # Full sequences for validation
    )
    test_dataset = ChordFormerDataset(
        features_dir, test_ids, norm_path,
        augment=False, sequence_length=None  # Full sequences for testing
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
