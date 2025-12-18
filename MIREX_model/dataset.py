"""
Dataset for MIREX Chord Estimation Model
Loads pre-extracted features and scale degree labels with pitch shift augmentation.
"""

import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

import config


class ChordDataset(Dataset):
    """
    Dataset for chord estimation with scale degree representation.

    Features pitch shift augmentation by rolling CQT bins and adjusting labels.
    """
    def __init__(
        self,
        features_dir,
        song_ids,
        sequence_length=None,
        augment=False,
        normalization=None
    ):
        """
        Args:
            features_dir: Path to extracted features directory
            song_ids: List of song IDs to include
            sequence_length: If set, extract random windows of this length
            augment: Whether to apply data augmentation
            normalization: Dict with 'mean' and 'std' arrays
        """
        self.features_dir = Path(features_dir)
        self.song_ids = song_ids
        self.sequence_length = sequence_length
        self.augment = augment
        self.normalization = normalization

        # Pre-load all data into memory for faster training
        self.data = []
        for song_id in song_ids:
            song_dir = self.features_dir / f"{song_id:04d}"
            if not song_dir.exists():
                continue

            try:
                features = np.load(song_dir / 'features.npy')
                key_labels = np.load(song_dir / 'key_labels.npy')
                degree_labels = np.load(song_dir / 'degree_labels.npy')
                bass_labels = np.load(song_dir / 'bass_labels.npy')
                pitch_abs = np.load(song_dir / 'pitch_abs_labels.npy')
                pitch_root = np.load(song_dir / 'pitch_root_labels.npy')
                pitch_bass = np.load(song_dir / 'pitch_bass_labels.npy')

                self.data.append({
                    'features': features,
                    'key': key_labels,
                    'degree': degree_labels,
                    'bass': bass_labels,
                    'pitch_abs': pitch_abs,
                    'pitch_root': pitch_root,
                    'pitch_bass': pitch_bass
                })
            except Exception as e:
                print(f"Warning: Could not load song {song_id}: {e}")
                continue

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        features = item['features'].copy()
        key = item['key'].copy()
        degree = item['degree'].copy()
        bass = item['bass'].copy()
        pitch_abs = item['pitch_abs'].copy()
        pitch_root = item['pitch_root'].copy()
        pitch_bass = item['pitch_bass'].copy()

        n_frames = features.shape[0]

        # Extract random window if sequence_length is set
        if self.sequence_length is not None and n_frames > self.sequence_length:
            start = random.randint(0, n_frames - self.sequence_length)
            end = start + self.sequence_length

            features = features[start:end]
            key = key[start:end]
            degree = degree[start:end]
            bass = bass[start:end]
            pitch_abs = pitch_abs[start:end]
            pitch_root = pitch_root[start:end]
            pitch_bass = pitch_bass[start:end]

        # Apply pitch shift augmentation
        if self.augment:
            features, key, degree, bass, pitch_abs, pitch_root, pitch_bass = \
                self._apply_pitch_shift(features, key, degree, bass, pitch_abs, pitch_root, pitch_bass)

        # Normalize features
        if self.normalization is not None:
            mean = np.array(self.normalization['mean'])
            std = np.array(self.normalization['std'])
            features = (features - mean) / std

        return {
            'features': torch.from_numpy(features).float(),
            'key': torch.from_numpy(key).long(),
            'degree': torch.from_numpy(degree).long(),
            'bass': torch.from_numpy(bass).long(),
            'pitch_abs': torch.from_numpy(pitch_abs).float(),
            'pitch_root': torch.from_numpy(pitch_root).float(),
            'pitch_bass': torch.from_numpy(pitch_bass).float()
        }

    def _apply_pitch_shift(self, features, key, degree, bass, pitch_abs, pitch_root, pitch_bass):
        """
        Apply pitch shift augmentation by rolling CQT bins and adjusting labels.

        Pitch shift range: -5 to +6 semitones (as per MIREX paper)
        """
        # Random pitch shift in semitones
        shift_semitones = random.randint(config.PITCH_SHIFT_RANGE[0], config.PITCH_SHIFT_RANGE[1])

        if shift_semitones == 0:
            return features, key, degree, bass, pitch_abs, pitch_root, pitch_bass

        # Shift features (roll CQT bins)
        # With 36 bins/octave, each semitone = 3 bins
        bins_per_semitone = config.BINS_PER_OCTAVE // 12
        shift_bins = shift_semitones * bins_per_semitone

        # Roll along frequency axis
        features = np.roll(features, shift_bins, axis=1)

        # Zero out wrapped bins (don't wrap around)
        if shift_bins > 0:
            features[:, :shift_bins] = features[:, shift_bins:shift_bins+1]  # Replicate edge
        elif shift_bins < 0:
            features[:, shift_bins:] = features[:, shift_bins-1:shift_bins]  # Replicate edge

        # Shift key labels (1-12 become shifted, 0 stays 0)
        key_shifted = np.where(key > 0, ((key - 1 + shift_semitones) % 12) + 1, 0)

        # Degree labels stay the same (they're relative to the key)

        # Shift bass labels (1-12 become shifted, 0 stays 0)
        bass_shifted = np.where(bass > 0, ((bass - 1 + shift_semitones) % 12) + 1, 0)

        # Shift absolute pitch presence (roll by semitones)
        pitch_abs_shifted = np.roll(pitch_abs, shift_semitones, axis=1)

        # Root and bass relative pitches stay the same (they're intervals)

        return features, key_shifted, degree, bass_shifted, pitch_abs_shifted, pitch_root, pitch_bass


def collate_fn(batch):
    """
    Collate function for variable-length sequences.
    Pads sequences to the maximum length in the batch.
    """
    # Find max length in batch
    max_len = max(item['features'].shape[0] for item in batch)

    batch_size = len(batch)
    n_bins = batch[0]['features'].shape[1]

    # Initialize padded tensors
    features = torch.zeros(batch_size, max_len, n_bins)
    key = torch.full((batch_size, max_len), -1, dtype=torch.long)  # -1 for ignore
    degree = torch.full((batch_size, max_len), -1, dtype=torch.long)
    bass = torch.full((batch_size, max_len), -1, dtype=torch.long)
    pitch_abs = torch.zeros(batch_size, max_len, 12)
    pitch_root = torch.zeros(batch_size, max_len, 12)
    pitch_bass = torch.zeros(batch_size, max_len, 12)
    lengths = torch.zeros(batch_size, dtype=torch.long)

    # Fill in data
    for i, item in enumerate(batch):
        seq_len = item['features'].shape[0]
        lengths[i] = seq_len

        features[i, :seq_len] = item['features']
        key[i, :seq_len] = item['key']
        degree[i, :seq_len] = item['degree']
        bass[i, :seq_len] = item['bass']
        pitch_abs[i, :seq_len] = item['pitch_abs']
        pitch_root[i, :seq_len] = item['pitch_root']
        pitch_bass[i, :seq_len] = item['pitch_bass']

    return {
        'features': features,
        'key': key,
        'degree': degree,
        'bass': bass,
        'pitch_abs': pitch_abs,
        'pitch_root': pitch_root,
        'pitch_bass': pitch_bass,
        'lengths': lengths
    }


def create_dataloaders(
    features_dir,
    batch_size=config.BATCH_SIZE,
    sequence_length=500,
    num_workers=4
):
    """
    Create train, validation, and test dataloaders.

    Returns:
        train_loader, val_loader, test_loader, normalization
    """
    features_dir = Path(features_dir)

    # Load data split
    with open(features_dir / 'data_split.json', 'r') as f:
        data_split = json.load(f)

    # Load normalization stats
    with open(features_dir / 'normalization.json', 'r') as f:
        normalization = json.load(f)

    train_ids = data_split['train']
    val_ids = data_split['val']
    test_ids = data_split['test']

    print(f"Dataset sizes: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")

    # Create datasets
    train_dataset = ChordDataset(
        features_dir,
        train_ids,
        sequence_length=sequence_length,
        augment=True,
        normalization=normalization
    )

    val_dataset = ChordDataset(
        features_dir,
        val_ids,
        sequence_length=sequence_length,
        augment=False,
        normalization=normalization
    )

    test_dataset = ChordDataset(
        features_dir,
        test_ids,
        sequence_length=None,  # Full sequences for testing
        augment=False,
        normalization=normalization
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # One song at a time for testing
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader, normalization


if __name__ == '__main__':
    # Test the dataset
    features_dir = config.DEFAULT_FEATURES_DIR

    if not Path(features_dir).exists():
        print(f"Features directory not found: {features_dir}")
        print("Run feature_extraction.py first.")
    else:
        train_loader, val_loader, test_loader, norm = create_dataloaders(features_dir)

        print(f"\nTrain batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")

        # Test one batch
        batch = next(iter(train_loader))
        print(f"\nBatch shapes:")
        print(f"  features: {batch['features'].shape}")
        print(f"  key: {batch['key'].shape}")
        print(f"  degree: {batch['degree'].shape}")
        print(f"  bass: {batch['bass'].shape}")
        print(f"  lengths: {batch['lengths']}")
