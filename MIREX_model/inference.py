"""
Inference script for MIREX Chord Estimation Model
Estimates chords from audio files using the trained model.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import librosa

import config
from model import ScaleDegreeChordModel


# Note names for output
NOTE_NAMES = ['N', 'C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
DEGREE_NAMES = ['N', 'I', 'bII', 'II', 'bIII', 'III', 'IV', '#IV', 'V', 'bVI', 'VI', 'bVII', 'VII']


def load_model(checkpoint_path, device='cpu'):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_config = checkpoint.get('config', {})

    model = ScaleDegreeChordModel(
        n_bins=model_config.get('n_bins', config.N_BINS),
        bins_per_octave=model_config.get('bins_per_octave', config.BINS_PER_OCTAVE),
        d_model=model_config.get('d_model', config.CONFORMER_DIM),
        n_heads=model_config.get('n_heads', config.CONFORMER_HEADS),
        d_ff=model_config.get('d_ff', config.CONFORMER_FF_DIM),
        n_layers=model_config.get('n_layers', config.CONFORMER_LAYERS),
        conv_kernel_size=config.CONFORMER_CONV_KERNEL,
        dropout=0.0,
        num_keys=config.NUM_KEYS,
        num_degrees=config.NUM_SCALE_DEGREES,
        num_bass=config.NUM_BASS_NOTES
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)

    return model


def extract_features(audio_path):
    """Extract CQT features from audio file."""
    y, sr = librosa.load(audio_path, sr=config.SAMPLE_RATE, mono=True)

    CQT = librosa.cqt(
        y,
        sr=sr,
        hop_length=config.HOP_LENGTH,
        n_bins=config.N_BINS,
        bins_per_octave=config.BINS_PER_OCTAVE,
        fmin=librosa.note_to_hz(config.FMIN_NOTE)
    )

    CQT_db = librosa.amplitude_to_db(np.abs(CQT), ref=np.max)
    features = CQT_db.T.astype(np.float32)

    return features


def predict_chords(model, features, normalization, device='cpu'):
    """
    Predict chords from CQT features.

    Returns:
        List of (start_time, end_time, chord_label) tuples
    """
    # Normalize features
    mean = np.array(normalization['mean'])
    std = np.array(normalization['std'])
    features_norm = (features - mean) / std

    # Convert to tensor
    x = torch.from_numpy(features_norm).float().unsqueeze(0).to(device)

    # Get predictions
    with torch.no_grad():
        outputs = model(x)

    key_preds = outputs['key'].argmax(dim=-1).squeeze(0).cpu().numpy()
    degree_preds = outputs['degree'].argmax(dim=-1).squeeze(0).cpu().numpy()
    bass_preds = outputs['bass'].argmax(dim=-1).squeeze(0).cpu().numpy()

    # Get probabilities for confidence
    key_probs = torch.softmax(outputs['key'], dim=-1).squeeze(0).cpu().numpy()
    degree_probs = torch.softmax(outputs['degree'], dim=-1).squeeze(0).cpu().numpy()

    return key_preds, degree_preds, bass_preds, key_probs, degree_probs


def frames_to_annotations(key_preds, degree_preds, bass_preds, hop_length=config.HOP_LENGTH, sr=config.SAMPLE_RATE):
    """
    Convert frame-level predictions to time-aligned chord annotations.

    Returns:
        List of (start_time, end_time, chord_label) tuples
    """
    n_frames = len(key_preds)
    frame_duration = hop_length / sr

    annotations = []
    current_chord = None
    start_frame = 0

    for i in range(n_frames):
        # Compute chord label from key and degree
        key_idx = key_preds[i]
        degree_idx = degree_preds[i]
        bass_idx = bass_preds[i]

        if key_idx == 0 or degree_idx == 0:
            chord = 'N'
        else:
            # Compute root from key and degree
            root_idx = ((key_idx - 1) + (degree_idx - 1)) % 12 + 1
            root_name = NOTE_NAMES[root_idx]

            # For now, just output root (can be extended with quality detection)
            chord = root_name

            # Add bass if different from root
            if bass_idx != 0 and bass_idx != root_idx:
                bass_name = NOTE_NAMES[bass_idx]
                chord = f"{root_name}/{bass_name}"

        if chord != current_chord:
            if current_chord is not None:
                start_time = start_frame * frame_duration
                end_time = i * frame_duration
                annotations.append((start_time, end_time, current_chord))

            current_chord = chord
            start_frame = i

    # Add final chord
    if current_chord is not None:
        start_time = start_frame * frame_duration
        end_time = n_frames * frame_duration
        annotations.append((start_time, end_time, current_chord))

    return annotations


def write_lab_file(annotations, output_path):
    """Write annotations to .lab file format."""
    with open(output_path, 'w') as f:
        for start, end, chord in annotations:
            f.write(f"{start:.6f}\t{end:.6f}\t{chord}\n")


def main():
    parser = argparse.ArgumentParser(description='Estimate chords from audio file')
    parser.add_argument('audio_path', type=str, help='Path to audio file')
    parser.add_argument('--checkpoint', type=str,
                        default=str(Path(config.DEFAULT_CHECKPOINT_DIR) / 'best_model.pt'),
                        help='Path to model checkpoint')
    parser.add_argument('--normalization', type=str,
                        default=str(Path(config.DEFAULT_CHECKPOINT_DIR) / 'normalization.json'),
                        help='Path to normalization stats')
    parser.add_argument('--output', type=str, default=None,
                        help='Output .lab file path (default: same as input with .lab extension)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cpu or cuda)')
    args = parser.parse_args()

    audio_path = Path(args.audio_path)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        return

    output_path = args.output or audio_path.with_suffix('.lab')

    # Load normalization
    with open(args.normalization, 'r') as f:
        normalization = json.load(f)

    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, args.device)

    # Extract features
    print(f"Extracting features from {audio_path}")
    features = extract_features(audio_path)
    print(f"  {features.shape[0]} frames, {features.shape[1]} bins")

    # Predict chords
    print("Predicting chords...")
    key_preds, degree_preds, bass_preds, key_probs, degree_probs = predict_chords(
        model, features, normalization, args.device
    )

    # Get most common key
    non_n_keys = key_preds[key_preds > 0]
    if len(non_n_keys) > 0:
        from collections import Counter
        key_counts = Counter(non_n_keys)
        most_common_key = key_counts.most_common(1)[0][0]
        print(f"  Estimated key: {NOTE_NAMES[most_common_key]}")

    # Convert to annotations
    annotations = frames_to_annotations(key_preds, degree_preds, bass_preds)

    # Write output
    write_lab_file(annotations, output_path)
    print(f"\nChord annotations written to: {output_path}")
    print(f"  {len(annotations)} chord segments")

    # Print first few chords
    print("\nFirst 10 chords:")
    for start, end, chord in annotations[:10]:
        print(f"  {start:6.2f} - {end:6.2f}: {chord}")


if __name__ == '__main__':
    main()
