"""
Inference Script for Chord Recognition
Predicts chords from audio files with temporal smoothing.
"""

import argparse
import json
import numpy as np
import torch
import librosa
from scipy.ndimage import median_filter
from pathlib import Path

from model import ChordRecognitionModel


# Audio processing parameters (must match training)
SAMPLE_RATE = 22050
HOP_LENGTH = 512
N_BINS = 84
BINS_PER_OCTAVE = 12
FMIN = librosa.note_to_hz('C1')


def extract_cqt_features(audio_path, normalization=None):
    """
    Extract CQT chromagram features from audio file.

    Args:
        audio_path: Path to audio file
        normalization: Dictionary with 'mean' and 'std' for normalization

    Returns:
        features: CQT features, shape [n_frames, 84]
    """
    # Load audio file
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

    # Compute CQT
    CQT = librosa.cqt(
        y,
        sr=sr,
        hop_length=HOP_LENGTH,
        n_bins=N_BINS,
        bins_per_octave=BINS_PER_OCTAVE,
        fmin=FMIN
    )

    # Apply log scaling with numerical stability
    features = np.log(np.abs(CQT) + 1e-6)

    # Transpose to shape [n_frames, 84]
    features = features.T.astype(np.float32)

    # Apply z-score normalization
    if normalization is not None:
        mean = np.array(normalization['mean'], dtype=np.float32)
        std = np.array(normalization['std'], dtype=np.float32)
        features = (features - mean) / std

    return features


def predict_chords(model, features, device):
    """
    Predict chord labels from features.

    Args:
        model: Trained ChordRecognitionModel
        features: CQT features [n_frames, 84]
        device: Device (cuda/cpu)

    Returns:
        predictions: Predicted chord indices [n_frames]
    """
    model.eval()

    with torch.no_grad():
        # Convert to tensor and add batch dimension
        features_tensor = torch.from_numpy(features).unsqueeze(0).to(device)  # [1, n_frames, 84]
        length = torch.tensor([features.shape[0]], dtype=torch.long)

        # Forward pass
        outputs = model(features_tensor, length)  # [1, n_frames, num_classes]

        # Get predictions
        predictions = outputs.argmax(dim=-1).squeeze(0).cpu().numpy()  # [n_frames]

    return predictions


def apply_temporal_smoothing(predictions, window_size=7):
    """
    Apply median filter for temporal smoothing.

    Args:
        predictions: Predicted chord indices [n_frames]
        window_size: Window size for median filter (odd number)

    Returns:
        smoothed_predictions: Smoothed chord indices [n_frames]
    """
    # Ensure window_size is odd
    if window_size % 2 == 0:
        window_size += 1

    smoothed = median_filter(predictions, size=window_size, mode='nearest')
    return smoothed


def frames_to_segments(predictions, idx_to_chord, sr=SAMPLE_RATE, hop_length=HOP_LENGTH):
    """
    Convert frame-level predictions to time-aligned segments.

    Args:
        predictions: Predicted chord indices [n_frames]
        idx_to_chord: Dictionary mapping indices to chord labels
        sr: Sample rate
        hop_length: Hop length

    Returns:
        segments: List of (start_time, end_time, chord_label) tuples
    """
    segments = []

    if len(predictions) == 0:
        return segments

    # Group consecutive frames with same chord
    current_chord = predictions[0]
    start_frame = 0

    for i in range(1, len(predictions)):
        if predictions[i] != current_chord:
            # Convert frames to time
            start_time = start_frame * hop_length / sr
            end_time = i * hop_length / sr
            chord_label = idx_to_chord.get(str(current_chord), 'N')

            segments.append((start_time, end_time, chord_label))

            # Update for next segment
            current_chord = predictions[i]
            start_frame = i

    # Add final segment
    start_time = start_frame * hop_length / sr
    end_time = len(predictions) * hop_length / sr
    chord_label = idx_to_chord.get(str(current_chord), 'N')
    segments.append((start_time, end_time, chord_label))

    return segments


def save_lab_file(segments, output_path):
    """
    Save segments to .lab file.

    Args:
        segments: List of (start_time, end_time, chord_label) tuples
        output_path: Path to output .lab file
    """
    with open(output_path, 'w') as f:
        for start_time, end_time, chord_label in segments:
            f.write(f"{start_time:.6f}\t{end_time:.6f}\t{chord_label}\n")


def main():
    parser = argparse.ArgumentParser(description='Predict chords from audio file')
    parser.add_argument('input_audio', type=str,
                        help='Path to input audio file (MP3 or WAV)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to output .lab file (default: input_audio.lab)')
    parser.add_argument('--checkpoint', type=str, default='../checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--features_dir', type=str, default='../features',
                        help='Path to features directory (for vocabulary and normalization)')
    parser.add_argument('--smoothing_window', type=int, default=7,
                        help='Window size for temporal smoothing (median filter)')
    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load chord vocabulary
    features_dir = Path(args.features_dir)
    idx_to_chord_path = features_dir / 'idx_to_chord.json'
    chord_to_idx_path = features_dir / 'chord_to_idx.json'

    with open(idx_to_chord_path, 'r') as f:
        idx_to_chord = json.load(f)

    with open(chord_to_idx_path, 'r') as f:
        chord_to_idx = json.load(f)

    num_classes = len(chord_to_idx)
    print(f"Loaded chord vocabulary: {num_classes} classes")

    # Load normalization statistics
    normalization = None
    norm_path = features_dir / 'normalization.json'
    if norm_path.exists():
        with open(norm_path, 'r') as f:
            normalization = json.load(f)
        print("Loaded normalization statistics")

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = ChordRecognitionModel(num_classes=num_classes, hidden_size=256, dropout=0.2)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    print(f"Model loaded (trained for {checkpoint['epoch']+1} epochs, val_acc: {checkpoint['val_acc']:.4f})")

    # Extract features
    print(f"Extracting features from {args.input_audio}...")
    features = extract_cqt_features(args.input_audio, normalization)
    print(f"Extracted {features.shape[0]} frames")

    # Predict chords
    print("Predicting chords...")
    predictions = predict_chords(model, features, device)

    # Apply temporal smoothing
    print(f"Applying temporal smoothing (window size: {args.smoothing_window})...")
    smoothed_predictions = apply_temporal_smoothing(predictions, window_size=args.smoothing_window)

    # Convert to segments
    print("Converting to time-aligned segments...")
    segments = frames_to_segments(smoothed_predictions, idx_to_chord)

    print(f"Detected {len(segments)} chord segments")

    # Determine output path
    if args.output is None:
        input_path = Path(args.input_audio)
        output_path = input_path.with_suffix('.lab')
    else:
        output_path = Path(args.output)

    # Save output
    save_lab_file(segments, output_path)
    print(f"Saved predictions to {output_path}")

    # Print first few segments
    print("\nFirst 10 segments:")
    for i, (start, end, chord) in enumerate(segments[:10]):
        print(f"{start:.3f}\t{end:.3f}\t{chord}")

    print("\nPrediction complete!")


if __name__ == '__main__':
    main()
