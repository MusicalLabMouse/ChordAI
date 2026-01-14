"""
MIREX Model Evaluation Script

Evaluates a trained MIREX model on test datasets using WCSR metrics
(Weighted Chord Symbol Recall) as used in the MIREX 2025 paper Table 1.

Usage:
    python evaluate_mirex.py --checkpoint best_model.pth --test_dir /path/to/test/audio
    python evaluate_mirex.py --checkpoint best_model.pth --test_dir /path/to/isophonics
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from inference import (
    load_model,
    extract_features,
    predict_chords
)
from evaluate import (
    compute_mirex_wcsr_metrics,
    print_evaluation_results
)
import config


def parse_lab_file(lab_path):
    """
    Parse a .lab chord annotation file.

    Args:
        lab_path: Path to .lab file

    Returns:
        intervals: numpy array [[start, end], ...]
        labels: list of chord labels
    """
    intervals = []
    labels = []

    with open(lab_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) >= 3:
                try:
                    start = float(parts[0])
                    end = float(parts[1])
                    chord = parts[2]
                    intervals.append([start, end])
                    labels.append(chord)
                except ValueError:
                    continue

    return np.array(intervals), labels


def annotations_to_mir_eval_format(annotations):
    """
    Convert (start, end, chord) annotations to mir_eval format.

    Args:
        annotations: List of (start, end, chord) tuples

    Returns:
        intervals: numpy array [[start, end], ...]
        labels: list of chord labels
    """
    if not annotations:
        return np.array([]).reshape(0, 2), []

    intervals = np.array([[start, end] for start, end, _ in annotations])
    labels = [chord for _, _, chord in annotations]

    return intervals, labels


def find_test_songs(test_dir):
    """
    Find all test songs (audio files with matching .lab files).

    Supports multiple directory structures:
    - Flat: test_dir/song.mp3, test_dir/song.lab
    - Nested: test_dir/song/audio.mp3, test_dir/song/ground_truth.lab
    - Isophonics: test_dir/artist/album/song.mp3 with separate annotations

    Args:
        test_dir: Path to test directory

    Returns:
        List of (audio_path, lab_path) tuples
    """
    test_dir = Path(test_dir)
    songs = []

    # Common audio extensions
    audio_exts = {'.mp3', '.wav', '.flac', '.m4a', '.ogg'}

    # Find all audio files
    audio_files = []
    for ext in audio_exts:
        audio_files.extend(test_dir.rglob(f'*{ext}'))

    for audio_path in audio_files:
        # Try to find matching .lab file
        lab_candidates = [
            audio_path.with_suffix('.lab'),  # same name
            audio_path.parent / 'ground_truth.lab',  # nested structure
            audio_path.parent / f'{audio_path.stem}.lab',
        ]

        for lab_path in lab_candidates:
            if lab_path.exists():
                songs.append((audio_path, lab_path))
                break

    return songs


def evaluate_mirex_model(
    checkpoint_path,
    test_dir,
    normalization_path,
    device='cuda',
    use_crf=True,
    transition_penalty=1.0
):
    """
    Evaluate MIREX model on a test dataset.

    Args:
        checkpoint_path: Path to model checkpoint
        test_dir: Path to test dataset directory
        normalization_path: Path to normalization.json
        device: Device for inference
        use_crf: Whether to use CRF decoding
        transition_penalty: CRF transition penalty

    Returns:
        Dict of averaged WCSR metrics
    """
    # Load model
    print(f"Loading model from {checkpoint_path}")
    model, checkpoint = load_model(checkpoint_path, device)
    model_type = checkpoint.get('model_type', 'bilstm')
    print(f"  Model type: {model_type}")

    if model_type != 'mirex':
        print(f"Warning: Model type is '{model_type}', not 'mirex'. Results may vary.")

    # Load normalization stats
    with open(normalization_path, 'r') as f:
        normalization = json.load(f)

    # Find test songs
    songs = find_test_songs(test_dir)
    print(f"Found {len(songs)} test songs")

    if not songs:
        print("No test songs found! Check that test_dir contains audio files with .lab annotations.")
        return {}

    # Evaluate each song
    all_metrics = {
        'root': [], 'majmin': [], 'majmin_inv': [],
        'seventh': [], 'seventh_inv': [], 'mirex': [],
        'thirds': [], 'triads': []
    }

    for audio_path, lab_path in tqdm(songs, desc="Evaluating"):
        try:
            # Extract features
            features = extract_features(audio_path, model_type)

            # Run inference
            annotations, detected_key = predict_chords(
                model, features, normalization, model_type, device,
                use_crf=use_crf, transition_penalty=transition_penalty
            )

            # Convert predictions to mir_eval format
            est_intervals, est_labels = annotations_to_mir_eval_format(annotations)

            # Load ground truth
            ref_intervals, ref_labels = parse_lab_file(lab_path)

            if len(ref_intervals) == 0 or len(est_intervals) == 0:
                continue

            # Compute metrics for this song
            song_metrics = compute_mirex_wcsr_metrics(
                ref_intervals, ref_labels,
                est_intervals, est_labels
            )

            # Accumulate
            for metric_name in all_metrics:
                if song_metrics.get(metric_name, 0) > 0:
                    all_metrics[metric_name].append(song_metrics[metric_name])

        except Exception as e:
            print(f"Error processing {audio_path.name}: {e}")
            continue

    # Compute averages
    averaged_metrics = {}
    for metric_name, values in all_metrics.items():
        if values:
            averaged_metrics[metric_name] = np.mean(values)
        else:
            averaged_metrics[metric_name] = 0.0

    return averaged_metrics


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate MIREX model using WCSR metrics (Table 1 from paper)'
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--test_dir', type=str, required=True,
                        help='Path to test dataset directory (audio + .lab files)')
    parser.add_argument('--normalization', type=str,
                        default='../features/normalization.json',
                        help='Path to normalization stats')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device for inference')
    parser.add_argument('--no_crf', action='store_true',
                        help='Disable CRF decoding')
    parser.add_argument('--transition_penalty', type=float, default=1.0,
                        help='CRF transition penalty')
    parser.add_argument('--output', type=str, default=None,
                        help='Save results to JSON file')

    args = parser.parse_args()

    # Check paths
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return

    if not Path(args.test_dir).exists():
        print(f"Error: Test directory not found: {args.test_dir}")
        return

    if not Path(args.normalization).exists():
        print(f"Error: Normalization file not found: {args.normalization}")
        return

    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    # Run evaluation
    metrics = evaluate_mirex_model(
        checkpoint_path=args.checkpoint,
        test_dir=args.test_dir,
        normalization_path=args.normalization,
        device=args.device,
        use_crf=not args.no_crf,
        transition_penalty=args.transition_penalty
    )

    if metrics:
        # Print results
        print_evaluation_results(metrics, f"MIREX Evaluation: {Path(args.test_dir).name}")

        # Compare with paper results
        print("\nComparison with MIREX 2025 Paper (Table 1):")
        print("-" * 50)
        paper_isophonics = {
            'root': 0.8234, 'majmin': 0.8139, 'majmin_inv': 0.7875,
            'seventh': 0.6645, 'seventh_inv': 0.6470
        }
        for metric in ['root', 'majmin', 'majmin_inv', 'seventh', 'seventh_inv']:
            yours = metrics.get(metric, 0) * 100
            paper = paper_isophonics.get(metric, 0) * 100
            diff = yours - paper
            sign = '+' if diff >= 0 else ''
            print(f"  {metric:<15} Yours: {yours:5.1f}%  Paper: {paper:5.1f}%  ({sign}{diff:.1f}%)")

        # Save to JSON if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"\nResults saved to {args.output}")
    else:
        print("No results computed. Check your test data.")


if __name__ == '__main__':
    main()
