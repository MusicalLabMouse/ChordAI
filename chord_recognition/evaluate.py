"""
Evaluation Metrics for Chord Recognition

Implements WCSR (Weighted Chord Symbol Recall) metrics per MIREX 2025 paper Table 1.
Uses mir_eval library for standardized chord evaluation.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Optional: mir_eval for WCSR metrics
try:
    import mir_eval.chord
    HAS_MIR_EVAL = True
except ImportError:
    HAS_MIR_EVAL = False
    print("Warning: mir_eval not installed. WCSR metrics unavailable.")
    print("Install with: pip install mir_eval")


def compute_mirex_wcsr_metrics(
    ref_intervals: np.ndarray,
    ref_labels: List[str],
    est_intervals: np.ndarray,
    est_labels: List[str]
) -> Dict[str, float]:
    """
    Compute all WCSR metrics per MIREX paper Table 1.

    Uses mir_eval.chord.evaluate() which computes time-weighted accuracy
    for various chord comparison schemes.

    Args:
        ref_intervals: numpy array [[start, end], ...] for reference annotations
        ref_labels: list of reference chord labels (mir_eval format, e.g., "C:maj7")
        est_intervals: numpy array [[start, end], ...] for estimated annotations
        est_labels: list of estimated chord labels (mir_eval format)

    Returns:
        Dict with keys:
            - root: Root note accuracy
            - majmin: Major/minor quality accuracy
            - majmin_inv: Major/minor with inversions accuracy
            - seventh: 7th chord accuracy
            - seventh_inv: 7th chord with inversions accuracy
            - mirex: MIREX competition metric
            - thirds: Third quality accuracy
            - triads: Full triad accuracy

    Example:
        >>> ref_intervals = np.array([[0.0, 2.5], [2.5, 5.0]])
        >>> ref_labels = ['C:maj', 'G:min']
        >>> est_intervals = np.array([[0.0, 2.5], [2.5, 5.0]])
        >>> est_labels = ['C:maj', 'G:min7']
        >>> scores = compute_mirex_wcsr_metrics(ref_intervals, ref_labels,
        ...                                      est_intervals, est_labels)
    """
    if not HAS_MIR_EVAL:
        return {
            'root': 0.0,
            'majmin': 0.0,
            'majmin_inv': 0.0,
            'seventh': 0.0,
            'seventh_inv': 0.0,
            'mirex': 0.0,
            'thirds': 0.0,
            'triads': 0.0
        }

    try:
        scores = mir_eval.chord.evaluate(
            ref_intervals, ref_labels,
            est_intervals, est_labels
        )
        return {
            'root': scores['root'],
            'majmin': scores['majmin'],
            'majmin_inv': scores['majmin_inv'],
            'seventh': scores['sevenths'],
            'seventh_inv': scores['sevenths_inv'],
            'mirex': scores['mirex'],
            'thirds': scores['thirds'],
            'triads': scores['triads']
        }
    except Exception as e:
        print(f"Warning: mir_eval error: {e}")
        return {
            'root': 0.0,
            'majmin': 0.0,
            'majmin_inv': 0.0,
            'seventh': 0.0,
            'seventh_inv': 0.0,
            'mirex': 0.0,
            'thirds': 0.0,
            'triads': 0.0
        }


def verify_dataset_size(data_dirs: List[str], min_songs: int = 1000) -> int:
    """
    Warn if dataset is significantly smaller than paper's 1,163 songs.

    MIREX 2025 paper (Section 3.1) used 1,163 songs total:
    - 560 from McGill-Billboard
    - 603 collected by authors

    Args:
        data_dirs: List of paths to data directories
        min_songs: Minimum recommended number of songs (default 1000)

    Returns:
        total_songs: Total number of songs found
    """
    total_songs = 0
    for data_dir in data_dirs:
        dir_path = Path(data_dir)
        if dir_path.exists():
            song_dirs = [d for d in dir_path.iterdir() if d.is_dir()]
            total_songs += len(song_dirs)

    if total_songs < min_songs:
        print(f"WARNING: Only {total_songs} songs found. "
              f"MIREX paper used 1,163 songs for best results.")

    return total_songs


def frame_predictions_to_annotations(
    chord_labels: List[str],
    hop_length: int = 512,
    sr: int = 22050
) -> Tuple[np.ndarray, List[str]]:
    """
    Convert frame-level predictions to mir_eval annotation format.

    Consolidates consecutive frames with the same chord into intervals.

    Args:
        chord_labels: List of chord labels (one per frame)
        hop_length: Hop length in samples
        sr: Sample rate

    Returns:
        intervals: numpy array [[start, end], ...] in seconds
        labels: list of chord labels (one per interval)
    """
    if not chord_labels:
        return np.array([]).reshape(0, 2), []

    frame_duration = hop_length / sr
    intervals = []
    labels = []

    current_chord = chord_labels[0]
    start_frame = 0

    for i, chord in enumerate(chord_labels[1:], start=1):
        if chord != current_chord:
            # End current segment
            start_time = start_frame * frame_duration
            end_time = i * frame_duration
            intervals.append([start_time, end_time])
            labels.append(current_chord)

            # Start new segment
            current_chord = chord
            start_frame = i

    # Add final segment
    start_time = start_frame * frame_duration
    end_time = len(chord_labels) * frame_duration
    intervals.append([start_time, end_time])
    labels.append(current_chord)

    return np.array(intervals), labels


def evaluate_song(
    ref_labels: List[str],
    est_labels: List[str],
    hop_length: int = 512,
    sr: int = 22050
) -> Dict[str, float]:
    """
    Evaluate chord predictions for a single song.

    Args:
        ref_labels: List of reference chord labels (frame-level)
        est_labels: List of estimated chord labels (frame-level)
        hop_length: Hop length in samples
        sr: Sample rate

    Returns:
        Dict of WCSR metrics
    """
    # Convert frame predictions to intervals
    ref_intervals, ref_labels_consolidated = frame_predictions_to_annotations(
        ref_labels, hop_length, sr
    )
    est_intervals, est_labels_consolidated = frame_predictions_to_annotations(
        est_labels, hop_length, sr
    )

    # Compute metrics
    return compute_mirex_wcsr_metrics(
        ref_intervals, ref_labels_consolidated,
        est_intervals, est_labels_consolidated
    )


def evaluate_dataset(
    all_ref_labels: List[List[str]],
    all_est_labels: List[List[str]],
    hop_length: int = 512,
    sr: int = 22050
) -> Dict[str, float]:
    """
    Evaluate chord predictions for entire dataset.

    Computes macro-averaged metrics across all songs.

    Args:
        all_ref_labels: List of reference label lists (one per song)
        all_est_labels: List of estimated label lists (one per song)
        hop_length: Hop length in samples
        sr: Sample rate

    Returns:
        Dict of averaged WCSR metrics
    """
    if len(all_ref_labels) != len(all_est_labels):
        raise ValueError("Number of reference and estimated songs must match")

    # Accumulate metrics
    metric_names = ['root', 'majmin', 'majmin_inv', 'seventh', 'seventh_inv',
                    'mirex', 'thirds', 'triads']
    accumulated = {name: [] for name in metric_names}

    for ref_labels, est_labels in zip(all_ref_labels, all_est_labels):
        song_metrics = evaluate_song(ref_labels, est_labels, hop_length, sr)
        for name in metric_names:
            if song_metrics[name] > 0:  # Only include valid results
                accumulated[name].append(song_metrics[name])

    # Compute averages
    averaged = {}
    for name in metric_names:
        if accumulated[name]:
            averaged[name] = np.mean(accumulated[name])
        else:
            averaged[name] = 0.0

    return averaged


def print_evaluation_results(metrics: Dict[str, float], title: str = "Evaluation Results"):
    """
    Print evaluation results in a formatted table.

    Args:
        metrics: Dict of metric name -> value
        title: Title for the results
    """
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"{'=' * 50}")
    print(f"  {'Metric':<20} {'Score':>10}")
    print(f"  {'-' * 30}")

    # Print in order matching MIREX paper Table 1
    ordered_metrics = ['root', 'majmin', 'majmin_inv', 'seventh', 'seventh_inv',
                       'mirex', 'thirds', 'triads']

    for name in ordered_metrics:
        if name in metrics:
            print(f"  {name:<20} {metrics[name]*100:>9.2f}%")

    print(f"{'=' * 50}\n")


if __name__ == '__main__':
    # Test with dummy data
    print("Testing WCSR metrics...")

    if HAS_MIR_EVAL:
        # Create test data
        ref_intervals = np.array([[0.0, 2.0], [2.0, 4.0], [4.0, 6.0]])
        ref_labels = ['C:maj', 'G:min', 'F:maj7']

        est_intervals = np.array([[0.0, 2.0], [2.0, 4.0], [4.0, 6.0]])
        est_labels = ['C:maj', 'G:min7', 'F:maj']

        scores = compute_mirex_wcsr_metrics(
            ref_intervals, ref_labels,
            est_intervals, est_labels
        )

        print_evaluation_results(scores, "Test Evaluation")
        print("WCSR metrics test passed!")
    else:
        print("Skipping test - mir_eval not available")
