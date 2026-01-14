"""
Inference Script for Chord Recognition
Supports BiLSTM, TCN, and ChordFormer models with optional CRF decoding.
"""

import argparse
import json
import numpy as np
import torch
import librosa
import warnings
from pathlib import Path

from model import ChordRecognitionModel, ChordRecognitionModelTCN, ChordFormerModel, MIREXChordFormerModel
from dataset import CHORDFORMER_HEADS, MIREX_HEADS_CATEGORICAL, MIREX_HEADS_BINARY
import config


# ChordFormer chord reconstruction mappings
ROOT_NAMES = config.ROOT_NAMES
TRIAD_TYPES = config.TRIAD_TYPES
SEVENTH_TYPES = config.SEVENTH_TYPES
NINTH_TYPES = config.NINTH_TYPES
ELEVENTH_TYPES = config.ELEVENTH_TYPES
THIRTEENTH_TYPES = config.THIRTEENTH_TYPES


def load_model(checkpoint_path, device='cpu'):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_type = checkpoint.get('model_type', 'bilstm')
    num_classes = checkpoint.get('num_classes', 25)
    hidden_size = checkpoint.get('hidden_size', 256)

    if model_type == 'mirex':
        # MIREX degree-based model
        model_config = checkpoint.get('config', {})
        model = MIREXChordFormerModel(
            n_bins=model_config.get('n_bins', config.N_BINS_CHORDFORMER),
            d_model=model_config.get('d_model', config.CONFORMER_DIM),
            n_heads=model_config.get('n_heads', config.CONFORMER_HEADS),
            d_ff=model_config.get('d_ff', config.CONFORMER_FF_DIM),
            n_layers=model_config.get('n_layers', config.CONFORMER_LAYERS),
            conv_kernel_size=model_config.get('conv_kernel_size', config.CONFORMER_CONV_KERNEL),
            dropout=0.0,
            octavewise_n_filters=model_config.get('octavewise_n_filters', config.OCTAVEWISE_N_FILTERS),
            num_keys=model_config.get('num_keys', config.MIREX_NUM_KEYS),
            num_degrees=model_config.get('num_degrees', config.MIREX_NUM_DEGREES),
            num_bass=model_config.get('num_bass', config.MIREX_NUM_BASS),
            num_pitches=model_config.get('num_pitches', config.MIREX_NUM_PITCHES)
        )
    elif model_type == 'chordformer':
        model_config = checkpoint.get('config', {})
        model = ChordFormerModel(
            n_bins=model_config.get('n_bins', config.N_BINS_CHORDFORMER),
            d_model=model_config.get('d_model', config.CONFORMER_DIM),
            n_heads=model_config.get('n_heads', config.CONFORMER_HEADS),
            d_ff=model_config.get('d_ff', config.CONFORMER_FF_DIM),
            n_layers=model_config.get('n_layers', config.CONFORMER_LAYERS),
            conv_kernel_size=model_config.get('conv_kernel_size', config.CONFORMER_CONV_KERNEL),
            dropout=0.0
        )
    elif model_type == 'tcn':
        model = ChordRecognitionModelTCN(
            num_classes=num_classes,
            tcn_channels=hidden_size,
            dropout=0.0
        )
    else:
        model = ChordRecognitionModel(
            num_classes=num_classes,
            hidden_size=hidden_size,
            dropout=0.0
        )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)

    return model, checkpoint


def extract_features(audio_path, model_type='chordformer'):
    """Extract CQT features from audio file."""
    if model_type in ('chordformer', 'mirex'):
        n_bins = config.N_BINS_CHORDFORMER
        bins_per_octave = config.BINS_PER_OCTAVE_CHORDFORMER
    else:
        n_bins = config.N_BINS_LEGACY
        bins_per_octave = config.BINS_PER_OCTAVE_LEGACY

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, sr = librosa.load(audio_path, sr=config.SAMPLE_RATE, mono=True)

    CQT = librosa.cqt(
        y,
        sr=sr,
        hop_length=config.HOP_LENGTH,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        fmin=librosa.note_to_hz(config.FMIN_NOTE)
    )

    # Convert to dB scale (matching training preprocessing)
    features = librosa.amplitude_to_db(np.abs(CQT), ref=np.max)
    features = features.T.astype(np.float32)

    return features


def viterbi_decode(log_probs, transition_penalty=1.0):
    """
    Viterbi decoding with transition penalty for temporal smoothness.

    This implements a simplified linear CRF where:
    - Observation potential = log_probs from model
    - Transition potential = -penalty for chord changes (diagonal = 0)

    Args:
        log_probs: Log probabilities [n_frames, n_classes]
        transition_penalty: Penalty for chord transitions (higher = smoother)

    Returns:
        best_path: Optimal sequence of class indices
    """
    n_frames, n_classes = log_probs.shape

    # Initialize Viterbi tables
    viterbi = np.zeros((n_frames, n_classes), dtype=np.float32)
    backpointers = np.zeros((n_frames, n_classes), dtype=np.int32)

    # Initial frame
    viterbi[0] = log_probs[0]

    # Forward pass with transition penalty
    for t in range(1, n_frames):
        for c in range(n_classes):
            # Best previous state considering transition penalty
            # Same state: no penalty, different state: penalty
            scores = viterbi[t-1].copy()
            scores -= transition_penalty  # Penalty for all transitions
            scores[c] += transition_penalty  # Remove penalty for self-transition

            best_prev = np.argmax(scores)
            viterbi[t, c] = scores[best_prev] + log_probs[t, c]
            backpointers[t, c] = best_prev

    # Backward pass to get best path
    best_path = np.zeros(n_frames, dtype=np.int32)
    best_path[-1] = np.argmax(viterbi[-1])

    for t in range(n_frames - 2, -1, -1):
        best_path[t] = backpointers[t + 1, best_path[t + 1]]

    return best_path


def logits_to_log_probs(logits):
    """Convert logits to log probabilities (log softmax)."""
    # Numerically stable log softmax
    max_logits = logits.max(axis=-1, keepdims=True)
    shifted = logits - max_logits
    log_sum_exp = np.log(np.exp(shifted).sum(axis=-1, keepdims=True) + 1e-10)
    return shifted - log_sum_exp


def decode_chordformer_output(outputs, use_crf=True, transition_penalty=1.0):
    """
    Decode ChordFormer's 6-head output into chord labels.

    Implements CRF decoding from Section III.F of ChordFormer paper.
    The paper's equations (10-12) describe joint CRF over all heads,
    but joint decoding over 79K states is expensive. Instead, we apply
    Viterbi independently to each head, which captures the smoothness
    constraint while remaining computationally tractable.

    Args:
        outputs: Dict of logits for each head
        use_crf: Whether to apply CRF/Viterbi decoding to all heads
        transition_penalty: γ parameter from equation (12) - penalty for transitions

    Returns:
        chord_labels: List of chord label strings
    """
    if use_crf:
        # Apply Viterbi decoding to ALL heads (following paper's CRF approach)
        # This enforces temporal smoothness on each chord component

        # Root+Triad (most important - determines main chord)
        root_triad_logits = outputs['root_triad'].squeeze(0).cpu().numpy()
        root_triad_log_probs = logits_to_log_probs(root_triad_logits)
        root_triad_preds = viterbi_decode(root_triad_log_probs, transition_penalty)

        # Bass note
        bass_logits = outputs['bass'].squeeze(0).cpu().numpy()
        bass_log_probs = logits_to_log_probs(bass_logits)
        bass_preds = viterbi_decode(bass_log_probs, transition_penalty)

        # 7th extension
        seventh_logits = outputs['7th'].squeeze(0).cpu().numpy()
        seventh_log_probs = logits_to_log_probs(seventh_logits)
        seventh_preds = viterbi_decode(seventh_log_probs, transition_penalty)

        # 9th extension
        ninth_logits = outputs['9th'].squeeze(0).cpu().numpy()
        ninth_log_probs = logits_to_log_probs(ninth_logits)
        ninth_preds = viterbi_decode(ninth_log_probs, transition_penalty)

        # 11th extension
        eleventh_logits = outputs['11th'].squeeze(0).cpu().numpy()
        eleventh_log_probs = logits_to_log_probs(eleventh_logits)
        eleventh_preds = viterbi_decode(eleventh_log_probs, transition_penalty)

        # 13th extension
        thirteenth_logits = outputs['13th'].squeeze(0).cpu().numpy()
        thirteenth_log_probs = logits_to_log_probs(thirteenth_logits)
        thirteenth_preds = viterbi_decode(thirteenth_log_probs, transition_penalty)
    else:
        # Simple argmax decoding (no temporal smoothing)
        root_triad_preds = outputs['root_triad'].argmax(dim=-1).squeeze(0).cpu().numpy()
        bass_preds = outputs['bass'].argmax(dim=-1).squeeze(0).cpu().numpy()
        seventh_preds = outputs['7th'].argmax(dim=-1).squeeze(0).cpu().numpy()
        ninth_preds = outputs['9th'].argmax(dim=-1).squeeze(0).cpu().numpy()
        eleventh_preds = outputs['11th'].argmax(dim=-1).squeeze(0).cpu().numpy()
        thirteenth_preds = outputs['13th'].argmax(dim=-1).squeeze(0).cpu().numpy()

    # Reconstruct chord labels from all 6 components
    chord_labels = []
    for i in range(len(root_triad_preds)):
        chord = reconstruct_chord_label(
            root_triad_preds[i],
            bass_preds[i],
            seventh_preds[i],
            ninth_preds[i],
            eleventh_preds[i],
            thirteenth_preds[i]
        )
        chord_labels.append(chord)

    return chord_labels


def reconstruct_chord_label(root_triad_idx, bass_idx, seventh_idx, ninth_idx, eleventh_idx, thirteenth_idx):
    """
    Reconstruct chord label from structured predictions.

    Args:
        root_triad_idx: 0=N, 1-84 = root*7 + triad + 1
        bass_idx: 0=N, 1-12 = bass note
        seventh_idx: 0=N, 1=7, 2=b7, 3=bb7
        ninth_idx: 0=N, 1=9, 2=#9, 3=b9
        eleventh_idx: 0=N, 1=11, 2=#11
        thirteenth_idx: 0=N, 1=13, 2=b13

    Returns:
        Chord label string (e.g., "C:maj7", "G:min/B")
    """
    # No chord
    if root_triad_idx == 0:
        return 'N'

    # Decode root and triad
    root_idx = (root_triad_idx - 1) // 7
    triad_idx = (root_triad_idx - 1) % 7

    root = ROOT_NAMES[root_idx]

    # Triad type (0-5 = maj, min, sus4, sus2, dim, aug; 6 = N-quality)
    if triad_idx < len(TRIAD_TYPES):
        triad = TRIAD_TYPES[triad_idx]
    else:
        triad = ''  # N-quality, just root

    # Build chord label
    if triad == 'maj':
        chord = f"{root}:maj"
    elif triad == 'min':
        chord = f"{root}:min"
    elif triad:
        chord = f"{root}:{triad}"
    else:
        chord = root

    # Add extensions
    extensions = []

    if seventh_idx > 0:
        ext = SEVENTH_TYPES[seventh_idx]
        if ext != 'N':
            extensions.append(ext)

    if ninth_idx > 0:
        ext = NINTH_TYPES[ninth_idx]
        if ext != 'N':
            extensions.append(ext)

    if eleventh_idx > 0:
        ext = ELEVENTH_TYPES[eleventh_idx]
        if ext != 'N':
            extensions.append(ext)

    if thirteenth_idx > 0:
        ext = THIRTEENTH_TYPES[thirteenth_idx]
        if ext != 'N':
            extensions.append(ext)

    # Combine extensions into chord label
    if extensions:
        # Simplify common patterns
        ext_str = ''.join(extensions)
        if 'maj' in chord and ext_str.startswith('7'):
            chord = chord.replace(':maj', ':maj7')
        elif 'min' in chord and ext_str.startswith('b7'):
            chord = chord.replace(':min', ':min7')
        else:
            chord = f"{chord}({ext_str})"

    # Add bass note if different from root
    if bass_idx > 0:
        bass_note = ROOT_NAMES[bass_idx - 1]
        if bass_note != root:
            chord = f"{chord}/{bass_note}"

    return chord


# =============================================================================
# MIREX 2025 Inference Functions
# =============================================================================

def intervals_to_quality(intervals_binary):
    """
    Convert 12-dim binary interval vector to chord quality string.

    This is the reverse mapping for inference - converts predicted pitch vectors
    back to chord quality names.

    Args:
        intervals_binary: [12] binary array where 1 = interval present

    Returns:
        Chord quality string (e.g., 'maj7', 'min', 'dim')

    Examples:
        [1,0,0,0,1,0,0,1,0,0,0,0] -> 'maj'  (P1, M3, P5)
        [1,0,0,1,0,0,0,1,0,0,0,0] -> 'min'  (P1, m3, P5)
        [1,0,0,1,0,0,1,0,0,0,0,0] -> 'dim'  (P1, m3, d5)
        [1,0,0,0,1,0,0,1,0,0,1,0] -> '7'    (P1, M3, P5, m7)
    """
    # Extract active interval indices
    if isinstance(intervals_binary, np.ndarray):
        active_intervals = tuple(i for i, v in enumerate(intervals_binary) if v > 0.5)
    else:
        active_intervals = tuple(i for i, v in enumerate(intervals_binary) if v > 0.5)

    # Direct lookup
    if active_intervals in config.INTERVAL_TO_QUALITY:
        return config.INTERVAL_TO_QUALITY[active_intervals]

    # Fuzzy matching for close matches (handles prediction noise)
    best_match = 'maj'  # default
    best_score = 0
    for pattern, quality in config.INTERVAL_TO_QUALITY.items():
        # Count matching intervals
        score = sum(1 for i in pattern if i in active_intervals)
        if score > best_score:
            best_score = score
            best_match = quality

    return best_match


def degree_to_absolute_root(key, degree):
    """
    Convert key and scale degree to absolute root note.

    Uses practical chord notation - no theoretical spellings like E#, Fb, B#, Cb.
    These don't exist in real chord symbols!

    Args:
        key: Key string (e.g., 'C', 'F#', 'Bb') or key index (0-12)
        degree: Scale degree string (e.g., 'I', '#IV', 'bVII') or degree index (0-17)

    Returns:
        Absolute root note using practical spelling (C, C#, Db, D, etc.)

    Examples:
        degree_to_absolute_root('C', 'I')    -> 'C'
        degree_to_absolute_root('C', 'V')    -> 'G'
        degree_to_absolute_root('B', '#IV')  -> 'F'   (NOT E# - that doesn't exist!)
        degree_to_absolute_root('Gb', 'VII') -> 'F'
        degree_to_absolute_root('G', 'IV')   -> 'C'
    """
    # Handle indices
    if isinstance(key, int):
        if key == 0:
            return 'N'
        key = config.SHARP_NAMES[key - 1] if key <= 12 else 'C'

    if isinstance(degree, int):
        if degree == 0 or degree >= len(config.SCALE_DEGREES):
            return 'N'
        degree = config.SCALE_DEGREES[degree]

    if degree == 'N' or key is None:
        return 'N'

    # Get key pitch class
    key_base = key.replace('b', '').replace('#', '')[0]
    if key_base not in config.MIREX_ROOT_TO_IDX:
        return 'N'

    key_pitch = config.MIREX_ROOT_TO_IDX[key_base]
    if '#' in key:
        key_pitch = (key_pitch + 1) % 12
    elif 'b' in key:
        key_pitch = (key_pitch - 1) % 12

    # Get semitone offset from degree
    semitones = config.DEGREE_SEMITONES.get(degree)
    if semitones is None:
        return 'N'

    # Compute target pitch class
    target_pitch = (key_pitch + semitones) % 12

    # Use key signature to determine sharp vs flat spelling
    # Sharp keys (G, D, A, E, B, F#) -> use sharp names
    # Flat keys (F, Bb, Eb, Ab, Db, Gb) -> use flat names
    key_sharps = config.KEY_SIGNATURES.get(key, 0)
    if key_sharps >= 0:
        return config.SHARP_NAMES[target_pitch]
    else:
        return config.FLAT_NAMES[target_pitch]


def decode_mirex_output(outputs, use_crf=True, transition_penalty=1.0):
    """
    Decode MIREX model's output into chord labels.

    Combines key, degree, and pitch vectors to reconstruct full chord symbols
    with proper enharmonic spelling.

    Args:
        outputs: Dict of logits for each head
        use_crf: Whether to apply CRF/Viterbi decoding
        transition_penalty: CRF transition penalty

    Returns:
        chord_labels: List of chord label strings
        key_preds: Array of key predictions per frame (0=N, 1-12=C through B)
    """
    if use_crf:
        # Apply Viterbi decoding to categorical heads
        key_logits = outputs['key'].squeeze(0).cpu().numpy()
        key_log_probs = logits_to_log_probs(key_logits)
        key_preds = viterbi_decode(key_log_probs, transition_penalty)

        degree_logits = outputs['degree'].squeeze(0).cpu().numpy()
        degree_log_probs = logits_to_log_probs(degree_logits)
        degree_preds = viterbi_decode(degree_log_probs, transition_penalty)

        bass_logits = outputs['bass'].squeeze(0).cpu().numpy()
        bass_log_probs = logits_to_log_probs(bass_logits)
        bass_preds = viterbi_decode(bass_log_probs, transition_penalty)

        # Binary heads use sigmoid + threshold
        pitches_abs = torch.sigmoid(outputs['pitches_abs']).squeeze(0).cpu().numpy()
        intervals_root = torch.sigmoid(outputs['intervals_root']).squeeze(0).cpu().numpy()
        intervals_bass = torch.sigmoid(outputs['intervals_bass']).squeeze(0).cpu().numpy()
    else:
        # Simple argmax for categorical heads
        key_preds = outputs['key'].argmax(dim=-1).squeeze(0).cpu().numpy()
        degree_preds = outputs['degree'].argmax(dim=-1).squeeze(0).cpu().numpy()
        bass_preds = outputs['bass'].argmax(dim=-1).squeeze(0).cpu().numpy()

        # Binary heads use sigmoid + threshold
        pitches_abs = torch.sigmoid(outputs['pitches_abs']).squeeze(0).cpu().numpy()
        intervals_root = torch.sigmoid(outputs['intervals_root']).squeeze(0).cpu().numpy()
        intervals_bass = torch.sigmoid(outputs['intervals_bass']).squeeze(0).cpu().numpy()

    # Reconstruct chord labels
    chord_labels = []
    for i in range(len(key_preds)):
        chord = reconstruct_mirex_chord_label(
            key_preds[i],
            degree_preds[i],
            bass_preds[i],
            intervals_root[i]
        )
        chord_labels.append(chord)

    return chord_labels, key_preds


def detect_overall_key(key_preds):
    """
    Detect the overall key of a song from frame-level key predictions.

    Uses majority voting across frames, excluding 'N' (no key) predictions.

    Args:
        key_preds: Array of key predictions per frame (0=N, 1-12=C through B)

    Returns:
        key_name: Detected key name (e.g., 'C', 'G#') or 'N' if undetermined
    """
    # Filter out 'N' predictions (index 0)
    valid_keys = key_preds[key_preds > 0]

    if len(valid_keys) == 0:
        return 'N'

    # Count occurrences of each key
    from collections import Counter
    key_counts = Counter(valid_keys)

    # Get the most common key
    most_common_key_idx = key_counts.most_common(1)[0][0]

    # Convert to key name (1-12 maps to C through B)
    key_name = config.SHARP_NAMES[most_common_key_idx - 1]

    return key_name


def reconstruct_mirex_chord_label(key_idx, degree_idx, bass_idx, intervals_root):
    """
    Reconstruct chord label from MIREX predictions.

    Args:
        key_idx: Key index (0=N, 1-12 = C through B)
        degree_idx: Scale degree index (0-17)
        bass_idx: Bass note index (0=N, 1-12 = C through B)
        intervals_root: [12] binary array of intervals from root

    Returns:
        Chord label string (e.g., "C:maj7", "G:min/B")
    """
    # Handle no-chord
    if degree_idx == 0:
        return 'N'

    # Get key name
    key = config.SHARP_NAMES[key_idx - 1] if key_idx > 0 and key_idx <= 12 else 'C'

    # Get degree name
    degree = config.SCALE_DEGREES[degree_idx] if degree_idx < len(config.SCALE_DEGREES) else 'I'

    # Convert degree to absolute root with enharmonic spelling
    root = degree_to_absolute_root(key, degree)

    if root == 'N':
        return 'N'

    # Determine chord quality from intervals
    quality = intervals_to_quality(intervals_root)

    # Build chord label
    if quality == 'maj':
        chord = f"{root}:maj"
    elif quality == 'min':
        chord = f"{root}:min"
    else:
        chord = f"{root}:{quality}"

    # Add bass note if different from root
    if bass_idx > 0 and bass_idx <= 12:
        bass_note = config.SHARP_NAMES[bass_idx - 1]
        # Check if bass is different from root (normalize for comparison)
        root_normalized = root.replace('#', '').replace('b', '')[0]
        bass_normalized = bass_note.replace('#', '').replace('b', '')[0]
        if bass_note != root and bass_normalized != root_normalized:
            chord = f"{chord}/{bass_note}"

    return chord


def frames_to_annotations(chord_labels, hop_length=config.HOP_LENGTH, sr=config.SAMPLE_RATE):
    """
    Convert frame-level chord predictions to time-aligned annotations.

    Returns:
        List of (start_time, end_time, chord_label) tuples
    """
    n_frames = len(chord_labels)
    frame_duration = hop_length / sr

    annotations = []
    current_chord = None
    start_frame = 0

    for i in range(n_frames):
        chord = chord_labels[i]

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


def write_lab_file(annotations, output_path, detected_key=None):
    """Write annotations to .lab file format.

    Args:
        annotations: List of (start, end, chord) tuples
        output_path: Path to output file
        detected_key: Optional detected key to include as header comment
    """
    with open(output_path, 'w') as f:
        if detected_key:
            f.write(f"# Key: {detected_key}\n")
        for start, end, chord in annotations:
            f.write(f"{start:.6f}\t{end:.6f}\t{chord}\n")


def predict_chords(model, features, normalization, model_type, device='cpu',
                   use_crf=True, transition_penalty=1.0, chunk_size=1000, overlap=100):
    """
    Run inference on audio features with chunked processing for long sequences.

    Args:
        model: Trained model
        features: CQT features [n_frames, n_bins]
        normalization: Dict with 'mean' and 'std'
        model_type: 'chordformer', 'mirex', 'bilstm', or 'tcn'
        device: Device for inference
        use_crf: Whether to use CRF decoding (ChordFormer/MIREX only)
        transition_penalty: CRF transition penalty
        chunk_size: Maximum frames per chunk (default 1000, matches training)
        overlap: Overlap between chunks for smooth transitions (default 100)

    Returns:
        annotations: List of (start, end, chord) tuples
        detected_key: Detected key name (e.g., 'C', 'G#') or None for non-MIREX models
    """
    # Normalize features
    mean = np.array(normalization['mean'], dtype=np.float32)
    std = np.array(normalization['std'], dtype=np.float32)
    features_norm = (features - mean) / std

    n_frames = features_norm.shape[0]
    detected_key = None

    # Run inference
    with torch.no_grad():
        if model_type == 'mirex':
            # MIREX degree-based model
            if n_frames <= chunk_size:
                x = torch.from_numpy(features_norm).float().unsqueeze(0).to(device)
                outputs = model(x)
            else:
                outputs = _chunked_mirex_inference(
                    model, features_norm, device, chunk_size, overlap
                )

            chord_labels, key_preds = decode_mirex_output(
                outputs, use_crf=use_crf, transition_penalty=transition_penalty
            )
            detected_key = detect_overall_key(key_preds)
        elif model_type == 'chordformer':
            # Use chunked inference for long sequences to avoid memory issues
            if n_frames <= chunk_size:
                # Short sequence - process directly
                x = torch.from_numpy(features_norm).float().unsqueeze(0).to(device)
                outputs = model(x)
            else:
                # Long sequence - process in overlapping chunks
                outputs = _chunked_chordformer_inference(
                    model, features_norm, device, chunk_size, overlap
                )

            chord_labels = decode_chordformer_output(
                outputs, use_crf=use_crf, transition_penalty=transition_penalty
            )
        else:
            x = torch.from_numpy(features_norm).float().unsqueeze(0).to(device)
            if model_type == 'bilstm':
                lengths = torch.tensor([x.shape[1]], dtype=torch.long)
                logits = model(x, lengths)
            else:
                logits = model(x)

            predictions = logits.argmax(dim=-1).squeeze(0).cpu().numpy()

            # Load chord mapping for legacy models
            # (Assumes idx_to_chord is available)
            chord_labels = [str(p) for p in predictions]  # Placeholder

    # Convert to annotations
    annotations = frames_to_annotations(chord_labels)

    return annotations, detected_key


def _chunked_mirex_inference(model, features_norm, device, chunk_size, overlap):
    """
    Process long sequences in overlapping chunks for MIREX model.

    Args:
        model: MIREX model
        features_norm: Normalized features [n_frames, n_bins]
        device: Device for inference
        chunk_size: Frames per chunk
        overlap: Overlap between chunks

    Returns:
        outputs: Dict of concatenated outputs for each head
    """
    n_frames = features_norm.shape[0]
    step = chunk_size - overlap

    # Initialize output storage
    head_names = ['key', 'degree', 'bass', 'pitches_abs', 'intervals_root', 'intervals_bass']
    all_outputs = {name: [] for name in head_names}

    start = 0
    while start < n_frames:
        end = min(start + chunk_size, n_frames)
        chunk = features_norm[start:end]

        x = torch.from_numpy(chunk).float().unsqueeze(0).to(device)
        chunk_outputs = model(x)

        # Determine which frames to keep
        if start == 0:
            keep_start = 0
        else:
            keep_start = overlap

        keep_end = end - start

        # Extract frames to keep
        for name in head_names:
            kept = chunk_outputs[name][:, keep_start:keep_end].cpu()
            all_outputs[name].append(kept)

        start += step

    # Concatenate all chunks
    outputs = {}
    for name in head_names:
        outputs[name] = torch.cat(all_outputs[name], dim=1).to(device)

    return outputs


def _chunked_chordformer_inference(model, features_norm, device, chunk_size, overlap):
    """
    Process long sequences in overlapping chunks to avoid memory issues.

    The relative positional encoding creates [seq_len, seq_len, d_model] tensors,
    which becomes prohibitively large for long sequences (e.g., 33000^2 * 256 * 4 bytes = ~1TB).

    Args:
        model: ChordFormer model
        features_norm: Normalized features [n_frames, n_bins]
        device: Device for inference
        chunk_size: Frames per chunk
        overlap: Overlap between chunks

    Returns:
        outputs: Dict of concatenated logits for each head
    """
    n_frames = features_norm.shape[0]
    step = chunk_size - overlap

    # Initialize output storage for each head
    head_names = ['root_triad', 'bass', '7th', '9th', '11th', '13th']
    all_outputs = {name: [] for name in head_names}

    # Track which frames we've processed (for handling overlaps)
    processed_frames = 0

    chunk_idx = 0
    start = 0
    while start < n_frames:
        end = min(start + chunk_size, n_frames)
        chunk = features_norm[start:end]

        # Convert to tensor
        x = torch.from_numpy(chunk).float().unsqueeze(0).to(device)

        # Run inference on chunk
        chunk_outputs = model(x)

        # Determine which frames to keep from this chunk
        if start == 0:
            # First chunk: keep all frames
            keep_start = 0
        else:
            # Later chunks: skip the overlap region (use predictions from previous chunk)
            keep_start = overlap

        if end == n_frames:
            # Last chunk: keep to the end
            keep_end = end - start
        else:
            # Middle chunks: keep up to the end
            keep_end = end - start

        # Extract the frames we want to keep
        for name in head_names:
            kept_logits = chunk_outputs[name][:, keep_start:keep_end, :].cpu()
            all_outputs[name].append(kept_logits)

        chunk_idx += 1
        start += step

    # Concatenate all chunks
    outputs = {}
    for name in head_names:
        outputs[name] = torch.cat(all_outputs[name], dim=1).to(device)

    return outputs


def main():
    parser = argparse.ArgumentParser(description='Estimate chords from audio file')
    parser.add_argument('audio_path', type=str, help='Path to audio file')
    parser.add_argument('--checkpoint', type=str,
                        default='../checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--normalization', type=str,
                        default='../features/normalization.json',
                        help='Path to normalization stats')
    parser.add_argument('--output', type=str, default=None,
                        help='Output .lab file path (default: same as input with .lab extension)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cpu or cuda)')
    parser.add_argument('--no_crf', action='store_true',
                        help='Disable CRF decoding (use argmax instead)')
    parser.add_argument('--transition_penalty', type=float, default=1.0,
                        help='CRF transition penalty (higher = smoother)')
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
    model, checkpoint = load_model(args.checkpoint, args.device)
    model_type = checkpoint.get('model_type', 'bilstm')
    print(f"  Model type: {model_type}")

    # Extract features
    print(f"Extracting features from {audio_path}")
    features = extract_features(audio_path, model_type)
    print(f"  {features.shape[0]} frames, {features.shape[1]} bins")

    # Predict chords
    print("Predicting chords...")
    annotations, detected_key = predict_chords(
        model, features, normalization, model_type, args.device,
        use_crf=not args.no_crf, transition_penalty=args.transition_penalty
    )

    # Display detected key
    if detected_key:
        print(f"\nDetected key: {detected_key}")

    # Write output
    write_lab_file(annotations, output_path, detected_key=detected_key)
    print(f"\nChord annotations written to: {output_path}")
    print(f"  {len(annotations)} chord segments")

    # Print first few chords
    print("\nFirst 10 chords:")
    for start, end, chord in annotations[:10]:
        print(f"  {start:6.2f} - {end:6.2f}: {chord}")


if __name__ == '__main__':
    main()
