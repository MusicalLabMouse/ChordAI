"""
Training Script for Chord Recognition
Supports BiLSTM, TCN, and ChordFormer models.
"""

import os
import argparse
import json
from collections import Counter

# Enable CUDA memory fragmentation optimization before importing torch
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

import numpy as np
import torch
import torch.nn as nn

# Disable cuDNN benchmarking to ensure consistent memory usage across GPUs
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from pathlib import Path

from model import (ChordRecognitionModel, ChordRecognitionModelTCN, ChordFormerModel,
                   ChordFormerLoss, MIREXChordFormerModel, MIREXLoss)
from dataset import (get_dataloaders, get_chordformer_dataloaders, get_mirex_dataloaders,
                     CHORDFORMER_HEADS, MIREX_HEADS_CATEGORICAL, MIREX_HEADS_BINARY, MIREX_HEADS)
from amp_helper import AMPHelper  # [AMP] Delete this line to remove AMP
from inference import viterbi_decode, reconstruct_chord_label  # CRF decoding (Section III.F)
import config

# Optional: mir_eval for WCSR metrics (Section IV.C of ChordFormer paper)
try:
    import mir_eval
    HAS_MIR_EVAL = True
except ImportError:
    HAS_MIR_EVAL = False


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train for one epoch.

    Args:
        model: ChordRecognitionModel
        train_loader: Training DataLoader
        criterion: Loss function
        optimizer: Optimizer
        device: Device (cuda/cpu)

    Returns:
        avg_loss: Average training loss
        avg_acc: Average training accuracy
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_frames = 0

    progress_bar = tqdm(train_loader, desc="Training", unit="batch",
                        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

    for features, labels, lengths in progress_bar:
        features = features.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(features, lengths)  # [batch, max_len, num_classes]

        # Reshape for loss computation
        outputs_flat = outputs.view(-1, outputs.shape[-1])  # [batch*max_len, num_classes]
        labels_flat = labels.view(-1)  # [batch*max_len]

        # Compute loss (CrossEntropyLoss ignores -1 labels)
        loss = criterion(outputs_flat, labels_flat)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        # Compute accuracy (only on non-padded frames)
        with torch.no_grad():
            predictions = outputs.argmax(dim=-1)  # [batch, max_len]
            mask = (labels != -1)  # Valid frames
            correct = ((predictions == labels) & mask).sum().item()
            total_correct += correct
            total_frames += mask.sum().item()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    avg_acc = total_correct / total_frames if total_frames > 0 else 0.0

    return avg_loss, avg_acc


def validate(model, val_loader, criterion, device):
    """
    Validate the model.

    Args:
        model: ChordRecognitionModel
        val_loader: Validation DataLoader
        criterion: Loss function
        device: Device (cuda/cpu)

    Returns:
        avg_loss: Average validation loss
        avg_acc: Average validation accuracy
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_frames = 0

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation", unit="batch",
                            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

        for features, labels, lengths in progress_bar:
            features = features.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(features, lengths)

            # Reshape for loss computation
            outputs_flat = outputs.view(-1, outputs.shape[-1])
            labels_flat = labels.view(-1)

            # Compute loss
            loss = criterion(outputs_flat, labels_flat)
            total_loss += loss.item()

            # Compute accuracy
            predictions = outputs.argmax(dim=-1)
            mask = (labels != -1)
            correct = ((predictions == labels) & mask).sum().item()
            total_correct += correct
            total_frames += mask.sum().item()

    avg_loss = total_loss / len(val_loader)
    avg_acc = total_correct / total_frames if total_frames > 0 else 0.0

    return avg_loss, avg_acc


def save_checkpoint(model, optimizer, epoch, val_acc, checkpoint_path, model_type='bilstm', num_classes=25, hidden_size=256, model_config=None):
    """
    Save model checkpoint.

    Args:
        model: ChordRecognitionModel, ChordRecognitionModelTCN, or ChordFormerModel
        optimizer: Optimizer
        epoch: Current epoch
        val_acc: Validation accuracy
        checkpoint_path: Path to save checkpoint
        model_type: Model architecture type ('bilstm', 'tcn', or 'chordformer')
        num_classes: Number of chord classes (for legacy models)
        hidden_size: Hidden size / TCN channels (for legacy models)
        model_config: Additional model configuration (for ChordFormer)
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'model_type': model_type,
        'num_classes': num_classes,
        'hidden_size': hidden_size
    }
    if model_config:
        checkpoint['config'] = model_config
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


# ===================== ChordFormer Training Functions =====================

def train_epoch_chordformer(model, train_loader, criterion, optimizer, device, max_grad_norm=5.0, amp=None):
    """
    Train ChordFormer for one epoch.

    Args:
        model: ChordFormerModel
        train_loader: Training DataLoader
        criterion: ChordFormerLoss
        optimizer: Optimizer
        device: Device (cuda/cpu)
        max_grad_norm: Gradient clipping norm
        amp: AMPHelper instance (None = disabled) [AMP]

    Returns:
        avg_loss: Average training loss
        head_accs: Dict of per-head accuracies
    """
    model.train()
    total_loss = 0.0
    head_correct = {head: 0 for head in CHORDFORMER_HEADS}
    head_total = {head: 0 for head in CHORDFORMER_HEADS}

    progress_bar = tqdm(train_loader, desc="Training", unit="batch",
                        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

    for features, labels, lengths in progress_bar:
        features = features.to(device)
        labels = {head: labels[head].to(device) for head in CHORDFORMER_HEADS}

        optimizer.zero_grad()

        # [AMP] Forward pass with optional mixed precision
        if amp:
            with amp.autocast():
                outputs = model(features, lengths)
                loss = criterion(outputs, labels)
        else:
            outputs = model(features, lengths)
            loss = criterion(outputs, labels)

        # [AMP] Backward pass (amp.backward handles scaling if enabled)
        if amp:
            amp.backward(loss, optimizer, model, max_grad_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

        # Compute accuracy for each head
        with torch.no_grad():
            for head in CHORDFORMER_HEADS:
                predictions = outputs[head].argmax(dim=-1)
                mask = (labels[head] != -1)
                correct = ((predictions == labels[head]) & mask).sum().item()
                head_correct[head] += correct
                head_total[head] += mask.sum().item()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    head_accs = {
        head: head_correct[head] / head_total[head] if head_total[head] > 0 else 0.0
        for head in CHORDFORMER_HEADS
    }

    return avg_loss, head_accs


def validate_chordformer(model, val_loader, criterion, device, use_crf=True, transition_penalty=None):
    """
    Validate ChordFormer model with optional CRF decoding.

    Uses Viterbi decoding for evaluation to match ChordFormer paper (Section III.F).
    Paper equations (10-12) describe CRF decoding with transition penalty γ.

    Args:
        model: ChordFormerModel
        val_loader: Validation DataLoader
        criterion: ChordFormerLoss
        device: Device (cuda/cpu)
        use_crf: Whether to use CRF/Viterbi decoding (default True, as per paper)
        transition_penalty: γ parameter from equation (12). If None, uses config value.

    Returns:
        avg_loss: Average validation loss
        head_accs: Dict of per-head accuracies
    """
    if transition_penalty is None:
        transition_penalty = config.TRANSITION_PENALTY

    model.eval()
    total_loss = 0.0
    head_correct = {head: 0 for head in CHORDFORMER_HEADS}
    head_total = {head: 0 for head in CHORDFORMER_HEADS}

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation", unit="batch",
                            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

        for features, labels, lengths in progress_bar:
            features = features.to(device)
            labels = {head: labels[head].to(device) for head in CHORDFORMER_HEADS}

            # Forward pass
            outputs = model(features, lengths)

            # Compute loss (always without CRF - loss is for training)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Compute accuracy for each head
            batch_size = features.shape[0]
            for head in CHORDFORMER_HEADS:
                if use_crf:
                    # CRF/Viterbi decoding (Section III.F of ChordFormer paper)
                    # Apply Viterbi independently to each sample in batch
                    logits = outputs[head]  # [batch, time, classes]
                    log_probs = torch.log_softmax(logits, dim=-1).cpu().numpy()

                    predictions = torch.zeros_like(labels[head])
                    for b in range(batch_size):
                        seq_len = int(lengths[b].item()) if lengths is not None else logits.shape[1]
                        # Viterbi decode this sample
                        pred_seq = viterbi_decode(log_probs[b, :seq_len], transition_penalty)
                        predictions[b, :seq_len] = torch.from_numpy(pred_seq)
                        # Padded region stays 0 (will be masked out anyway)
                else:
                    # Simple argmax (faster but less accurate)
                    predictions = outputs[head].argmax(dim=-1)

                mask = (labels[head] != -1)
                correct = ((predictions == labels[head]) & mask).sum().item()
                head_correct[head] += correct
                head_total[head] += mask.sum().item()

    avg_loss = total_loss / len(val_loader)
    head_accs = {
        head: head_correct[head] / head_total[head] if head_total[head] > 0 else 0.0
        for head in CHORDFORMER_HEADS
    }

    return avg_loss, head_accs


# ===================== MIREX Training Functions =====================

def train_epoch_mirex(model, train_loader, criterion, optimizer, device, max_grad_norm=5.0, amp=None):
    """
    Train MIREX model for one epoch.

    Args:
        model: MIREXChordFormerModel
        train_loader: Training DataLoader
        criterion: MIREXLoss
        optimizer: Optimizer
        device: Device (cuda/cpu)
        max_grad_norm: Gradient clipping norm
        amp: AMPHelper instance (None = disabled)

    Returns:
        avg_loss: Average training loss
        head_accs: Dict of per-head accuracies (categorical heads only)
    """
    model.train()
    total_loss = 0.0
    head_correct = {head: 0 for head in MIREX_HEADS_CATEGORICAL}
    head_total = {head: 0 for head in MIREX_HEADS_CATEGORICAL}

    progress_bar = tqdm(train_loader, desc="Training", unit="batch",
                        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

    for features, labels, lengths in progress_bar:
        features = features.to(device)
        labels_device = {}
        for head in MIREX_HEADS_CATEGORICAL:
            labels_device[head] = labels[head].to(device)
        for head in MIREX_HEADS_BINARY:
            labels_device[head] = labels[head].to(device)

        optimizer.zero_grad()

        # Debug: Memory before forward pass (first batch only)
        if progress_bar.n == 0 and torch.cuda.is_available():
            print(f"\n  [DEBUG] Before forward: allocated={torch.cuda.memory_allocated() / 1024**3:.2f} GB, "
                  f"reserved={torch.cuda.memory_reserved() / 1024**3:.2f} GB")
            print(f"  [DEBUG] Features shape: {features.shape}, dtype: {features.dtype}")

        # Forward pass with optional mixed precision
        if amp:
            with amp.autocast():
                outputs = model(features, lengths)
                loss = criterion(outputs, labels_device)
        else:
            outputs = model(features, lengths)
            loss = criterion(outputs, labels_device)

        # Debug: Memory after forward pass (first batch only)
        if progress_bar.n == 0 and torch.cuda.is_available():
            print(f"  [DEBUG] After forward: allocated={torch.cuda.memory_allocated() / 1024**3:.2f} GB, "
                  f"reserved={torch.cuda.memory_reserved() / 1024**3:.2f} GB")

        # Backward pass
        if amp:
            amp.backward(loss, optimizer, model, max_grad_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

        # Debug: Memory after backward pass (first batch only)
        if progress_bar.n == 0 and torch.cuda.is_available():
            print(f"  [DEBUG] After backward: allocated={torch.cuda.memory_allocated() / 1024**3:.2f} GB, "
                  f"reserved={torch.cuda.memory_reserved() / 1024**3:.2f} GB")

        # Compute accuracy for categorical heads
        with torch.no_grad():
            for head in MIREX_HEADS_CATEGORICAL:
                predictions = outputs[head].argmax(dim=-1)
                mask = (labels_device[head] != -1)
                correct = ((predictions == labels_device[head]) & mask).sum().item()
                head_correct[head] += correct
                head_total[head] += mask.sum().item()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    head_accs = {
        head: head_correct[head] / head_total[head] if head_total[head] > 0 else 0.0
        for head in MIREX_HEADS_CATEGORICAL
    }

    return avg_loss, head_accs


def validate_mirex(model, val_loader, criterion, device, use_crf=True, transition_penalty=None):
    """
    Validate MIREX model.

    Args:
        model: MIREXChordFormerModel
        val_loader: Validation DataLoader
        criterion: MIREXLoss
        device: Device (cuda/cpu)
        use_crf: Whether to use CRF/Viterbi decoding
        transition_penalty: CRF transition penalty

    Returns:
        avg_loss: Average validation loss
        head_accs: Dict of per-head accuracies (categorical heads)
    """
    if transition_penalty is None:
        transition_penalty = config.TRANSITION_PENALTY

    model.eval()
    total_loss = 0.0
    head_correct = {head: 0 for head in MIREX_HEADS_CATEGORICAL}
    head_total = {head: 0 for head in MIREX_HEADS_CATEGORICAL}

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation", unit="batch",
                            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

        for features, labels, lengths in progress_bar:
            features = features.to(device)
            labels_device = {}
            for head in MIREX_HEADS_CATEGORICAL:
                labels_device[head] = labels[head].to(device)
            for head in MIREX_HEADS_BINARY:
                labels_device[head] = labels[head].to(device)

            # Forward pass
            outputs = model(features, lengths)

            # Compute loss
            loss = criterion(outputs, labels_device)
            total_loss += loss.item()

            # Compute accuracy for categorical heads
            batch_size = features.shape[0]
            for head in MIREX_HEADS_CATEGORICAL:
                if use_crf:
                    # CRF/Viterbi decoding
                    logits = outputs[head]
                    log_probs = torch.log_softmax(logits, dim=-1).cpu().numpy()

                    predictions = torch.zeros_like(labels_device[head])
                    for b in range(batch_size):
                        seq_len = int(lengths[b].item()) if lengths is not None else logits.shape[1]
                        pred_seq = viterbi_decode(log_probs[b, :seq_len], transition_penalty)
                        predictions[b, :seq_len] = torch.from_numpy(pred_seq)
                else:
                    predictions = outputs[head].argmax(dim=-1)

                mask = (labels_device[head] != -1)
                correct = ((predictions == labels_device[head]) & mask).sum().item()
                head_correct[head] += correct
                head_total[head] += mask.sum().item()

    avg_loss = total_loss / len(val_loader)
    head_accs = {
        head: head_correct[head] / head_total[head] if head_total[head] > 0 else 0.0
        for head in MIREX_HEADS_CATEGORICAL
    }

    return avg_loss, head_accs


def compute_mirex_class_weights(train_loader, head_sizes, gamma=0.5, w_max=10.0):
    """
    Compute class re-weighting factors for MIREX heads.

    Args:
        train_loader: Training DataLoader
        head_sizes: Dict of number of classes per head
        gamma: Balancing factor
        w_max: Maximum weight clamp

    Returns:
        weights: Dict of weight tensors for categorical heads
    """
    print("Computing MIREX class weights from training data...")

    head_counts = {head: Counter() for head in MIREX_HEADS_CATEGORICAL}

    for _, labels, _ in tqdm(train_loader, desc="Counting classes"):
        for head in MIREX_HEADS_CATEGORICAL:
            valid_labels = labels[head][labels[head] != -1].numpy()
            head_counts[head].update(valid_labels)

    weights = {}
    for head in MIREX_HEADS_CATEGORICAL:
        counts = head_counts[head]
        if not counts:
            weights[head] = None
            continue

        max_count = max(counts.values())
        num_classes = head_sizes[head]

        weight_array = np.full(num_classes, w_max, dtype=np.float32)
        for cls, count in counts.items():
            w = min((count / max_count) ** (-gamma), w_max)
            weight_array[cls] = w

        weights[head] = torch.from_numpy(weight_array)
        print(f"  {head}: {len(counts)} classes found (of {num_classes}), max_weight={weight_array.max():.2f}")

    return weights


# ===================== Evaluation Metrics (Section IV.C-D of ChordFormer paper) =====================

def compute_accframe(predictions, targets, mask):
    """
    Compute mean frame-wise accuracy (equation 14 from ChordFormer paper).

    accframe = Σ z_i / Σ Z_i

    Where z_i = correctly predicted frames, Z_i = total frames for track i.

    Args:
        predictions: Tensor [batch, time] of predicted class indices
        targets: Tensor [batch, time] of ground truth class indices
        mask: Boolean tensor [batch, time] where True = valid frame

    Returns:
        accframe: Float, frame-wise accuracy
    """
    correct = ((predictions == targets) & mask).sum().item()
    total = mask.sum().item()
    return correct / total if total > 0 else 0.0


def compute_accclass(predictions, targets, mask, num_classes):
    """
    Compute mean class-wise accuracy (equation 15 from ChordFormer paper).

    accclass = (1/|V|) Σ_{v∈V} (Σ z_i^v / Σ Z_i^v)

    Where z_i^v = correctly predicted frames for class v,
          Z_i^v = total frames labeled as class v.

    This metric averages accuracy across all classes, giving equal weight
    to rare and common classes. Important for imbalanced chord datasets.

    Args:
        predictions: Tensor [batch, time] of predicted class indices
        targets: Tensor [batch, time] of ground truth class indices
        mask: Boolean tensor [batch, time] where True = valid frame
        num_classes: Number of classes

    Returns:
        accclass: Float, class-wise accuracy
    """
    class_correct = torch.zeros(num_classes, device=predictions.device)
    class_total = torch.zeros(num_classes, device=predictions.device)

    for c in range(num_classes):
        class_mask = (targets == c) & mask
        class_total[c] = class_mask.sum()
        class_correct[c] = ((predictions == c) & class_mask).sum()

    # Average over classes with samples (avoid division by zero)
    valid = class_total > 0
    if valid.sum() == 0:
        return 0.0

    per_class_acc = class_correct[valid] / class_total[valid]
    return per_class_acc.mean().item()


def compute_mir_eval_wcsr(ref_intervals, ref_labels, est_intervals, est_labels):
    """
    Compute Weighted Chord Symbol Recall (WCSR) using mir_eval library.

    WCSR (equation 13 from ChordFormer paper):
    WCSR = (Σ z_i / Σ Z_i) × 100

    Uses mir_eval.chord.evaluate() which returns various metrics:
    - Root: Root note accuracy
    - Thirds: Major/minor third accuracy
    - Triads: Full triad accuracy (root + quality)
    - Tetrads: Triad + 7th accuracy
    - Mirex: MIREX competition metric
    - Majmin: Major/minor quality accuracy
    - Sevenths: 7th extension accuracy

    Args:
        ref_intervals: numpy array [[start, end], ...] for reference
        ref_labels: list of reference chord labels (mir_eval format)
        est_intervals: numpy array [[start, end], ...] for estimation
        est_labels: list of estimated chord labels (mir_eval format)

    Returns:
        scores: Dict of metric names to values
    """
    if not HAS_MIR_EVAL:
        return {}

    try:
        scores = mir_eval.chord.evaluate(ref_intervals, ref_labels, est_intervals, est_labels)
        # mir_eval returns lowercase keys
        return {
            'root': scores['root'],
            'thirds': scores['thirds'],
            'triads': scores['triads'],
            'tetrads': scores['tetrads'],
            'mirex': scores['mirex'],
            'majmin': scores['majmin'],
            'sevenths': scores['sevenths']
        }
    except Exception as e:
        print(f"Warning: mir_eval error: {e}")
        return {}


def predictions_to_chord_labels(outputs, use_crf=True, transition_penalty=1.0, lengths=None):
    """
    Convert model outputs to chord label strings for mir_eval evaluation.

    Args:
        outputs: Dict of model outputs {'root_triad': [batch, time, classes], ...}
        use_crf: Whether to use CRF/Viterbi decoding
        transition_penalty: CRF transition penalty (γ parameter)
        lengths: Sequence lengths for each sample [batch]

    Returns:
        List of lists of chord labels [[chord1, chord2, ...], ...]
    """
    batch_size = outputs['root_triad'].shape[0]
    max_len = outputs['root_triad'].shape[1]
    all_chord_labels = []

    for b in range(batch_size):
        seq_len = int(lengths[b].item()) if lengths is not None else max_len

        if use_crf:
            # Viterbi decode each head
            root_triad_log_probs = torch.log_softmax(outputs['root_triad'][b, :seq_len], dim=-1).cpu().numpy()
            bass_log_probs = torch.log_softmax(outputs['bass'][b, :seq_len], dim=-1).cpu().numpy()
            seventh_log_probs = torch.log_softmax(outputs['7th'][b, :seq_len], dim=-1).cpu().numpy()
            ninth_log_probs = torch.log_softmax(outputs['9th'][b, :seq_len], dim=-1).cpu().numpy()
            eleventh_log_probs = torch.log_softmax(outputs['11th'][b, :seq_len], dim=-1).cpu().numpy()
            thirteenth_log_probs = torch.log_softmax(outputs['13th'][b, :seq_len], dim=-1).cpu().numpy()

            root_triad_preds = viterbi_decode(root_triad_log_probs, transition_penalty)
            bass_preds = viterbi_decode(bass_log_probs, transition_penalty)
            seventh_preds = viterbi_decode(seventh_log_probs, transition_penalty)
            ninth_preds = viterbi_decode(ninth_log_probs, transition_penalty)
            eleventh_preds = viterbi_decode(eleventh_log_probs, transition_penalty)
            thirteenth_preds = viterbi_decode(thirteenth_log_probs, transition_penalty)
        else:
            root_triad_preds = outputs['root_triad'][b, :seq_len].argmax(dim=-1).cpu().numpy()
            bass_preds = outputs['bass'][b, :seq_len].argmax(dim=-1).cpu().numpy()
            seventh_preds = outputs['7th'][b, :seq_len].argmax(dim=-1).cpu().numpy()
            ninth_preds = outputs['9th'][b, :seq_len].argmax(dim=-1).cpu().numpy()
            eleventh_preds = outputs['11th'][b, :seq_len].argmax(dim=-1).cpu().numpy()
            thirteenth_preds = outputs['13th'][b, :seq_len].argmax(dim=-1).cpu().numpy()

        # Reconstruct chord labels
        chord_labels = []
        for i in range(seq_len):
            chord = reconstruct_chord_label(
                root_triad_preds[i], bass_preds[i], seventh_preds[i],
                ninth_preds[i], eleventh_preds[i], thirteenth_preds[i]
            )
            chord_labels.append(chord)

        all_chord_labels.append(chord_labels)

    return all_chord_labels


def compute_class_weights_from_data(train_loader, head_sizes, gamma=0.5, w_max=10.0):
    """
    Compute class re-weighting factors from training data.

    Uses inverse frequency weighting as per ChordFormer paper:
    w_c = min((n_c / max_n)^(-gamma), w_max)

    Args:
        train_loader: Training DataLoader
        head_sizes: Dict of number of classes per head
        gamma: Balancing factor (0=uniform, 1=full inverse)
        w_max: Maximum weight clamp

    Returns:
        weights: Dict of weight tensors for each head
    """
    print("Computing class weights from training data...")

    # Count class occurrences for each head
    head_counts = {head: Counter() for head in CHORDFORMER_HEADS}

    for _, labels, _ in tqdm(train_loader, desc="Counting classes"):
        for head in CHORDFORMER_HEADS:
            # Get non-padded labels
            valid_labels = labels[head][labels[head] != -1].numpy()
            head_counts[head].update(valid_labels)

    # Compute weights for each head
    weights = {}
    for head in CHORDFORMER_HEADS:
        counts = head_counts[head]
        if not counts:
            weights[head] = None
            continue

        max_count = max(counts.values())
        # Use head_sizes for correct number of classes (not just max index found)
        num_classes = head_sizes[head]

        # Initialize all weights to w_max (unseen classes get max weight)
        weight_array = np.full(num_classes, w_max, dtype=np.float32)
        for cls, count in counts.items():
            w = min((count / max_count) ** (-gamma), w_max)
            weight_array[cls] = w

        weights[head] = torch.from_numpy(weight_array)
        print(f"  {head}: {len(counts)} classes found (of {num_classes}), max_weight={weight_array.max():.2f}")

    return weights


def main():
    parser = argparse.ArgumentParser(description='Train chord recognition model')
    parser.add_argument('--features_dir', type=str, default='../features',
                        help='Path to features directory')
    parser.add_argument('--checkpoint_dir', type=str, default='../checkpoints',
                        help='Path to save checkpoints')
    parser.add_argument('--model_type', type=str, default='bilstm',
                        choices=['bilstm', 'tcn', 'chordformer', 'mirex'],
                        help='Model architecture: bilstm, tcn, chordformer, or mirex')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (default: 24 for chordformer, 64 for legacy). LR auto-scales with batch size.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate. If not set, uses linear scaling: lr = 1e-3 * (batch_size/24) for chordformer')
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='LSTM hidden size (bilstm) or TCN channels (tcn) or Conformer dim (chordformer)')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout probability')
    parser.add_argument('--early_stopping_patience', type=int, default=20,
                        help='Early stopping patience')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers')
    parser.add_argument('--sequence_length', type=int, default=1000,
                        help='Training sequence length for ChordFormer (frames)')
    parser.add_argument('--no_augment', action='store_true',
                        help='Disable data augmentation for training')
    parser.add_argument('--use_amp', action='store_true',
                        help='[AMP] Enable mixed precision training (faster on T4/V100/A100)')
    args = parser.parse_args()

    # Set defaults based on model type
    if args.batch_size is None:
        args.batch_size = config.BATCH_SIZE if args.model_type == 'chordformer' else 64

    # Learning rate scaling (linear scaling rule)
    # Reference: Goyal et al. "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour"
    # LR scales linearly with batch size: lr = base_lr * (batch_size / reference_batch_size)
    if args.lr is None:
        if args.model_type == 'chordformer':
            base_lr = config.BASE_LEARNING_RATE
            ref_batch = config.REFERENCE_BATCH_SIZE
            args.lr = base_lr * (args.batch_size / ref_batch)
            if args.batch_size != ref_batch:
                print(f"Linear LR scaling: batch_size={args.batch_size} (ref={ref_batch}) → LR={args.lr:.2e} (base={base_lr:.0e})")
        else:
            args.lr = 1e-4

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ===================== ChordFormer Training Path =====================
    if args.model_type == 'chordformer':
        print("Loading ChordFormer data...")
        train_loader, val_loader, test_loader, head_sizes = get_chordformer_dataloaders(
            args.features_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sequence_length=args.sequence_length,
            augment_train=not args.no_augment
        )

        # Create ChordFormer model
        model_config = {
            'n_bins': config.N_BINS_CHORDFORMER,
            'bins_per_octave': config.BINS_PER_OCTAVE_CHORDFORMER,
            'd_model': args.hidden_size,
            'n_heads': config.CONFORMER_HEADS,
            'd_ff': config.CONFORMER_FF_DIM,
            'n_layers': config.CONFORMER_LAYERS,
            'conv_kernel_size': config.CONFORMER_CONV_KERNEL,
            'dropout': args.dropout
        }

        print(f"Creating ChordFormer model...")
        print(f"  Conformer dim: {args.hidden_size}")
        print(f"  Heads: {config.CONFORMER_HEADS}, Layers: {config.CONFORMER_LAYERS}")

        model = ChordFormerModel(
            n_bins=config.N_BINS_CHORDFORMER,
            d_model=args.hidden_size,
            n_heads=config.CONFORMER_HEADS,
            d_ff=config.CONFORMER_FF_DIM,
            n_layers=config.CONFORMER_LAYERS,
            conv_kernel_size=config.CONFORMER_CONV_KERNEL,
            dropout=args.dropout
        )
        model = model.to(device)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")

        # Compute class weights for re-weighting
        class_weights = compute_class_weights_from_data(
            train_loader,
            head_sizes=head_sizes,
            gamma=config.REWEIGHT_GAMMA,
            w_max=config.REWEIGHT_MAX
        )
        # Move weights to device
        for head in class_weights:
            if class_weights[head] is not None:
                class_weights[head] = class_weights[head].to(device)

        # Create multi-head loss
        criterion = ChordFormerLoss(class_weights=class_weights)

        # AdamW optimizer (as per ChordFormer paper)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=config.WEIGHT_DECAY
        )

        # LR scheduler: reduce by 90% after patience epochs (monitors validation loss)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',  # Minimize validation loss (as per ChordFormer paper)
            factor=config.LR_SCHEDULER_FACTOR,  # 0.1 = 90% reduction
            patience=config.LR_SCHEDULER_PATIENCE,
            min_lr=config.LR_SCHEDULER_MIN_LR
        )

        # [AMP] Create mixed precision helper (None if disabled)
        amp = AMPHelper(enabled=args.use_amp) if args.use_amp else None

        # Training loop for ChordFormer
        best_val_acc = 0.0
        epochs_without_improvement = 0
        training_history = []

        print(f"\nStarting ChordFormer training for {args.epochs} epochs...")
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")

            # Train
            train_loss, train_accs = train_epoch_chordformer(
                model, train_loader, criterion, optimizer, device,
                max_grad_norm=config.MAX_GRAD_NORM,
                amp=amp  # [AMP] Pass helper (None if disabled)
            )

            # Validate
            val_loss, val_accs = validate_chordformer(model, val_loader, criterion, device)

            # Print metrics
            print(f"Train Loss: {train_loss:.4f}")
            print(f"  Train Accs: root_triad={train_accs['root_triad']:.4f}, bass={train_accs['bass']:.4f}, 7th={train_accs['7th']:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"  Val Accs: root_triad={val_accs['root_triad']:.4f}, bass={val_accs['bass']:.4f}, 7th={val_accs['7th']:.4f}")

            # Use root_triad accuracy as main metric for early stopping/checkpointing
            val_acc = val_accs['root_triad']

            # Learning rate scheduling (based on validation loss, as per paper)
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Learning rate: {current_lr:.6f}")

            # Check for LR-based stopping
            if current_lr < config.LR_SCHEDULER_MIN_LR:
                print(f"Learning rate below minimum ({config.LR_SCHEDULER_MIN_LR}), stopping training")
                break

            # Save training history
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_accs': train_accs,
                'val_loss': val_loss,
                'val_accs': val_accs,
                'lr': current_lr
            })

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_without_improvement = 0
                save_checkpoint(
                    model, optimizer, epoch, val_acc,
                    checkpoint_dir / 'best_model.pth',
                    model_type='chordformer',
                    model_config=model_config
                )
                print(f"New best root_triad accuracy: {val_acc:.4f}")
            else:
                epochs_without_improvement += 1
                print(f"No improvement for {epochs_without_improvement} epoch(s)")

            # Early stopping
            if epochs_without_improvement >= args.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

        # Save final model
        save_checkpoint(
            model, optimizer, epoch, val_acc,
            checkpoint_dir / 'final_model.pth',
            model_type='chordformer',
            model_config=model_config
        )

        # Save training history
        with open(checkpoint_dir / 'training_history.json', 'w') as f:
            json.dump(training_history, f, indent=2)

        print(f"\nTraining complete!")
        print(f"Best root_triad accuracy: {best_val_acc:.4f}")

        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_loss, test_accs = validate_chordformer(model, test_loader, criterion, device)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accs: root_triad={test_accs['root_triad']:.4f}, bass={test_accs['bass']:.4f}")

        # Save test results
        test_results = {
            'test_loss': test_loss,
            'test_accs': test_accs,
            'best_val_acc': best_val_acc
        }
        with open(checkpoint_dir / 'test_results.json', 'w') as f:
            json.dump(test_results, f, indent=2)

        return  # Exit after ChordFormer training

    # ===================== MIREX Training Path =====================
    if args.model_type == 'mirex':
        print("Loading MIREX data...")
        train_loader, val_loader, test_loader, head_sizes = get_mirex_dataloaders(
            args.features_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sequence_length=args.sequence_length,
            augment_train=not args.no_augment,
            noise_prob=config.AUGMENT_NOISE_PROB,
            time_stretch_prob=config.AUGMENT_TIME_STRETCH_PROB
        )

        # Create MIREX model
        model_config = {
            'n_bins': config.N_BINS_CHORDFORMER,
            'bins_per_octave': config.BINS_PER_OCTAVE_CHORDFORMER,
            'd_model': args.hidden_size,
            'n_heads': config.CONFORMER_HEADS,
            'd_ff': config.CONFORMER_FF_DIM,
            'n_layers': config.CONFORMER_LAYERS,
            'conv_kernel_size': config.CONFORMER_CONV_KERNEL,
            'dropout': args.dropout,
            'octavewise_n_filters': config.OCTAVEWISE_N_FILTERS,
            'num_keys': config.MIREX_NUM_KEYS,
            'num_degrees': config.MIREX_NUM_DEGREES,
            'num_bass': config.MIREX_NUM_BASS,
            'num_pitches': config.MIREX_NUM_PITCHES
        }

        print(f"Creating MIREX ChordFormer model...")
        print(f"  Conformer dim: {args.hidden_size}")
        print(f"  OctavewiseConv filters: {config.OCTAVEWISE_N_FILTERS}")
        print(f"  Output heads: key={config.MIREX_NUM_KEYS}, degree={config.MIREX_NUM_DEGREES}, bass={config.MIREX_NUM_BASS}")

        model = MIREXChordFormerModel(
            n_bins=config.N_BINS_CHORDFORMER,
            d_model=args.hidden_size,
            n_heads=config.CONFORMER_HEADS,
            d_ff=config.CONFORMER_FF_DIM,
            n_layers=config.CONFORMER_LAYERS,
            conv_kernel_size=config.CONFORMER_CONV_KERNEL,
            dropout=args.dropout,
            octavewise_n_filters=config.OCTAVEWISE_N_FILTERS,
            num_keys=config.MIREX_NUM_KEYS,
            num_degrees=config.MIREX_NUM_DEGREES,
            num_bass=config.MIREX_NUM_BASS,
            num_pitches=config.MIREX_NUM_PITCHES
        )
        model = model.to(device)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")

        # Compute class weights for re-weighting (categorical heads only)
        class_weights = compute_mirex_class_weights(
            train_loader,
            head_sizes=head_sizes,
            gamma=config.REWEIGHT_GAMMA,
            w_max=config.REWEIGHT_MAX
        )
        # Move weights to device
        for head in class_weights:
            if class_weights[head] is not None:
                class_weights[head] = class_weights[head].to(device)

        # Create MIREX loss
        criterion = MIREXLoss(
            class_weights=class_weights,
            ce_weight=config.MIREX_CE_WEIGHT,
            bce_weight=config.MIREX_BCE_WEIGHT
        )

        # AdamW optimizer (as per MIREX paper)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=config.WEIGHT_DECAY
        )

        # LR scheduler: reduce by 90% after patience epochs
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.LR_SCHEDULER_FACTOR,
            patience=config.LR_SCHEDULER_PATIENCE,
            min_lr=config.LR_SCHEDULER_MIN_LR
        )

        # Mixed precision helper
        amp = AMPHelper(enabled=args.use_amp) if args.use_amp else None

        # Training loop for MIREX
        best_val_acc = 0.0
        epochs_without_improvement = 0
        training_history = []

        print(f"\nStarting MIREX training for {args.epochs} epochs...")

        # Debug: Print memory and config before training
        print(f"\n=== DEBUG INFO ===")
        print(f"Batch size: {args.batch_size}")
        print(f"Sequence length: {config.SEQUENCE_LENGTH}")
        print(f"OctavewiseConv filters: {config.OCTAVEWISE_N_FILTERS}")
        print(f"Device: {device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"Memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        print(f"==================\n")

        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")

            # Train
            train_loss, train_accs = train_epoch_mirex(
                model, train_loader, criterion, optimizer, device,
                max_grad_norm=config.MAX_GRAD_NORM,
                amp=amp
            )

            # Validate
            val_loss, val_accs = validate_mirex(model, val_loader, criterion, device)

            # Print metrics
            print(f"Train Loss: {train_loss:.4f}")
            print(f"  Train Accs: key={train_accs['key']:.4f}, degree={train_accs['degree']:.4f}, bass={train_accs['bass']:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"  Val Accs: key={val_accs['key']:.4f}, degree={val_accs['degree']:.4f}, bass={val_accs['bass']:.4f}")

            # Debug: GPU memory after epoch
            if torch.cuda.is_available():
                print(f"  GPU Memory: allocated={torch.cuda.memory_allocated() / 1024**3:.2f} GB, "
                      f"reserved={torch.cuda.memory_reserved() / 1024**3:.2f} GB, "
                      f"max_allocated={torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

            # Use degree accuracy as main metric (most important for chord quality)
            val_acc = val_accs['degree']

            # Learning rate scheduling
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Learning rate: {current_lr:.6f}")

            # Check for LR-based stopping
            if current_lr < config.LR_SCHEDULER_MIN_LR:
                print(f"Learning rate below minimum ({config.LR_SCHEDULER_MIN_LR}), stopping training")
                break

            # Save training history
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_accs': train_accs,
                'val_loss': val_loss,
                'val_accs': val_accs,
                'lr': current_lr
            })

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_without_improvement = 0
                save_checkpoint(
                    model, optimizer, epoch, val_acc,
                    checkpoint_dir / 'best_model.pth',
                    model_type='mirex',
                    model_config=model_config
                )
                print(f"New best degree accuracy: {val_acc:.4f}")
            else:
                epochs_without_improvement += 1
                print(f"No improvement for {epochs_without_improvement} epoch(s)")

            # Early stopping
            if epochs_without_improvement >= args.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

        # Save final model
        save_checkpoint(
            model, optimizer, epoch, val_acc,
            checkpoint_dir / 'final_model.pth',
            model_type='mirex',
            model_config=model_config
        )

        # Save training history
        with open(checkpoint_dir / 'training_history.json', 'w') as f:
            json.dump(training_history, f, indent=2)

        print(f"\nTraining complete!")
        print(f"Best degree accuracy: {best_val_acc:.4f}")

        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_loss, test_accs = validate_mirex(model, test_loader, criterion, device)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accs: key={test_accs['key']:.4f}, degree={test_accs['degree']:.4f}, bass={test_accs['bass']:.4f}")

        # Save test results
        test_results = {
            'test_loss': test_loss,
            'test_accs': test_accs,
            'best_val_acc': best_val_acc
        }
        with open(checkpoint_dir / 'test_results.json', 'w') as f:
            json.dump(test_results, f, indent=2)

        return  # Exit after MIREX training

    # ===================== Legacy Training Path (BiLSTM/TCN) =====================
    print("Loading data...")
    train_loader, val_loader, test_loader, num_classes = get_dataloaders(
        args.features_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Create model
    print(f"Creating {args.model_type.upper()} model with {num_classes} classes...")
    if args.model_type == 'tcn':
        model = ChordRecognitionModelTCN(
            num_classes=num_classes,
            tcn_channels=args.hidden_size,
            dropout=args.dropout
        )
    else:
        model = ChordRecognitionModel(
            num_classes=num_classes,
            hidden_size=args.hidden_size,
            dropout=args.dropout
        )
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.1)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)

    # Learning rate scheduler - gentle decay to allow continued learning
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',  # Maximize validation accuracy
        factor=0.7,  # Reduce LR by 30% when plateau
        patience=7,  # Wait 7 epochs before reducing
        min_lr=1e-6
    )

    # Training loop
    best_val_acc = 0.0
    epochs_without_improvement = 0
    training_history = []

    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Learning rate scheduling
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {current_lr:.6f}")

        # Save training history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': current_lr
        })

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            save_checkpoint(
                model, optimizer, epoch, val_acc,
                checkpoint_dir / 'best_model.pth',
                model_type=args.model_type,
                num_classes=num_classes,
                hidden_size=args.hidden_size
            )
            print(f"New best validation accuracy: {val_acc:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s)")

        # Early stopping
        if epochs_without_improvement >= args.early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

    # Save final model
    save_checkpoint(
        model, optimizer, epoch, val_acc,
        checkpoint_dir / 'final_model.pth',
        model_type=args.model_type,
        num_classes=num_classes,
        hidden_size=args.hidden_size
    )

    # Save training history
    with open(checkpoint_dir / 'training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)

    print(f"\nTraining complete!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    # Save test results
    test_results = {
        'test_loss': test_loss,
        'test_acc': test_acc,
        'best_val_acc': best_val_acc
    }
    with open(checkpoint_dir / 'test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)


if __name__ == '__main__':
    main()
