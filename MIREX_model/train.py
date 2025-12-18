"""
Training script for MIREX Chord Estimation Model
Uses AdamW optimizer with learning rate scheduling as described in the MIREX paper.
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

import config
from model import ScaleDegreeChordModel, ChordEstimationLoss
from dataset import create_dataloaders


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_samples = 0
    loss_components = {'key': 0, 'degree': 0, 'bass': 0, 'pitch': 0}

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch in pbar:
        features = batch['features'].to(device)
        lengths = batch['lengths'].to(device)

        targets = {
            'key': batch['key'].to(device),
            'degree': batch['degree'].to(device),
            'bass': batch['bass'].to(device),
            'pitch_abs': batch['pitch_abs'].to(device),
            'pitch_root': batch['pitch_root'].to(device),
            'pitch_bass': batch['pitch_bass'].to(device)
        }

        optimizer.zero_grad()

        outputs = model(features, lengths)
        loss, losses = criterion(outputs, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size = features.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        for key in loss_components:
            loss_components[key] += losses[key].item() * batch_size

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / total_samples
    for key in loss_components:
        loss_components[key] /= total_samples

    return avg_loss, loss_components


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_samples = 0
    loss_components = {'key': 0, 'degree': 0, 'bass': 0, 'pitch': 0}

    # Accuracy tracking
    correct = {'key': 0, 'degree': 0, 'bass': 0}
    total_frames = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            features = batch['features'].to(device)
            lengths = batch['lengths'].to(device)

            targets = {
                'key': batch['key'].to(device),
                'degree': batch['degree'].to(device),
                'bass': batch['bass'].to(device),
                'pitch_abs': batch['pitch_abs'].to(device),
                'pitch_root': batch['pitch_root'].to(device),
                'pitch_bass': batch['pitch_bass'].to(device)
            }

            outputs = model(features, lengths)
            loss, losses = criterion(outputs, targets)

            batch_size = features.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            for key in loss_components:
                loss_components[key] += losses[key].item() * batch_size

            # Calculate accuracy (exclude padding and N frames)
            for key_name in ['key', 'degree', 'bass']:
                preds = outputs[key_name].argmax(dim=-1)
                mask = targets[key_name] >= 0  # Valid frames only

                correct[key_name] += ((preds == targets[key_name]) & mask).sum().item()

            total_frames += (targets['key'] >= 0).sum().item()

    avg_loss = total_loss / total_samples
    for key in loss_components:
        loss_components[key] /= total_samples

    accuracy = {k: v / max(total_frames, 1) for k, v in correct.items()}

    return avg_loss, loss_components, accuracy


def main():
    parser = argparse.ArgumentParser(description='Train MIREX Chord Estimation Model')
    parser.add_argument('--features_dir', type=str, default=config.DEFAULT_FEATURES_DIR,
                        help='Path to extracted features')
    parser.add_argument('--checkpoint_dir', type=str, default=config.DEFAULT_CHECKPOINT_DIR,
                        help='Path to save checkpoints')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=config.NUM_EPOCHS)
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE)
    parser.add_argument('--sequence_length', type=int, default=500,
                        help='Sequence length for training')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # Setup directories
    features_dir = Path(args.features_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create dataloaders
    print(f"\nLoading data from {features_dir}...")
    train_loader, val_loader, test_loader, normalization = create_dataloaders(
        features_dir,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        num_workers=4 if os.name != 'nt' else 0  # Windows compatibility
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Create model
    model = ScaleDegreeChordModel(
        n_bins=config.N_BINS,
        bins_per_octave=config.BINS_PER_OCTAVE,
        d_model=config.CONFORMER_DIM,
        n_heads=config.CONFORMER_HEADS,
        d_ff=config.CONFORMER_FF_DIM,
        n_layers=config.CONFORMER_LAYERS,
        conv_kernel_size=config.CONFORMER_CONV_KERNEL,
        dropout=config.DROPOUT,
        num_keys=config.NUM_KEYS,
        num_degrees=config.NUM_SCALE_DEGREES,
        num_bass=config.NUM_BASS_NOTES
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Loss function
    criterion = ChordEstimationLoss(
        key_weight=1.0,
        degree_weight=1.0,
        bass_weight=0.5,
        pitch_weight=0.3
    )

    # Optimizer (AdamW as per MIREX paper)
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=config.WEIGHT_DECAY
    )

    # Learning rate scheduler (reduce by 90% if no improvement for 5 epochs)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.LR_SCHEDULER_FACTOR,
        patience=config.LR_SCHEDULER_PATIENCE,
        min_lr=config.LR_SCHEDULER_MIN_LR,
        verbose=True
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume:
        print(f"\nResuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'lr': []
    }

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 60)

    patience_counter = 0

    for epoch in range(start_epoch, args.epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch + 1}/{args.epochs} (LR: {current_lr:.2e})")

        # Check if LR is below minimum (early stopping per MIREX paper)
        if current_lr < config.LR_SCHEDULER_MIN_LR:
            print(f"\nLearning rate below minimum ({config.LR_SCHEDULER_MIN_LR}). Stopping training.")
            break

        # Train
        train_loss, train_losses = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_losses, val_accuracy = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step(val_loss)

        # Log results
        print(f"Train Loss: {train_loss:.4f} (key: {train_losses['key']:.4f}, "
              f"degree: {train_losses['degree']:.4f}, bass: {train_losses['bass']:.4f})")
        print(f"Val Loss: {val_loss:.4f} (key: {val_losses['key']:.4f}, "
              f"degree: {val_losses['degree']:.4f}, bass: {val_losses['bass']:.4f})")
        print(f"Val Accuracy - Key: {val_accuracy['key']:.4f}, "
              f"Degree: {val_accuracy['degree']:.4f}, Bass: {val_accuracy['bass']:.4f}")

        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['lr'].append(current_lr)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            checkpoint_path = checkpoint_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'val_accuracy': val_accuracy,
                'config': {
                    'n_bins': config.N_BINS,
                    'bins_per_octave': config.BINS_PER_OCTAVE,
                    'd_model': config.CONFORMER_DIM,
                    'n_heads': config.CONFORMER_HEADS,
                    'd_ff': config.CONFORMER_FF_DIM,
                    'n_layers': config.CONFORMER_LAYERS
                }
            }, checkpoint_path)
            print(f"Saved best model (val_loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch + 1}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss
            }, checkpoint_path)
            print(f"Saved checkpoint at epoch {epoch + 1}")

        # Early stopping check
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"\nNo improvement for {config.EARLY_STOPPING_PATIENCE} epochs. Early stopping.")
            break

    # Save training history
    with open(checkpoint_dir / 'training_history.json', 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        history_serializable = {
            'train_loss': [float(x) for x in history['train_loss']],
            'val_loss': [float(x) for x in history['val_loss']],
            'val_accuracy': [{k: float(v) for k, v in acc.items()} for acc in history['val_accuracy']],
            'lr': [float(x) for x in history['lr']]
        }
        json.dump(history_serializable, f, indent=2)

    # Save normalization stats with model
    with open(checkpoint_dir / 'normalization.json', 'w') as f:
        json.dump(normalization, f)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")


if __name__ == '__main__':
    main()
