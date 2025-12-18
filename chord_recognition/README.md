# AI-Driven Chord Recognition System

A complete PyTorch implementation of an automatic chord recognition system using CNN+BiLSTM architecture.

## Features

- **End-to-end pipeline**: Feature extraction, training, and inference
- **CQT-based features**: Constant-Q Transform chromagram (84 bins)
- **CNN+BiLSTM architecture**: Convolutional feature extraction + temporal modeling
- **Extended chord vocabulary**: Supports 60+ chord types (major, minor, 7ths, inversions, extensions)
- **GPU-optimized**: Designed for NVIDIA GTX 1070 (8GB VRAM)
- **Temporal smoothing**: Median filtering for stable predictions

## Architecture

```
Input: [batch, max_len, 84]
↓
CNN Feature Extractor (3 layers)
  - Conv2D + BatchNorm + ReLU + MaxPool + Dropout
  - Pools only on frequency axis
↓
BiLSTM (1 layer, bidirectional)
  - Hidden size: 256
  - Packed sequences for efficiency
↓
Fully Connected + Softmax
↓
Output: [batch, max_len, num_classes]
```

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Extract Features

Extract CQT features from all MP3 files and create frame-level labels:

```bash
cd chord_recognition
python feature_extraction.py --data_dir ../training_data --output_dir ../features
```

This will:
- Build chord vocabulary from all .lab files
- Extract CQT features (84 bins, hop_length=512)
- Apply log scaling and z-score normalization
- Save features and labels as .npy files
- Create 80/10/10 train/val/test split

**Expected output:**
```
features/
├── chord_to_idx.json
├── idx_to_chord.json
├── data_split.json
├── normalization.json
└── 0003/, 0012/, ... (468 song directories)
    ├── features.npy
    └── labels.npy
```

### Step 2: Train Model

Train the CNN+BiLSTM model:

```bash
python train.py --features_dir ../features --epochs 100 --batch_size 8 --lr 1e-4
```

**Training features:**
- Adam optimizer with ReduceLROnPlateau scheduler
- Early stopping (patience=10 epochs)
- Gradient clipping (max_norm=5.0)
- Automatic checkpointing of best model
- Progress tracking with tqdm

**Expected training time:** 4-8 hours on GTX 1070

### Step 3: Run Inference

Predict chords from a new audio file:

```bash
python predict.py input.mp3 --output output.lab --checkpoint ../checkpoints/best_model.pth
```

**Output format (.lab file):**
```
0.000000    0.162000    N
0.162000    2.415000    A:maj
2.415000    4.668000    D:min
4.668000    6.921000    G:maj
...
```

## Model Performance

- **Frame-level accuracy:** 60-75% (typical for extended vocabulary)
- **Training time:** ~4-8 hours (100 epochs with early stopping)
- **Inference speed:** 10-20x faster than real-time
- **GPU memory:** ~2-3 GB (batch_size=8)

## Dataset

- **Source:** McGill Billboard Dataset
- **Total songs:** 468
- **Format:** MP3 audio + .lab annotations
- **Chord vocabulary:** 60+ classes (major, minor, 7ths, inversions, extensions, N)
- **Split:** 374 train, 47 validation, 47 test

## Configuration

All hyperparameters are defined in `config.py`:
- Audio parameters: sample rate, hop length, CQT bins
- Model architecture: hidden size, dropout
- Training: batch size, learning rate, epochs
- Temporal smoothing: median filter window size

## File Structure

```
chord_recognition/
├── feature_extraction.py   # Extract CQT features from audio
├── dataset.py              # PyTorch Dataset and DataLoader
├── model.py                # CNN+BiLSTM model architecture
├── train.py                # Training loop with validation
├── predict.py              # Inference with temporal smoothing
├── config.py               # Configuration and hyperparameters
├── requirements.txt        # Python dependencies
└── README.md               # This file

features/                   # Pre-computed features (created by feature_extraction.py)
checkpoints/                # Model checkpoints (created by train.py)
training_data/              # Original dataset (MP3 + .lab files)
```

## Implementation Details

### Audio Processing
- **Sample rate:** 22050 Hz
- **CQT bins:** 84 (7 octaves × 12 bins/octave)
- **Hop length:** 512 samples (~23.2 ms per frame)
- **Log scaling:** `log(|CQT| + 1e-6)` for numerical stability
- **Normalization:** Z-score using training set statistics

### Model Architecture
- **CNN:** 3 layers (32→64→128 channels)
- **BiLSTM:** 1 layer, hidden_size=256, bidirectional
- **Dropout:** 0.2 throughout network
- **Total parameters:** ~2-3 million

### Training Strategy
- **Loss:** CrossEntropyLoss (ignores padding)
- **Optimizer:** Adam (lr=1e-4, no weight decay)
- **LR Scheduler:** ReduceLROnPlateau (factor=0.95, patience=3)
- **Early Stopping:** 10 epochs without validation improvement
- **Gradient Clipping:** max_norm=5.0

### Inference
- **Temporal Smoothing:** Median filter (window_size=7 frames)
- **Segment Conversion:** Group consecutive identical predictions
- **Output:** Time-aligned .lab file

## Improvements from State-of-the-Art

Based on BTC-ISMIR19 (Bi-Directional Transformer for Chord Recognition):

1. **Numerical stability:** `log(|CQT| + 1e-6)` instead of `log(1 + C*|CQT|)`
2. **Feature normalization:** Z-score normalization from training set
3. **Dropout regularization:** 0.2 dropout throughout network
4. **Adaptive LR scheduling:** ReduceLROnPlateau for stable training
5. **Early stopping:** Prevents overfitting
6. **Gradient clipping:** Prevents exploding gradients
7. **Lower learning rate:** 1e-4 for more stable convergence

Note: The reference uses Transformer architecture, but this implementation uses CNN+BiLSTM as required by the specification.

## Troubleshooting

### Out of Memory (OOM) errors
- Reduce `batch_size` (try 4 or 2)
- Reduce `hidden_size` (try 128 or 192)

### Poor accuracy
- Train for more epochs
- Adjust learning rate
- Check data alignment (features vs labels)

### Slow training
- Ensure CUDA is available: `torch.cuda.is_available()`
- Use `num_workers > 0` in DataLoader (Windows: keep at 0)

## Citation

If you use this code, please cite the McGill Billboard Dataset:

```
Burgoyne, J. A., Wild, J., & Fujinaga, I. (2011).
An Expert Ground Truth Set for Audio Chord Recognition and Music Analysis.
In Proceedings of the 12th International Society for Music Information Retrieval Conference (ISMIR).
```

## License

MIT License
