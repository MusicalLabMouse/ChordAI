"""
Configuration file for chord recognition system.
Centralizes all hyperparameters and settings.
"""

# Audio Processing Parameters
SAMPLE_RATE = 22050
HOP_LENGTH = 512

# Feature parameters (model-dependent)
# For BiLSTM/TCN: 84 bins (12 bins/octave)
# For ChordFormer: 252 bins (36 bins/octave)
N_BINS_LEGACY = 84
BINS_PER_OCTAVE_LEGACY = 12
N_BINS_CHORDFORMER = 252
BINS_PER_OCTAVE_CHORDFORMER = 36
N_OCTAVES = 7
FMIN_NOTE = 'C1'  # 32.7 Hz

# Model Architecture
MODEL_TYPE = 'chordformer'  # 'bilstm', 'tcn', or 'chordformer'
HIDDEN_SIZE = 256  # LSTM hidden size or TCN channels
DROPOUT = 0.1

# TCN-specific parameters (for real-time inference)
TCN_KERNEL_SIZE = 3
TCN_DILATIONS = [1, 2, 4, 8]  # Receptive field: ~31 frames (~2.9 seconds)

# ChordFormer / Conformer Architecture
CONFORMER_DIM = 256
CONFORMER_HEADS = 4
CONFORMER_FF_DIM = 1024
CONFORMER_LAYERS = 4
CONFORMER_CONV_KERNEL = 31

# ChordFormer Output Vocabulary (Structured Chord Representation)
# Root + Triad: N + 12 roots × 7 triads (maj, min, sus4, sus2, dim, aug, N-quality)
NUM_ROOT_TRIAD = 85  # 1 (N) + 12*7 = 85
# Bass note: N + 12 notes
NUM_BASS = 13
# 7th extension: N, 7, b7, bb7
NUM_7TH = 4
# 9th extension: N, 9, #9, b9
NUM_9TH = 4
# 11th extension: N, 11, #11
NUM_11TH = 3
# 13th extension: N, 13, b13
NUM_13TH = 3

# Root names (chromatic order)
ROOT_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
# Triad types
TRIAD_TYPES = ['maj', 'min', 'sus4', 'sus2', 'dim', 'aug']
# 7th types
SEVENTH_TYPES = ['N', '7', 'b7', 'bb7']
# 9th types
NINTH_TYPES = ['N', '9', '#9', 'b9']
# 11th types
ELEVENTH_TYPES = ['N', '11', '#11']
# 13th types
THIRTEENTH_TYPES = ['N', '13', 'b13']

# Class re-weighting parameters (for handling imbalanced data)
REWEIGHT_GAMMA = 0.5  # Balancing factor (0=uniform, 1=full inverse)
REWEIGHT_MAX = 10.0   # Maximum weight clamp

# Training Parameters
BATCH_SIZE = 24  # ChordFormer paper uses 24 segments per mini-batch
LEARNING_RATE = 1e-3  # ChordFormer uses 1e-3 with AdamW
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
SEQUENCE_LENGTH = 1000  # ~23 seconds per training segment

# Learning Rate Scheduler (ChordFormer: reduce by 90% after 5 epochs)
LR_SCHEDULER_FACTOR = 0.1  # Reduce to 10% (90% reduction)
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_MIN_LR = 1e-6  # Stop training when LR drops below this

# Gradient Clipping
MAX_GRAD_NORM = 5.0

# Data Split (train/val/test)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Temporal Smoothing
SMOOTHING_WINDOW_SIZE = 7  # frames (median filter)

# Paths
DEFAULT_DATA_DIRS = [
    '../training_datasets/training_data',
    '../training_datasets/training_data_set2'
]
DEFAULT_FEATURES_DIR = '../features'
DEFAULT_CHECKPOINT_DIR = '../checkpoints'
DEFAULT_WEB_DIR = '../web'

# ONNX Export Parameters
ONNX_OPSET_VERSION = 15  # Use 15 for broad browser compatibility
ONNX_INPUT_NAMES = ['features']
ONNX_OUTPUT_NAMES = ['predictions']
