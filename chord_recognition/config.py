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
REFERENCE_BATCH_SIZE = 24  # Reference batch size for LR scaling (paper default)
BASE_LEARNING_RATE = 1e-3  # ChordFormer uses 1e-3 with AdamW at reference batch size
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

# CRF Decoding (Section III.F of ChordFormer paper)
# γ parameter from equation (12) - penalty for chord transitions
# Higher values = smoother predictions (fewer chord changes)
TRANSITION_PENALTY = 1.0

# Data Split (train/val/test) - ChordFormer paper uses 60/20/20
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

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

# =============================================================================
# MIREX 2025 Degree-Based Chord Recognition Configuration
# Paper: "Degree-Based Automatic Chord Recognition with Enharmonic Distinction"
# =============================================================================

# MIREX mode toggle
USE_MIREX_MODE = False  # Set to True to use MIREX degree-based model

# Key confidence threshold - NOTE: Project-specific, not from MIREX paper
MIN_KEY_CONFIDENCE = 0.7

# Scale Degree Vocabulary (18 classes for full enharmonic distinction)
NUM_SCALE_DEGREES = 18
SCALE_DEGREES = [
    'N',      # 0: No chord
    'I',      # 1: Tonic (0 semitones)
    '#I',     # 2: Raised tonic (1 semitone, sharp spelling)
    'bII',    # 3: Flat supertonic (1 semitone, flat spelling)
    'II',     # 4: Supertonic (2 semitones)
    '#II',    # 5: Raised supertonic (3 semitones, sharp spelling)
    'bIII',   # 6: Flat mediant (3 semitones, flat spelling)
    'III',    # 7: Mediant (4 semitones)
    'IV',     # 8: Subdominant (5 semitones)
    '#IV',    # 9: Raised subdominant (6 semitones, sharp spelling)
    'bV',     # 10: Flat dominant (6 semitones, flat spelling)
    'V',      # 11: Dominant (7 semitones)
    '#V',     # 12: Raised dominant (8 semitones, sharp spelling)
    'bVI',    # 13: Flat submediant (8 semitones, flat spelling)
    'VI',     # 14: Submediant (9 semitones)
    '#VI',    # 15: Raised submediant (10 semitones, sharp spelling)
    'bVII',   # 16: Flat leading tone (10 semitones, flat spelling)
    'VII',    # 17: Leading tone (11 semitones)
]

# Semitone to degree index mapping (for label generation)
SEMITONE_TO_DEGREE_SHARP = {0: 1, 1: 2, 2: 4, 3: 5, 4: 7, 5: 8, 6: 9, 7: 11, 8: 12, 9: 14, 10: 15, 11: 17}
SEMITONE_TO_DEGREE_FLAT = {0: 1, 1: 3, 2: 4, 3: 6, 4: 7, 5: 8, 6: 10, 7: 11, 8: 13, 9: 14, 10: 16, 11: 17}

# MIREX Output Head Sizes
MIREX_NUM_KEYS = 13          # N + 12 keys
MIREX_NUM_DEGREES = 18       # N + 17 degrees (with enharmonic distinction)
MIREX_NUM_BASS = 13          # N + 12 bass notes
MIREX_NUM_PITCHES = 12       # 12 pitch classes (for binary vectors)

# Octavewise Convolution Module - NOTE: n_filters not specified in paper, may need tuning
OCTAVEWISE_N_FILTERS = 16    # Increased from 4 for better feature extraction capacity
OCTAVEWISE_D_MODEL = 256     # Paper specifies 256-dim output

# MIREX Loss Weights
MIREX_CE_WEIGHT = 1.0        # Weight for categorical (CrossEntropy) losses
MIREX_BCE_WEIGHT = 1.0       # Weight for binary (BCE) losses - balanced with CE for better interval learning

# Data Augmentation per MIREX paper Section 3.2
# Paper: "each with a probability of 50%"
AUGMENT_NOISE_PROB = 0.5              # Gaussian noise probability
AUGMENT_TIME_STRETCH_PROB = 0.5       # Time-stretch probability (per MIREX paper Section 3.2)
AUGMENT_TIME_STRETCH_RANGE = (0.8, 1.2)  # Paper: "0.8x to 1.2x"

# Experimental augmentation (not in MIREX paper)
USE_EXPERIMENTAL_FREQ_MASKING = False  # Not mentioned in paper Section 3.2

# Chord Quality Intervals (semitones from root)
CHORD_INTERVALS = {
    # Triads
    'maj':     [0, 4, 7],              # P1, M3, P5
    'min':     [0, 3, 7],              # P1, m3, P5
    'dim':     [0, 3, 6],              # P1, m3, d5
    'aug':     [0, 4, 8],              # P1, M3, A5
    'sus4':    [0, 5, 7],              # P1, P4, P5
    'sus2':    [0, 2, 7],              # P1, M2, P5
    # 7th chords
    '7':       [0, 4, 7, 10],          # dominant 7th
    'maj7':    [0, 4, 7, 11],          # major 7th
    'min7':    [0, 3, 7, 10],          # minor 7th
    'dim7':    [0, 3, 6, 9],           # diminished 7th
    'hdim7':   [0, 3, 6, 10],          # half-diminished 7th
    'minmaj7': [0, 3, 7, 11],          # minor-major 7th
    # Extended chords
    '9':       [0, 4, 7, 10, 14],      # dominant 9
    'maj9':    [0, 4, 7, 11, 14],      # major 9
    'min9':    [0, 3, 7, 10, 14],      # minor 9
    '11':      [0, 4, 7, 10, 14, 17],  # dominant 11
    '13':      [0, 4, 7, 10, 14, 21],  # dominant 13
}

# Reverse mapping: interval pattern -> chord quality (for inference)
INTERVAL_TO_QUALITY = {
    (0, 4, 7): 'maj',
    (0, 3, 7): 'min',
    (0, 3, 6): 'dim',
    (0, 4, 8): 'aug',
    (0, 5, 7): 'sus4',
    (0, 2, 7): 'sus2',
    (0, 4, 7, 10): '7',
    (0, 4, 7, 11): 'maj7',
    (0, 3, 7, 10): 'min7',
    (0, 3, 6, 9): 'dim7',
    (0, 3, 6, 10): 'hdim7',
    (0, 3, 7, 11): 'minmaj7',
}

# Enharmonic Spelling Constants
MIREX_ROOT_TO_IDX = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
SHARP_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
FLAT_NAMES = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

KEY_SIGNATURES = {
    'C': 0, 'G': 1, 'D': 2, 'A': 3, 'E': 4, 'B': 5, 'F#': 6, 'Gb': -6,
    'Db': -5, 'Ab': -4, 'Eb': -3, 'Bb': -2, 'F': -1
}

DEGREE_SEMITONES = {
    'N': None, 'I': 0, '#I': 1, 'bII': 1, 'II': 2, '#II': 3, 'bIII': 3,
    'III': 4, 'IV': 5, '#IV': 6, 'bV': 6, 'V': 7, '#V': 8, 'bVI': 8,
    'VI': 9, '#VI': 10, 'bVII': 10, 'VII': 11
}