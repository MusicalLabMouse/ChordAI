"""
Configuration for MIREX Chord Estimation Model
Based on ChordFormer with Octavewise Convolution and Scale Degree estimation.
"""

# Audio Processing Parameters
SAMPLE_RATE = 22050
HOP_LENGTH = 512  # Finer temporal resolution than original model
N_OCTAVES = 7  # C1 to C8
BINS_PER_OCTAVE = 36  # High frequency resolution
N_BINS = N_OCTAVES * BINS_PER_OCTAVE  # 252 total bins
FMIN_NOTE = 'C1'  # 32.7 Hz

# Octavewise Convolution
OCTAVE_CONV_OUT_CHANNELS = 64
OCTAVE_CONV_KERNEL = BINS_PER_OCTAVE  # One octave
OCTAVE_CONV_STRIDE = BINS_PER_OCTAVE // 12  # One semitone = 3 bins

# Conformer Architecture
CONFORMER_DIM = 256
CONFORMER_HEADS = 4
CONFORMER_FF_DIM = 1024
CONFORMER_CONV_KERNEL = 31
CONFORMER_LAYERS = 4
DROPOUT = 0.1

# Output Dimensions (Scale Degree Representation)
NUM_KEYS = 13  # N + 12 keys (C, C#, D, ..., B)
NUM_SCALE_DEGREES = 13  # N + 12 degrees (I, #I/bII, II, ..., VII)
NUM_BASS_NOTES = 13  # N + 12 bass notes
NUM_PITCH_CLASSES = 12  # For pitch presence vectors

# Training Parameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15

# Learning Rate Scheduler
LR_SCHEDULER_FACTOR = 0.1
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_MIN_LR = 1e-6

# Data Augmentation
PITCH_SHIFT_RANGE = (-5, 6)  # Semitones
TIME_STRETCH_RANGE = (0.8, 1.2)
SPEC_AUGMENT_PROB = 0.5

# Paths
DEFAULT_DATA_DIRS = [
    '../training_datasets/training_data',
    '../training_datasets/training_data_set2'
]
DEFAULT_FEATURES_DIR = '../features_mirex'
DEFAULT_CHECKPOINT_DIR = '../checkpoints_mirex'

# ONNX Export
ONNX_OPSET_VERSION = 15

# Chord/Scale Mappings
NOTE_NAMES = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
SCALE_DEGREE_NAMES = ['I', 'bII', 'II', 'bIII', 'III', 'IV', '#IV', 'V', 'bVI', 'VI', 'bVII', 'VII']

# Enharmonic normalization (prefer flats except F#)
ENHARMONIC_MAP = {
    'A#': 'Bb',
    'C#': 'Db',
    'D#': 'Eb',
    'Gb': 'F#',
    'G#': 'Ab',
    'Cb': 'B',
    'Fb': 'E',
    'B#': 'C',
    'E#': 'F'
}
