"""
Export MIREX Chord Estimation Model to ONNX format.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

import config
from model import ScaleDegreeChordModel


class ScaleDegreeChordModelForExport(nn.Module):
    """
    Wrapper model for ONNX export.
    Converts scale degree outputs to absolute chord predictions.
    """
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model

    def forward(self, x):
        """
        Args:
            x: [batch, time, n_bins] - CQT spectrogram

        Returns:
            key_probs: [batch, time, 13] - Key probabilities
            degree_probs: [batch, time, 13] - Scale degree probabilities
            bass_probs: [batch, time, 13] - Bass note probabilities
            root_probs: [batch, time, 13] - Absolute root probabilities (derived)
        """
        outputs = self.model(x)

        # Get probabilities
        key_probs = torch.softmax(outputs['key'], dim=-1)
        degree_probs = torch.softmax(outputs['degree'], dim=-1)
        bass_probs = torch.softmax(outputs['bass'], dim=-1)

        # Compute absolute root probabilities from key and degree
        # root = (key + degree - 2) % 12 + 1 for non-N classes
        batch, time, _ = key_probs.shape

        # Create root probability matrix
        # For each (key, degree) combination, compute the resulting root
        root_probs = torch.zeros(batch, time, 13, device=x.device)

        # N class probability
        root_probs[:, :, 0] = key_probs[:, :, 0] + degree_probs[:, :, 0] - key_probs[:, :, 0] * degree_probs[:, :, 0]

        # For each key (1-12) and degree (1-12), compute root (1-12)
        for k in range(1, 13):
            for d in range(1, 13):
                root = (k - 1 + d - 1) % 12 + 1
                root_probs[:, :, root] += key_probs[:, :, k] * degree_probs[:, :, d]

        return key_probs, degree_probs, bass_probs, root_probs


def export_model(checkpoint_path, output_path, quantize=False):
    """Export model to ONNX format."""
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)

    print(f"Loading checkpoint from {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Get model config from checkpoint or use defaults
    model_config = checkpoint.get('config', {})

    # Create model
    model = ScaleDegreeChordModel(
        n_bins=model_config.get('n_bins', config.N_BINS),
        bins_per_octave=model_config.get('bins_per_octave', config.BINS_PER_OCTAVE),
        d_model=model_config.get('d_model', config.CONFORMER_DIM),
        n_heads=model_config.get('n_heads', config.CONFORMER_HEADS),
        d_ff=model_config.get('d_ff', config.CONFORMER_FF_DIM),
        n_layers=model_config.get('n_layers', config.CONFORMER_LAYERS),
        conv_kernel_size=config.CONFORMER_CONV_KERNEL,
        dropout=0.0,  # No dropout for inference
        num_keys=config.NUM_KEYS,
        num_degrees=config.NUM_SCALE_DEGREES,
        num_bass=config.NUM_BASS_NOTES
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Wrap for export
    export_model = ScaleDegreeChordModelForExport(model)
    export_model.eval()

    # Create dummy input
    # Use dynamic batch and time dimensions
    dummy_input = torch.randn(1, 100, config.N_BINS)

    # Export to ONNX
    print(f"Exporting to {output_path}")

    torch.onnx.export(
        export_model,
        dummy_input,
        str(output_path),
        opset_version=config.ONNX_OPSET_VERSION,
        input_names=['cqt_spectrogram'],
        output_names=['key_probs', 'degree_probs', 'bass_probs', 'root_probs'],
        dynamic_axes={
            'cqt_spectrogram': {0: 'batch', 1: 'time'},
            'key_probs': {0: 'batch', 1: 'time'},
            'degree_probs': {0: 'batch', 1: 'time'},
            'bass_probs': {0: 'batch', 1: 'time'},
            'root_probs': {0: 'batch', 1: 'time'}
        }
    )

    print(f"Model exported successfully!")

    # Get file size
    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"Model size: {file_size:.2f} MB")

    # Quantize if requested
    if quantize:
        quantized_path = output_path.parent / f"{output_path.stem}_quantized.onnx"
        quantize_model(output_path, quantized_path)

    return output_path


def quantize_model(input_path, output_path):
    """Quantize ONNX model to int8."""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType

        print(f"\nQuantizing model to {output_path}")

        quantize_dynamic(
            model_input=str(input_path),
            model_output=str(output_path),
            weight_type=QuantType.QUInt8
        )

        # Compare sizes
        original_size = Path(input_path).stat().st_size / (1024 * 1024)
        quantized_size = Path(output_path).stat().st_size / (1024 * 1024)
        reduction = (1 - quantized_size / original_size) * 100

        print(f"Original size: {original_size:.2f} MB")
        print(f"Quantized size: {quantized_size:.2f} MB")
        print(f"Size reduction: {reduction:.1f}%")

        return output_path

    except ImportError:
        print("Warning: onnxruntime.quantization not available. Skipping quantization.")
        return input_path


def verify_model(onnx_path, normalization_path=None):
    """Verify the exported ONNX model."""
    try:
        import onnxruntime as ort

        print(f"\nVerifying model: {onnx_path}")

        # Create session
        session = ort.InferenceSession(str(onnx_path))

        # Get input/output info
        inputs = session.get_inputs()
        outputs = session.get_outputs()

        print(f"\nInputs:")
        for inp in inputs:
            print(f"  {inp.name}: {inp.shape} ({inp.type})")

        print(f"\nOutputs:")
        for out in outputs:
            print(f"  {out.name}: {out.shape} ({out.type})")

        # Test inference
        dummy_input = np.random.randn(1, 50, config.N_BINS).astype(np.float32)

        # Apply normalization if available
        if normalization_path and Path(normalization_path).exists():
            with open(normalization_path, 'r') as f:
                norm = json.load(f)
            mean = np.array(norm['mean']).astype(np.float32)
            std = np.array(norm['std']).astype(np.float32)
            dummy_input = (dummy_input - mean) / std

        results = session.run(None, {'cqt_spectrogram': dummy_input})

        print(f"\nTest inference successful!")
        print(f"  Key probs shape: {results[0].shape}")
        print(f"  Degree probs shape: {results[1].shape}")
        print(f"  Bass probs shape: {results[2].shape}")
        print(f"  Root probs shape: {results[3].shape}")

        # Check probability sums
        print(f"\n  Key probs sum (should be ~1.0): {results[0][0, 0].sum():.4f}")
        print(f"  Degree probs sum (should be ~1.0): {results[1][0, 0].sum():.4f}")

        return True

    except Exception as e:
        print(f"Verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Export MIREX model to ONNX')
    parser.add_argument('--checkpoint', type=str,
                        default=str(Path(config.DEFAULT_CHECKPOINT_DIR) / 'best_model.pt'),
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str,
                        default=str(Path(config.DEFAULT_CHECKPOINT_DIR) / 'mirex_chord_model.onnx'),
                        help='Output ONNX path')
    parser.add_argument('--quantize', action='store_true',
                        help='Also export quantized version')
    parser.add_argument('--verify', action='store_true',
                        help='Verify the exported model')
    args = parser.parse_args()

    # Export
    output_path = export_model(args.checkpoint, args.output, args.quantize)

    # Verify
    if args.verify:
        norm_path = Path(config.DEFAULT_CHECKPOINT_DIR) / 'normalization.json'
        verify_model(output_path, norm_path)


if __name__ == '__main__':
    main()
