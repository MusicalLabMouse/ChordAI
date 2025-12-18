"""
ONNX Export Script for Chord Recognition Model
Exports trained PyTorch model to ONNX format for browser deployment.
Supports BiLSTM, TCN, and ChordFormer models.
"""

import argparse
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from model import ChordRecognitionModel, ChordRecognitionModelTCN, ChordFormerModel
from dataset import CHORDFORMER_HEADS
import config


def load_model(checkpoint_path, device='cpu'):
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on

    Returns:
        model: Loaded model
        checkpoint: Checkpoint dictionary
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_type = checkpoint.get('model_type', 'bilstm')
    num_classes = checkpoint.get('num_classes', 25)
    hidden_size = checkpoint.get('hidden_size', 256)

    print(f"Loading {model_type.upper()} model...")

    if model_type == 'chordformer':
        model_config = checkpoint.get('config', {})
        model = ChordFormerModel(
            n_bins=model_config.get('n_bins', config.N_BINS_CHORDFORMER),
            d_model=model_config.get('d_model', config.CONFORMER_DIM),
            n_heads=model_config.get('n_heads', config.CONFORMER_HEADS),
            d_ff=model_config.get('d_ff', config.CONFORMER_FF_DIM),
            n_layers=model_config.get('n_layers', config.CONFORMER_LAYERS),
            conv_kernel_size=model_config.get('conv_kernel_size', config.CONFORMER_CONV_KERNEL),
            dropout=0.0  # Disable dropout for inference
        )
        print(f"  Conformer dim: {model_config.get('d_model', config.CONFORMER_DIM)}")
        print(f"  Output heads: {CHORDFORMER_HEADS}")
    elif model_type == 'tcn':
        model = ChordRecognitionModelTCN(
            num_classes=num_classes,
            tcn_channels=hidden_size,
            dropout=0.0  # Disable dropout for inference
        )
        print(f"  Classes: {num_classes}")
    else:
        model = ChordRecognitionModel(
            num_classes=num_classes,
            hidden_size=hidden_size,
            dropout=0.0  # Disable dropout for inference
        )
        print(f"  Classes: {num_classes}")

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, checkpoint


class ChordFormerONNXWrapper(nn.Module):
    """
    Wrapper to convert ChordFormerModel's dict output to tuple for ONNX export.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        outputs = self.model(x)
        # Return outputs in fixed order for ONNX
        return (
            outputs['root_triad'],
            outputs['bass'],
            outputs['7th'],
            outputs['9th'],
            outputs['11th'],
            outputs['13th']
        )


def export_to_onnx(model, output_path, model_type='tcn', num_frames=32):
    """
    Export model to ONNX format.

    Args:
        model: PyTorch model
        output_path: Path to save ONNX model
        model_type: Model architecture type
        num_frames: Number of frames for dummy input (can be dynamic)
    """
    model.eval()

    # Create dummy input based on model type
    batch_size = 1

    if model_type == 'chordformer':
        # ChordFormer uses 252 bins
        n_bins = config.N_BINS_CHORDFORMER
        dummy_input = torch.randn(batch_size, num_frames, n_bins)

        # Wrap model for tuple output
        wrapped_model = ChordFormerONNXWrapper(model)
        wrapped_model.eval()

        inputs = (dummy_input,)
        input_names = ['features']
        output_names = ['root_triad', 'bass', 'seventh', 'ninth', 'eleventh', 'thirteenth']
        dynamic_axes = {
            'features': {1: 'num_frames'},
            'root_triad': {1: 'num_frames'},
            'bass': {1: 'num_frames'},
            'seventh': {1: 'num_frames'},
            'ninth': {1: 'num_frames'},
            'eleventh': {1: 'num_frames'},
            'thirteenth': {1: 'num_frames'}
        }

        export_model = wrapped_model
    else:
        # Legacy models use 84 bins
        n_bins = config.N_BINS_LEGACY
        dummy_input = torch.randn(batch_size, num_frames, n_bins)

        # For BiLSTM, we also need lengths
        if model_type == 'bilstm':
            dummy_lengths = torch.tensor([num_frames], dtype=torch.long)
            inputs = (dummy_input, dummy_lengths)
            input_names = ['features', 'lengths']
            dynamic_axes = {
                'features': {1: 'num_frames'},
                'lengths': {},
                'predictions': {1: 'num_frames'}
            }
        else:
            inputs = (dummy_input,)
            input_names = ['features']
            dynamic_axes = {
                'features': {1: 'num_frames'},
                'predictions': {1: 'num_frames'}
            }

        output_names = ['predictions']
        export_model = model

    print(f"Exporting to ONNX with input shape: {dummy_input.shape}")

    # Export to a temporary path first
    temp_path = output_path.parent / 'temp_model.onnx'

    torch.onnx.export(
        export_model,
        inputs,
        str(temp_path),
        export_params=True,
        opset_version=config.ONNX_OPSET_VERSION,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=False  # Use legacy exporter for better compatibility
    )

    # Convert to single file (embed external data)
    import onnx
    print("Converting to single-file format...")
    onnx_model = onnx.load(str(temp_path))
    onnx.save_model(onnx_model, str(output_path), save_as_external_data=False)

    # Clean up temp files
    temp_path.unlink(missing_ok=True)
    temp_data = temp_path.parent / 'temp_model.onnx.data'
    if temp_data.exists():
        temp_data.unlink()

    print(f"Exported ONNX model to {output_path}")
    if model_type == 'chordformer':
        print(f"  Output heads: {output_names}")


def quantize_model(input_path, output_path):
    """
    Quantize ONNX model to int8 for faster inference.
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType

    print("Quantizing model to int8...")

    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QUInt8
    )

    # Get file sizes for comparison
    import os
    original_size = os.path.getsize(input_path) / 1024 / 1024
    quantized_size = os.path.getsize(output_path) / 1024 / 1024

    print(f"Original model: {original_size:.2f} MB")
    print(f"Quantized model: {quantized_size:.2f} MB")
    print(f"Size reduction: {(1 - quantized_size/original_size)*100:.1f}%")


def verify_onnx_model(onnx_path, dummy_input, pytorch_model, model_type):
    """
    Verify ONNX model produces same output as PyTorch model.

    Args:
        onnx_path: Path to ONNX model
        dummy_input: Dummy input tensor
        pytorch_model: Original PyTorch model
        model_type: Model architecture type
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("Warning: onnxruntime not installed. Skipping verification.")
        print("Install with: pip install onnxruntime")
        return

    print("Verifying ONNX model...")

    # Get PyTorch output
    pytorch_model.eval()
    with torch.no_grad():
        if model_type == 'bilstm':
            lengths = torch.tensor([dummy_input.shape[1]], dtype=torch.long)
            pytorch_output = pytorch_model(dummy_input, lengths)
        else:
            pytorch_output = pytorch_model(dummy_input)

    # Get ONNX output
    session = ort.InferenceSession(str(onnx_path))

    if model_type == 'bilstm':
        onnx_inputs = {
            'features': dummy_input.numpy(),
            'lengths': np.array([dummy_input.shape[1]], dtype=np.int64)
        }
    else:
        onnx_inputs = {'features': dummy_input.numpy()}

    onnx_output = session.run(None, onnx_inputs)[0]

    # Compare outputs
    pytorch_output_np = pytorch_output.numpy()
    max_diff = np.abs(pytorch_output_np - onnx_output).max()

    print(f"Maximum difference between PyTorch and ONNX: {max_diff:.6f}")

    if max_diff < 1e-4:
        print("Verification PASSED: ONNX model matches PyTorch model")
    else:
        print("Warning: Outputs differ more than expected. This may affect accuracy.")


def export_metadata(output_dir, checkpoint, features_dir):
    """
    Export metadata needed for web inference.

    Args:
        output_dir: Directory to save metadata
        checkpoint: Model checkpoint dictionary
        features_dir: Path to features directory (for normalization stats)
    """
    output_dir = Path(output_dir)

    # Copy chord mapping
    chord_to_idx_path = Path(features_dir) / 'chord_to_idx.json'
    if chord_to_idx_path.exists():
        with open(chord_to_idx_path, 'r') as f:
            chord_to_idx = json.load(f)

        # Create reverse mapping (idx to chord)
        idx_to_chord = {v: k for k, v in chord_to_idx.items()}

        with open(output_dir / 'chord_mapping.json', 'w') as f:
            json.dump({
                'chord_to_idx': chord_to_idx,
                'idx_to_chord': idx_to_chord,
                'num_classes': len(chord_to_idx)
            }, f, indent=2)

        print(f"Exported chord mapping to {output_dir / 'chord_mapping.json'}")

    # Copy normalization stats
    norm_path = Path(features_dir) / 'normalization.json'
    if norm_path.exists():
        with open(norm_path, 'r') as f:
            normalization = json.load(f)

        with open(output_dir / 'normalization.json', 'w') as f:
            json.dump(normalization, f, indent=2)

        print(f"Exported normalization stats to {output_dir / 'normalization.json'}")

    # Export model config
    model_type = checkpoint.get('model_type', 'bilstm')

    if model_type == 'chordformer':
        model_config = {
            'model_type': 'chordformer',
            'sample_rate': config.SAMPLE_RATE,
            'hop_length': config.HOP_LENGTH,
            'n_bins': config.N_BINS_CHORDFORMER,
            'bins_per_octave': config.BINS_PER_OCTAVE_CHORDFORMER,
            'frame_duration_ms': config.HOP_LENGTH / config.SAMPLE_RATE * 1000,
            'output_heads': {
                'root_triad': config.NUM_ROOT_TRIAD,
                'bass': config.NUM_BASS,
                '7th': config.NUM_7TH,
                '9th': config.NUM_9TH,
                '11th': config.NUM_11TH,
                '13th': config.NUM_13TH
            },
            'head_order': ['root_triad', 'bass', 'seventh', 'ninth', 'eleventh', 'thirteenth'],
            'root_names': config.ROOT_NAMES,
            'triad_types': config.TRIAD_TYPES
        }
    else:
        model_config = {
            'model_type': model_type,
            'num_classes': checkpoint.get('num_classes', 25),
            'hidden_size': checkpoint.get('hidden_size', 256),
            'sample_rate': config.SAMPLE_RATE,
            'hop_length': config.HOP_LENGTH,
            'n_bins': config.N_BINS_LEGACY,
            'frame_duration_ms': config.HOP_LENGTH / config.SAMPLE_RATE * 1000
        }

    with open(output_dir / 'model_config.json', 'w') as f:
        json.dump(model_config, f, indent=2)

    print(f"Exported model config to {output_dir / 'model_config.json'}")


def main():
    parser = argparse.ArgumentParser(description='Export chord recognition model to ONNX')
    parser.add_argument('--checkpoint', type=str, default='../checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='../web/public/model/chord_model.onnx',
                        help='Path for output ONNX model')
    parser.add_argument('--features_dir', type=str, default='../features',
                        help='Path to features directory (for normalization stats)')
    parser.add_argument('--num_frames', type=int, default=32,
                        help='Number of frames for test input')
    parser.add_argument('--quantize', action='store_true',
                        help='Quantize model to int8 for faster inference')
    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load model
    model, checkpoint = load_model(args.checkpoint)
    model_type = checkpoint.get('model_type', 'bilstm')

    # Export to ONNX
    export_to_onnx(model, output_path, model_type, args.num_frames)

    # Quantize if requested
    if args.quantize:
        quantized_path = output_path.parent / 'chord_model_quantized.onnx'
        quantize_model(output_path, quantized_path)
        # Replace original with quantized
        output_path.unlink()
        quantized_path.rename(output_path)
        print(f"Replaced with quantized model: {output_path}")

    # Export metadata
    export_metadata(output_path.parent, checkpoint, args.features_dir)

    print("\nExport complete!")
    print(f"ONNX model: {output_path}")
    print(f"Metadata: {output_path.parent}")


if __name__ == '__main__':
    main()
