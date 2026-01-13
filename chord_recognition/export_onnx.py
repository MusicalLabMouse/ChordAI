"""
Export ChordFormer model to ONNX format for browser inference.
"""

import argparse
import json
import torch
from pathlib import Path

from model import ChordFormerModel
import config


def export_to_onnx(checkpoint_path, output_path, opset_version=18):
    """
    Export ChordFormer model to ONNX format.

    Args:
        checkpoint_path: Path to the .pth checkpoint file
        output_path: Path for the output .onnx file
        opset_version: ONNX opset version (default 15 for browser compatibility)
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model_type = checkpoint.get('model_type', 'bilstm')
    if model_type != 'chordformer':
        raise ValueError(f"Expected chordformer model, got {model_type}")

    # Get model configuration from checkpoint
    model_config = checkpoint.get('config', {})

    # Create model with same configuration
    model = ChordFormerModel(
        n_bins=model_config.get('n_bins', config.N_BINS_CHORDFORMER),
        d_model=model_config.get('d_model', config.CONFORMER_DIM),
        n_heads=model_config.get('n_heads', config.CONFORMER_HEADS),
        d_ff=model_config.get('d_ff', config.CONFORMER_FF_DIM),
        n_layers=model_config.get('n_layers', config.CONFORMER_LAYERS),
        conv_kernel_size=model_config.get('conv_kernel_size', config.CONFORMER_CONV_KERNEL),
        dropout=0.0  # No dropout for inference
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded: ChordFormer with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create dummy input for tracing
    # Shape: [batch, time, n_bins]
    batch_size = 1
    seq_len = 100  # Example sequence length
    n_bins = model_config.get('n_bins', config.N_BINS_CHORDFORMER)

    dummy_input = torch.randn(batch_size, seq_len, n_bins)

    # Define input/output names
    input_names = ['features']
    output_names = ['root_triad', 'bass', '7th', '9th', '11th', '13th']

    # Define dynamic axes for variable batch and sequence length
    dynamic_axes = {
        'features': {0: 'batch', 1: 'time'},
        'root_triad': {0: 'batch', 1: 'time'},
        'bass': {0: 'batch', 1: 'time'},
        '7th': {0: 'batch', 1: 'time'},
        '9th': {0: 'batch', 1: 'time'},
        '11th': {0: 'batch', 1: 'time'},
        '13th': {0: 'batch', 1: 'time'},
    }

    print(f"Exporting to ONNX (opset {opset_version})...")

    # Custom forward wrapper to return tuple instead of dict
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            outputs = self.model(x, lengths=None)
            return (
                outputs['root_triad'],
                outputs['bass'],
                outputs['7th'],
                outputs['9th'],
                outputs['11th'],
                outputs['13th']
            )

    wrapped_model = ModelWrapper(model)
    wrapped_model.eval()

    # Export to ONNX
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True,
    )

    print(f"ONNX model exported to {output_path}")

    # Consolidate external data into single file (for browser compatibility)
    try:
        import onnx
        from pathlib import Path

        print("Consolidating model into single file...")
        model = onnx.load(output_path)

        # Convert external data to internal
        from onnx.external_data_helper import convert_model_to_external_data
        onnx.save_model(
            model,
            output_path,
            save_as_external_data=False,  # Embed all data in the .onnx file
        )

        # Remove the .data file if it exists
        data_file = Path(output_path).with_suffix('.onnx.data')
        if data_file.exists():
            data_file.unlink()
            print(f"Removed external data file: {data_file}")

        print(f"ONNX model saved to {output_path} (single file)")

    except Exception as e:
        print(f"Warning: Could not consolidate model: {e}")
        print("The model may have external data files that need to be served alongside it.")

    # Verify the exported model
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification passed!")

        # Print model info
        print(f"\nModel inputs:")
        for inp in onnx_model.graph.input:
            print(f"  {inp.name}: {[d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]}")

        print(f"\nModel outputs:")
        for out in onnx_model.graph.output:
            print(f"  {out.name}: {[d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim]}")

    except ImportError:
        print("Note: Install 'onnx' package for model verification")

    return output_path


def create_model_config(output_dir):
    """
    Create model_config.json for the web application.
    """
    model_config = {
        "modelType": "chordformer",
        "sampleRate": config.SAMPLE_RATE,
        "hopLength": config.HOP_LENGTH,
        "nBins": config.N_BINS_CHORDFORMER,
        "binsPerOctave": config.BINS_PER_OCTAVE_CHORDFORMER,
        "fmin": 32.7,  # C1 frequency
        "outputs": {
            "rootTriad": config.NUM_ROOT_TRIAD,
            "bass": config.NUM_BASS,
            "7th": config.NUM_7TH,
            "9th": config.NUM_9TH,
            "11th": config.NUM_11TH,
            "13th": config.NUM_13TH
        },
        "mappings": {
            "roots": config.ROOT_NAMES,
            "triads": config.TRIAD_TYPES,
            "sevenths": config.SEVENTH_TYPES,
            "ninths": config.NINTH_TYPES,
            "elevenths": config.ELEVENTH_TYPES,
            "thirteenths": config.THIRTEENTH_TYPES
        }
    }

    config_path = Path(output_dir) / "model_config.json"
    with open(config_path, 'w') as f:
        json.dump(model_config, f, indent=2)

    print(f"Model config saved to {config_path}")
    return config_path


def main():
    parser = argparse.ArgumentParser(description='Export ChordFormer model to ONNX')
    parser.add_argument('--checkpoint', type=str,
                        default='../checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str,
                        default='../web/public/model/chord_model.onnx',
                        help='Output path for ONNX model')
    parser.add_argument('--opset', type=int, default=18,
                        help='ONNX opset version (18 is PyTorch default)')
    parser.add_argument('--config-dir', type=str,
                        default='../web/public/model',
                        help='Directory for model_config.json')
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Export model
    export_to_onnx(checkpoint_path, str(output_path), args.opset)

    # Create model config
    config_dir = Path(args.config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    create_model_config(config_dir)

    print("\nExport complete! Next steps:")
    print(f"1. Copy normalization.json to {config_dir}")
    print("2. Run 'npm install' in web directory")
    print("3. Run 'npm run dev' to start the development server")


if __name__ == '__main__':
    main()
