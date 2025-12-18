"""
MIREX Chord Estimation Model
Based on ChordFormer with Conformer architecture, Octavewise Convolution,
and Scale Degree output representation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class OctavewiseConvolution(nn.Module):
    """
    Convolves across frequency bins with kernel size of one octave,
    sliding by one semitone. This captures intervallic and scale-related features.
    """
    def __init__(self, in_channels=1, out_channels=64, bins_per_octave=36):
        super().__init__()
        self.bins_per_octave = bins_per_octave
        kernel_size = bins_per_octave  # One octave
        stride = bins_per_octave // 12  # One semitone (3 bins for 36 bins/octave)

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()

    def forward(self, x):
        """
        Args:
            x: [batch, time, freq_bins] - CQT spectrogram
        Returns:
            out: [batch, time, out_channels * num_positions]
        """
        batch, time, freq = x.shape

        # Process each time frame independently
        # Reshape to [batch * time, 1, freq]
        x = x.reshape(batch * time, 1, freq)

        # Apply octavewise convolution
        x = self.conv(x)  # [batch * time, out_channels, positions]
        x = self.bn(x)
        x = self.activation(x)

        # Flatten and reshape back
        x = x.reshape(batch, time, -1)  # [batch, time, out_channels * positions]

        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ConformerConvModule(nn.Module):
    """
    Conformer convolution module with depthwise separable convolution.
    """
    def __init__(self, d_model, kernel_size=31, dropout=0.1):
        super().__init__()

        self.layer_norm = nn.LayerNorm(d_model)

        # Pointwise conv (expand)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)

        # Depthwise conv
        padding = (kernel_size - 1) // 2
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=padding, groups=d_model
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()

        # Pointwise conv (project)
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: [batch, time, d_model]
        Returns:
            out: [batch, time, d_model]
        """
        residual = x
        x = self.layer_norm(x)

        # [batch, time, d_model] -> [batch, d_model, time]
        x = x.transpose(1, 2)

        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)

        # [batch, d_model, time] -> [batch, time, d_model]
        x = x.transpose(1, 2)

        return residual + x


class ConformerFeedForward(nn.Module):
    """Feed-forward module with expansion factor."""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return residual + 0.5 * x  # Half-step residual


class ConformerBlock(nn.Module):
    """
    Conformer block: FFN -> Self-Attention -> Conv -> FFN
    """
    def __init__(self, d_model, n_heads, d_ff, conv_kernel_size=31, dropout=0.1):
        super().__init__()

        # First feed-forward (half-step)
        self.ff1 = ConformerFeedForward(d_model, d_ff, dropout)

        # Multi-head self-attention
        self.attn_layer_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attn_dropout = nn.Dropout(dropout)

        # Convolution module
        self.conv_module = ConformerConvModule(d_model, conv_kernel_size, dropout)

        # Second feed-forward (half-step)
        self.ff2 = ConformerFeedForward(d_model, d_ff, dropout)

        # Final layer norm
        self.final_layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        Args:
            x: [batch, time, d_model]
            mask: Optional attention mask
        Returns:
            out: [batch, time, d_model]
        """
        # First FFN
        x = self.ff1(x)

        # Self-attention
        residual = x
        x = self.attn_layer_norm(x)
        x, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = self.attn_dropout(x)
        x = residual + x

        # Convolution
        x = self.conv_module(x)

        # Second FFN
        x = self.ff2(x)

        # Final layer norm
        x = self.final_layer_norm(x)

        return x


class ScaleDegreeChordModel(nn.Module):
    """
    Chord estimation model using scale degree representation.

    Outputs:
    - Key: 13 classes (N + 12 keys)
    - Scale Degree: 13 classes (N + 12 degrees)
    - Bass: 13 classes (N + 12 bass notes)
    - Pitch presence vectors (optional, for richer representation)
    """
    def __init__(
        self,
        n_bins=252,
        bins_per_octave=36,
        d_model=256,
        n_heads=4,
        d_ff=1024,
        n_layers=4,
        conv_kernel_size=31,
        dropout=0.1,
        num_keys=13,
        num_degrees=13,
        num_bass=13
    ):
        super().__init__()

        self.n_bins = n_bins
        self.d_model = d_model

        # Octavewise convolution
        self.octave_conv = OctavewiseConvolution(
            in_channels=1,
            out_channels=config.OCTAVE_CONV_OUT_CHANNELS,
            bins_per_octave=bins_per_octave
        )

        # Calculate output dimension of octave conv
        # With 252 bins, kernel=36, stride=3: output positions = (252 - 36) // 3 + 1 = 73
        n_positions = (n_bins - bins_per_octave) // (bins_per_octave // 12) + 1
        octave_conv_dim = config.OCTAVE_CONV_OUT_CHANNELS * n_positions

        # Linear projection to model dimension
        self.input_projection = nn.Linear(octave_conv_dim, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        # Conformer layers
        self.conformer_layers = nn.ModuleList([
            ConformerBlock(d_model, n_heads, d_ff, conv_kernel_size, dropout)
            for _ in range(n_layers)
        ])

        # Output heads
        self.key_head = nn.Linear(d_model, num_keys)
        self.degree_head = nn.Linear(d_model, num_degrees)
        self.bass_head = nn.Linear(d_model, num_bass)

        # Optional: Pitch presence heads (12 dims each)
        self.pitch_abs_head = nn.Linear(d_model, 12)  # Absolute pitches
        self.pitch_root_head = nn.Linear(d_model, 12)  # Intervals from root
        self.pitch_bass_head = nn.Linear(d_model, 12)  # Intervals from bass

    def forward(self, x, lengths=None):
        """
        Args:
            x: [batch, time, n_bins] - CQT spectrogram
            lengths: Optional sequence lengths for masking

        Returns:
            dict with keys: 'key', 'degree', 'bass', 'pitch_abs', 'pitch_root', 'pitch_bass'
            Each value is [batch, time, num_classes]
        """
        # Octavewise convolution
        x = self.octave_conv(x)  # [batch, time, octave_conv_dim]

        # Project to model dimension
        x = self.input_projection(x)  # [batch, time, d_model]

        # Add positional encoding
        x = self.pos_encoding(x)

        # Create attention mask if lengths provided
        mask = None
        if lengths is not None:
            batch_size, max_len = x.shape[:2]
            mask = torch.arange(max_len, device=x.device).expand(batch_size, max_len)
            mask = mask >= lengths.unsqueeze(1)

        # Conformer layers
        for layer in self.conformer_layers:
            x = layer(x, mask)

        # Output predictions
        outputs = {
            'key': self.key_head(x),
            'degree': self.degree_head(x),
            'bass': self.bass_head(x),
            'pitch_abs': torch.sigmoid(self.pitch_abs_head(x)),
            'pitch_root': torch.sigmoid(self.pitch_root_head(x)),
            'pitch_bass': torch.sigmoid(self.pitch_bass_head(x))
        }

        return outputs

    def get_chord_predictions(self, outputs):
        """
        Convert model outputs to chord labels.

        Args:
            outputs: dict from forward()

        Returns:
            chords: List of chord strings
        """
        key_preds = outputs['key'].argmax(dim=-1)
        degree_preds = outputs['degree'].argmax(dim=-1)
        bass_preds = outputs['bass'].argmax(dim=-1)

        # Convert to chord labels
        # key + degree -> root note
        # This is done in post-processing
        return key_preds, degree_preds, bass_preds


class ChordEstimationLoss(nn.Module):
    """
    Multi-task loss for chord estimation.
    Combines cross-entropy for key/degree/bass with BCE for pitch vectors.
    """
    def __init__(self, key_weight=1.0, degree_weight=1.0, bass_weight=0.5, pitch_weight=0.3):
        super().__init__()
        self.key_weight = key_weight
        self.degree_weight = degree_weight
        self.bass_weight = bass_weight
        self.pitch_weight = pitch_weight

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.bce_loss = nn.BCELoss(reduction='none')

    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict from model forward
            targets: dict with 'key', 'degree', 'bass', 'pitch_abs', etc.

        Returns:
            total_loss, loss_dict
        """
        losses = {}

        # Classification losses
        losses['key'] = self.ce_loss(
            outputs['key'].reshape(-1, outputs['key'].shape[-1]),
            targets['key'].reshape(-1)
        )

        losses['degree'] = self.ce_loss(
            outputs['degree'].reshape(-1, outputs['degree'].shape[-1]),
            targets['degree'].reshape(-1)
        )

        losses['bass'] = self.ce_loss(
            outputs['bass'].reshape(-1, outputs['bass'].shape[-1]),
            targets['bass'].reshape(-1)
        )

        # Pitch presence losses (masked by valid frames)
        if 'pitch_abs' in targets:
            mask = (targets['key'].reshape(-1) != -1).float()

            pitch_loss = 0
            for key in ['pitch_abs', 'pitch_root', 'pitch_bass']:
                if key in targets:
                    pred = outputs[key].reshape(-1, 12)
                    tgt = targets[key].reshape(-1, 12)
                    loss = self.bce_loss(pred, tgt).mean(dim=-1)
                    loss = (loss * mask).sum() / (mask.sum() + 1e-8)
                    pitch_loss += loss

            losses['pitch'] = pitch_loss / 3
        else:
            losses['pitch'] = torch.tensor(0.0, device=outputs['key'].device)

        # Combine losses
        total_loss = (
            self.key_weight * losses['key'] +
            self.degree_weight * losses['degree'] +
            self.bass_weight * losses['bass'] +
            self.pitch_weight * losses['pitch']
        )

        return total_loss, losses


if __name__ == '__main__':
    # Test the model
    model = ScaleDegreeChordModel(
        n_bins=252,
        bins_per_octave=36,
        d_model=256,
        n_heads=4,
        d_ff=1024,
        n_layers=4
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Test forward pass
    batch_size = 2
    time_steps = 100
    x = torch.randn(batch_size, time_steps, 252)

    outputs = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Key output shape: {outputs['key'].shape}")
    print(f"Degree output shape: {outputs['degree'].shape}")
    print(f"Bass output shape: {outputs['bass'].shape}")
