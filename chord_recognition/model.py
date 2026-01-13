"""
Chord Recognition Models
Supports: CNN+BiLSTM, CNN+TCN, and ChordFormer architectures.
"""

import math
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ChordRecognitionModel(nn.Module):
    """
    Chord recognition model with CNN feature extraction and BiLSTM temporal modeling.

    Architecture:
        Input: [batch, max_len, 84]
        → Reshape to [batch, 1, 84, max_len] for 2D convolution
        → CNN (3 layers with BatchNorm, ReLU, MaxPool, Dropout)
        → BiLSTM (1 layer with dropout)
        → Fully Connected → Softmax
        Output: [batch, max_len, num_classes]
    """

    def __init__(self, num_classes, hidden_size=256, dropout=0.2):
        """
        Args:
            num_classes: Number of chord classes
            hidden_size: Hidden size for BiLSTM
            dropout: Dropout probability
        """
        super(ChordRecognitionModel, self).__init__()

        self.num_classes = num_classes
        self.hidden_size = hidden_size

        # CNN Feature Extractor
        # Input: [batch, 1, 84, time_frames]

        # Conv layer 1: [batch, 1, 84, T] → [batch, 32, freq1, T]
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(12, 3), stride=1, padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((2, 1))  # Pool only on frequency axis
        self.dropout1 = nn.Dropout2d(dropout)

        # Conv layer 2: [batch, 32, freq1, T] → [batch, 64, freq2, T]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(6, 3), stride=1, padding=(0, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d((2, 1))  # Pool only on frequency axis
        self.dropout2 = nn.Dropout2d(dropout)

        # Conv layer 3: [batch, 64, freq2, T] → [batch, 128, freq3, T]
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=(0, 1))
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d((2, 1))  # Pool only on frequency axis
        self.dropout3 = nn.Dropout2d(dropout)

        # Calculate CNN output dimensions
        # Input freq: 84
        # After conv1 (kernel_size=(12,3), padding=(0,1)): 84 - 12 + 1 = 73
        # After pool1 (2,1): 73 // 2 = 36
        # After conv2 (kernel_size=(6,3), padding=(0,1)): 36 - 6 + 1 = 31
        # After pool2 (2,1): 31 // 2 = 15
        # After conv3 (kernel_size=(3,3), padding=(0,1)): 15 - 3 + 1 = 13
        # After pool3 (2,1): 13 // 2 = 6

        cnn_output_freq = 6
        cnn_output_channels = 128
        cnn_output_dim = cnn_output_channels * cnn_output_freq  # 128 * 6 = 768

        # BiLSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=cnn_output_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0  # Only 1 layer, so no dropout between layers
        )

        # Fully connected layer
        self.dropout_fc = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional

    def forward(self, x, lengths):
        """
        Forward pass.

        Args:
            x: Input features [batch, max_len, 84]
            lengths: Sequence lengths [batch]

        Returns:
            output: Chord predictions [batch, max_len, num_classes]
        """
        batch_size, max_len, _ = x.shape

        # Reshape for CNN: [batch, max_len, 84] → [batch, 1, 84, max_len]
        x = x.transpose(1, 2).unsqueeze(1)  # [batch, 1, 84, max_len]

        # CNN feature extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        # Reshape for LSTM: [batch, channels, freq, time] → [batch, time, channels*freq]
        x = x.permute(0, 3, 1, 2)  # [batch, time, channels, freq]
        x = x.contiguous().view(batch_size, max_len, -1)  # [batch, time, channels*freq]

        # Pack sequences for efficient LSTM processing
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)

        # BiLSTM
        packed_output, _ = self.lstm(packed_input)

        # Unpack sequences
        x, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=max_len)

        # Fully connected layer
        x = self.dropout_fc(x)
        output = self.fc(x)  # [batch, max_len, num_classes]

        return output


class CausalConv1d(nn.Module):
    """
    Causal 1D convolution for temporal modeling.
    Only looks at past and current time steps (no future lookahead).
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )

    def forward(self, x):
        # x: [batch, channels, time]
        x = self.conv(x)
        # Remove future padding (keep only causal part)
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        return x


class TCNBlock(nn.Module):
    """
    Temporal Convolutional Network block with residual connection.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super(TCNBlock, self).__init__()

        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

        # Residual connection (1x1 conv if dimensions don't match)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        # x: [batch, channels, time]
        residual = self.residual(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = torch.relu(out + residual)

        return out


class ChordRecognitionModelTCN(nn.Module):
    """
    Chord recognition model with CNN feature extraction and TCN temporal modeling.
    Designed for low-latency real-time inference (no future lookahead).

    Architecture:
        Input: [batch, max_len, 84]
        → Reshape to [batch, 1, 84, max_len] for 2D convolution
        → CNN (3 layers with BatchNorm, ReLU, MaxPool, Dropout)
        → TCN (4 blocks with dilations 1, 2, 4, 8)
        → Fully Connected
        Output: [batch, max_len, num_classes]

    Receptive field: ~31 frames (~2.9 seconds at 93ms/frame)
    """

    def __init__(self, num_classes, tcn_channels=256, kernel_size=3, dropout=0.2):
        """
        Args:
            num_classes: Number of chord classes
            tcn_channels: Number of channels in TCN blocks
            kernel_size: Kernel size for TCN convolutions
            dropout: Dropout probability
        """
        super(ChordRecognitionModelTCN, self).__init__()

        self.num_classes = num_classes
        self.tcn_channels = tcn_channels

        # CNN Feature Extractor (same as BiLSTM model)
        # Input: [batch, 1, 84, time_frames]

        self.conv1 = nn.Conv2d(1, 32, kernel_size=(12, 3), stride=1, padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((2, 1))
        self.dropout1 = nn.Dropout2d(dropout)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(6, 3), stride=1, padding=(0, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d((2, 1))
        self.dropout2 = nn.Dropout2d(dropout)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=(0, 1))
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d((2, 1))
        self.dropout3 = nn.Dropout2d(dropout)

        # CNN output: 128 channels * 6 frequency bins = 768
        cnn_output_dim = 128 * 6

        # TCN for temporal modeling (causal - no future lookahead)
        # Dilations: 1, 2, 4, 8 give receptive field of 31 frames
        self.tcn_blocks = nn.ModuleList([
            TCNBlock(cnn_output_dim, tcn_channels, kernel_size, dilation=1, dropout=dropout),
            TCNBlock(tcn_channels, tcn_channels, kernel_size, dilation=2, dropout=dropout),
            TCNBlock(tcn_channels, tcn_channels, kernel_size, dilation=4, dropout=dropout),
            TCNBlock(tcn_channels, tcn_channels, kernel_size, dilation=8, dropout=dropout),
        ])

        # Fully connected layer
        self.dropout_fc = nn.Dropout(dropout)
        self.fc = nn.Linear(tcn_channels, num_classes)

    def forward(self, x, lengths=None):
        """
        Forward pass.

        Args:
            x: Input features [batch, max_len, 84]
            lengths: Sequence lengths [batch] (optional, for compatibility)

        Returns:
            output: Chord predictions [batch, max_len, num_classes]
        """
        batch_size, max_len, _ = x.shape

        # Reshape for CNN: [batch, max_len, 84] → [batch, 1, 84, max_len]
        x = x.transpose(1, 2).unsqueeze(1)

        # CNN feature extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        # Reshape for TCN: [batch, channels, freq, time] → [batch, channels*freq, time]
        x = x.view(batch_size, -1, max_len)  # [batch, 768, time]

        # TCN temporal modeling
        for tcn_block in self.tcn_blocks:
            x = tcn_block(x)

        # Reshape for FC: [batch, channels, time] → [batch, time, channels]
        x = x.transpose(1, 2)  # [batch, time, tcn_channels]

        # Fully connected layer
        x = self.dropout_fc(x)
        output = self.fc(x)  # [batch, max_len, num_classes]

        return output

    def forward_single_frame(self, x, tcn_states=None):
        """
        Forward pass for single frame (real-time inference).
        Note: For proper context, caller should manage a buffer of past frames.

        Args:
            x: Input features [batch, 1, 84] (single frame)
            tcn_states: Reserved for stateful inference (not implemented)

        Returns:
            output: Chord predictions [batch, 1, num_classes]
        """
        return self.forward(x, lengths=None)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer (absolute positions)."""
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


class RelativePositionalEncoding(nn.Module):
    """
    Relative sinusoidal positional encoding for Transformer-XL style attention.
    Generates encodings for relative positions from -max_len to +max_len.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model

        # Create encodings for positions from -max_len to +max_len
        # We'll index with offset max_len (so position 0 is at index max_len)
        pe = torch.zeros(2 * max_len + 1, d_model)
        position = torch.arange(-max_len, max_len + 1, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)
        self.max_len = max_len

    def forward(self, seq_len):
        """
        Get relative positional encodings for a sequence.

        Args:
            seq_len: Length of the sequence

        Returns:
            rel_pos: [seq_len, seq_len, d_model] relative position encodings
        """
        # Create relative position matrix
        # positions[i, j] = j - i (relative position of j with respect to i)
        positions = torch.arange(seq_len, device=self.pe.device)
        rel_positions = positions.unsqueeze(0) - positions.unsqueeze(1)  # [seq_len, seq_len]

        # Clamp to valid range and add offset
        rel_positions = rel_positions.clamp(-self.max_len, self.max_len) + self.max_len

        # Look up encodings
        rel_pos_enc = self.pe[rel_positions]  # [seq_len, seq_len, d_model]

        return rel_pos_enc


class RelativeMultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention with relative positional encoding (Transformer-XL style).

    The attention score includes relative position information:
    a_{i,j} = (q_i + u)^T k_j + (q_i + v)^T r_{i-j}

    Where:
    - q_i, k_j are query and key vectors
    - r_{i-j} is the relative positional encoding for distance (i-j)
    - u, v are learnable bias vectors (per head)
    """
    def __init__(self, d_model, n_heads, dropout=0.1, max_len=5000):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        # Linear projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        # Relative position encoding projection
        self.w_r = nn.Linear(d_model, d_model, bias=False)

        # Learnable biases for content and position (Transformer-XL style)
        # u: content bias, v: position bias
        self.u = nn.Parameter(torch.zeros(n_heads, self.d_head))
        self.v = nn.Parameter(torch.zeros(n_heads, self.d_head))

        # Relative positional encoding generator
        self.rel_pos_enc = RelativePositionalEncoding(d_model, max_len)

        self.dropout = nn.Dropout(dropout)

        # Initialize
        nn.init.xavier_uniform_(self.w_q.weight)
        nn.init.xavier_uniform_(self.w_k.weight)
        nn.init.xavier_uniform_(self.w_v.weight)
        nn.init.xavier_uniform_(self.w_o.weight)
        nn.init.xavier_uniform_(self.w_r.weight)

    def forward(self, x, mask=None):
        """
        Args:
            x: [batch, seq_len, d_model]
            mask: Optional attention mask [batch, seq_len] (True = masked)

        Returns:
            output: [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape

        # Linear projections and reshape to [batch, n_heads, seq_len, d_head]
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        # Get relative positional encodings [seq_len, seq_len, d_model]
        rel_pos = self.rel_pos_enc(seq_len)

        # Project relative positions [seq_len, seq_len, d_model] -> [seq_len, seq_len, n_heads, d_head]
        rel_pos_proj = self.w_r(rel_pos).view(seq_len, seq_len, self.n_heads, self.d_head)

        # Compute attention scores with relative position bias
        # Term 1: (q + u) @ k^T - content-based attention
        q_with_u = q + self.u.unsqueeze(0).unsqueeze(2)  # [batch, n_heads, seq_len, d_head]
        content_score = torch.matmul(q_with_u, k.transpose(-2, -1))  # [batch, n_heads, seq_len, seq_len]

        # Term 2: (q + v) @ r^T - position-based attention
        q_with_v = q + self.v.unsqueeze(0).unsqueeze(2)  # [batch, n_heads, seq_len, d_head]
        # rel_pos_proj: [seq_len, seq_len, n_heads, d_head] -> [n_heads, seq_len, seq_len, d_head]
        rel_pos_proj = rel_pos_proj.permute(2, 0, 1, 3)
        # q_with_v: [batch, n_heads, seq_len, d_head]
        # We need: [batch, n_heads, seq_len (query), seq_len (key)]
        # For each query position i, we want to attend to all key positions j using r_{i-j}
        position_score = torch.einsum('bnid,nijd->bnij', q_with_v, rel_pos_proj)

        # Combine scores
        attn_scores = (content_score + position_score) * self.scale

        # Apply mask if provided
        if mask is not None:
            # mask: [batch, seq_len] -> [batch, 1, 1, seq_len]
            mask = mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        # Softmax and dropout
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, v)  # [batch, n_heads, seq_len, d_head]

        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(output)

        return output


class ConformerFeedForward(nn.Module):
    """
    Feed-forward module with half-step residual (Macaron-style).
    Linear → Swish → Dropout → Linear → Dropout
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.SiLU()  # Swish activation
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


class ConformerConvModule(nn.Module):
    """
    Conformer convolution module with depthwise separable convolution.
    LayerNorm → Pointwise Conv → GLU → Depthwise Conv → BatchNorm → Swish → Pointwise Conv → Dropout
    """
    def __init__(self, d_model, kernel_size=31, dropout=0.1):
        super().__init__()

        self.layer_norm = nn.LayerNorm(d_model)

        # Pointwise conv (expand to 2x for GLU)
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

        # Pointwise conv (project back)
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


class ConformerBlock(nn.Module):
    """
    Conformer block: FFN → Self-Attention → Conv → FFN → LayerNorm
    Uses Macaron-style half-step residuals for feed-forward modules.
    Uses relative positional encoding in self-attention (Transformer-XL style).
    """
    def __init__(self, d_model, n_heads, d_ff, conv_kernel_size=31, dropout=0.1):
        super().__init__()

        # First feed-forward (half-step)
        self.ff1 = ConformerFeedForward(d_model, d_ff, dropout)

        # Multi-head self-attention with relative positional encoding
        self.attn_layer_norm = nn.LayerNorm(d_model)
        self.self_attn = RelativeMultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout
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
            mask: Optional attention mask [batch, time] (True = masked)
        Returns:
            out: [batch, time, d_model]
        """
        # First FFN (half-step)
        x = self.ff1(x)

        # Self-attention with relative positional encoding
        residual = x
        x = self.attn_layer_norm(x)
        x = self.self_attn(x, mask=mask)
        x = self.attn_dropout(x)
        x = residual + x

        # Convolution
        x = self.conv_module(x)

        # Second FFN (half-step)
        x = self.ff2(x)

        # Final layer norm
        x = self.final_layer_norm(x)

        return x


class ChordFormerModel(nn.Module):
    """
    ChordFormer: Conformer-based architecture for large-vocabulary chord recognition.

    Architecture:
        Input: [batch, time, 252] CQT spectrogram (36 bins/octave)
        → Linear projection to d_model
        → Dropout
        → Conformer blocks with relative positional encoding (×4)
        → 6 output heads (Root+Triad, Bass, 7th, 9th, 11th, 13th)

    Uses relative sinusoidal positional encoding (Transformer-XL style) inside
    the self-attention mechanism, not absolute positional encoding at input.

    Based on: "ChordFormer: A Conformer-Based Architecture for Large-Vocabulary
              Audio Chord Recognition" (Akram et al., 2025)
    """
    def __init__(
        self,
        n_bins=252,
        d_model=256,
        n_heads=4,
        d_ff=1024,
        n_layers=4,
        conv_kernel_size=31,
        dropout=0.1,
        num_root_triad=85,
        num_bass=13,
        num_7th=4,
        num_9th=4,
        num_11th=3,
        num_13th=3
    ):
        super().__init__()

        self.n_bins = n_bins
        self.d_model = d_model

        # Input projection
        self.input_projection = nn.Linear(n_bins, d_model)

        # Dropout after projection (replaces absolute positional encoding dropout)
        self.input_dropout = nn.Dropout(dropout)

        # Conformer layers (with relative positional encoding in attention)
        self.conformer_layers = nn.ModuleList([
            ConformerBlock(d_model, n_heads, d_ff, conv_kernel_size, dropout)
            for _ in range(n_layers)
        ])

        # Output heads (structured chord representation)
        self.root_triad_head = nn.Linear(d_model, num_root_triad)
        self.bass_head = nn.Linear(d_model, num_bass)
        self.seventh_head = nn.Linear(d_model, num_7th)
        self.ninth_head = nn.Linear(d_model, num_9th)
        self.eleventh_head = nn.Linear(d_model, num_11th)
        self.thirteenth_head = nn.Linear(d_model, num_13th)

    def forward(self, x, lengths=None):
        """
        Args:
            x: Input features [batch, time, n_bins]
            lengths: Optional sequence lengths for masking

        Returns:
            dict with keys: 'root_triad', 'bass', '7th', '9th', '11th', '13th'
            Each value is [batch, time, num_classes]
        """
        # Project to model dimension
        x = self.input_projection(x)  # [batch, time, d_model]

        # Apply dropout (no absolute positional encoding - using relative in attention)
        x = self.input_dropout(x)

        # Create attention mask if lengths provided
        mask = None
        if lengths is not None:
            batch_size, max_len = x.shape[:2]
            lengths = lengths.to(x.device)  # Ensure lengths is on same device
            mask = torch.arange(max_len, device=x.device).expand(batch_size, max_len)
            mask = mask >= lengths.unsqueeze(1)

        # Conformer layers (relative positional encoding is inside attention)
        for layer in self.conformer_layers:
            x = layer(x, mask)

        # Output predictions (6 heads)
        outputs = {
            'root_triad': self.root_triad_head(x),
            'bass': self.bass_head(x),
            '7th': self.seventh_head(x),
            '9th': self.ninth_head(x),
            '11th': self.eleventh_head(x),
            '13th': self.thirteenth_head(x)
        }

        return outputs


def compute_class_weights(class_counts, gamma=0.5, w_max=10.0):
    """
    Compute class weights for handling imbalance using equation (9) from paper.

    w_m = min{(n_m / max_m' n_m')^(-gamma), w_max}

    Args:
        class_counts: Array of sample counts per class [num_classes]
        gamma: Balancing factor (0 = uniform, 1 = full inverse frequency)
               Paper uses values like 0.3, 0.5, 0.7
        w_max: Maximum weight clamp to prevent extreme values
               Paper uses values like 10.0, 20.0

    Returns:
        weights: Tensor of class weights [num_classes]
    """
    counts = torch.tensor(class_counts, dtype=torch.float32)

    # Avoid division by zero for classes with no samples
    counts = torch.clamp(counts, min=1.0)

    # Normalize by max count
    max_count = counts.max()
    normalized = counts / max_count  # Values in (0, 1]

    # Apply inverse frequency with gamma
    # (n_m / max_n)^(-gamma) = (max_n / n_m)^gamma
    weights = torch.pow(normalized, -gamma)

    # Clamp to maximum weight
    weights = torch.clamp(weights, max=w_max)

    return weights


def compute_all_class_weights(train_labels, gamma=0.5, w_max=10.0):
    """
    Compute class weights for all 6 ChordFormer heads from training labels.

    Args:
        train_labels: List of label dicts from training set, or aggregated counts dict
                      {'root_triad': counts_array, 'bass': counts_array, ...}
        gamma: Balancing factor for re-weighting
        w_max: Maximum weight clamp

    Returns:
        class_weights: Dict of weight tensors for each head
    """
    head_sizes = {
        'root_triad': 85,
        'bass': 13,
        '7th': 4,
        '9th': 4,
        '11th': 3,
        '13th': 3
    }

    class_weights = {}

    for head, num_classes in head_sizes.items():
        if isinstance(train_labels, dict) and head in train_labels:
            counts = train_labels[head]
        else:
            # If counts not provided, use uniform weights
            counts = torch.ones(num_classes)

        weights = compute_class_weights(counts, gamma, w_max)
        class_weights[head] = weights

    return class_weights


class ChordFormerLoss(nn.Module):
    """
    Multi-task loss for ChordFormer with class re-weighting.

    Implements equation (8) from the ChordFormer paper:
    L = -Σ_t Σ_j Σ_m w_m^(j) * I[m = z_j^(t)] * log(β_m^(t,j))

    This is weighted cross-entropy summed over all 6 output heads,
    where w_m^(j) are class weights computed using equation (9).
    """
    def __init__(
        self,
        class_weights=None
    ):
        """
        Args:
            class_weights: Dict of weight tensors for each head
                          {'root_triad': tensor, 'bass': tensor, ...}
                          Weights computed using equation (9) from paper.
        """
        super().__init__()

        # Create loss functions for each head
        # Paper uses simple sum over all heads (no head-specific weighting)
        self.losses = nn.ModuleDict()
        heads = ['root_triad', 'bass', '7th', '9th', '11th', '13th']

        for head in heads:
            weight = class_weights.get(head) if class_weights else None
            self.losses[head] = nn.CrossEntropyLoss(
                weight=weight,
                ignore_index=-1  # For padded sequences
                # No label smoothing - paper doesn't mention it
            )

    def forward(self, outputs, targets):
        """
        Compute total loss as sum of weighted CE losses over all 6 heads.

        Args:
            outputs: Dict from model forward {'root_triad': [B,T,C], ...}
            targets: Dict with same keys, containing label tensors [B,T]

        Returns:
            total_loss: Scalar loss (sum over all heads)
        """
        total_loss = 0

        for head in ['root_triad', 'bass', '7th', '9th', '11th', '13th']:
            if head in outputs and head in targets:
                # Reshape: [batch, time, classes] -> [batch*time, classes]
                pred = outputs[head].reshape(-1, outputs[head].shape[-1])
                # Reshape: [batch, time] -> [batch*time]
                tgt = targets[head].reshape(-1)

                loss = self.losses[head](pred, tgt)
                # Paper equation (8): simple sum over all heads
                total_loss = total_loss + loss

        return total_loss


# =============================================================================
# MIREX 2025 Degree-Based Chord Recognition Model
# Paper: "Degree-Based Automatic Chord Recognition with Enharmonic Distinction"
# =============================================================================

class OctavewiseConvModule(nn.Module):
    """
    Octavewise convolution for capturing intervallic patterns.

    From MIREX 2025 paper Section 2.2:
    "The acoustic features are first processed with a convolutional kernel of
    size one octave in the frequency direction (kernel_size = bins_per_octave),
    sliding by one semitone (stride = bins_per_octave // 12). The output is
    then passed through a linear layer to obtain a 256-dimensional representation."

    Architecture:
        Input:  [B, T, 252] - CQT features (7 octaves x 36 bins/octave)
        Reshape: [B*T, 1, 252] - treat each frame independently
        Conv1d: kernel=36, stride=3 -> [B*T, n_filters, 73]
        BatchNorm + SiLU (Swish)
        Flatten + Linear: n_filters * 73 -> d_model (256)
        Output: [B, T, 256]
    """

    def __init__(self, n_bins=252, bins_per_octave=36, d_model=256, n_filters=64):
        """
        Args:
            n_bins: Number of CQT frequency bins (default 252 = 7 octaves * 36)
            bins_per_octave: CQT bins per octave (default 36)
            d_model: Output dimension (default 256 per paper)
            n_filters: Number of conv filters (default 64, not specified in paper)
        """
        super().__init__()

        kernel_size = bins_per_octave  # 36 = 1 octave
        stride = bins_per_octave // 12  # 3 = 1 semitone

        # After conv: (252 - 36) / 3 + 1 = 73 positions
        conv_out_positions = (n_bins - kernel_size) // stride + 1  # 73

        self.conv = nn.Conv1d(1, n_filters, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm1d(n_filters)
        self.activation = nn.SiLU()  # Swish activation
        self.linear = nn.Linear(n_filters * conv_out_positions, d_model)

        self.n_bins = n_bins
        self.n_filters = n_filters
        self.conv_out_positions = conv_out_positions

    def forward(self, x):
        """
        Args:
            x: CQT features [B, T, n_bins]

        Returns:
            out: Projected features [B, T, d_model]
        """
        B, T, F = x.shape

        # Reshape: [B, T, F] -> [B*T, 1, F]
        x = x.view(B * T, 1, F)

        # Conv1d: [B*T, 1, 252] -> [B*T, n_filters, 73]
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)

        # Flatten: [B*T, n_filters, 73] -> [B*T, n_filters * 73]
        x = x.view(B * T, -1)

        # Linear: [B*T, n_filters * 73] -> [B*T, d_model]
        x = self.linear(x)

        # Reshape back: [B*T, d_model] -> [B, T, d_model]
        x = x.view(B, T, -1)

        return x


class MIREXChordFormerModel(nn.Module):
    """
    MIREX 2025 Degree-Based ChordFormer Model.

    Architecture changes from standard ChordFormer:
    1. Replaces input linear projection with OctavewiseConvModule
    2. Uses 6 output heads (3 categorical + 3 binary) instead of original 6 heads
    3. Outputs key-relative scale degrees instead of absolute pitches

    Output Structure (80 total neurons):
        Categorical Heads (CrossEntropy):
            - Key:         13 classes {N, C, C#, D, D#, E, F, F#, G, G#, A, A#, B}
            - Root Degree: 18 classes (N + 17 scale degrees with enharmonic distinction)
            - Bass:        13 classes {N, C, C#, D, D#, E, F, F#, G, G#, A, A#, B}

        Binary Heads (BCE with sigmoid):
            - Absolute Pitches:    12 binary (pitch class presence)
            - Intervals from Root: 12 binary (interval presence)
            - Intervals from Bass: 12 binary (interval presence)
    """

    def __init__(
        self,
        n_bins=252,
        d_model=256,
        n_heads=4,
        d_ff=1024,
        n_layers=4,
        conv_kernel_size=31,
        dropout=0.1,
        octavewise_n_filters=64,
        num_keys=13,
        num_degrees=18,
        num_bass=13,
        num_pitches=12
    ):
        """
        Args:
            n_bins: Number of CQT frequency bins (default 252)
            d_model: Model dimension (default 256)
            n_heads: Number of attention heads (default 4)
            d_ff: Feed-forward dimension (default 1024)
            n_layers: Number of Conformer blocks (default 4)
            conv_kernel_size: Conformer conv kernel size (default 31)
            dropout: Dropout probability (default 0.1)
            octavewise_n_filters: Number of filters in OctavewiseConvModule (default 64)
            num_keys: Number of key classes (default 13 = N + 12 keys)
            num_degrees: Number of degree classes (default 18)
            num_bass: Number of bass classes (default 13 = N + 12 notes)
            num_pitches: Number of pitch classes (default 12)
        """
        super().__init__()

        self.n_bins = n_bins
        self.d_model = d_model

        # Octavewise Convolution Module (replaces simple linear projection)
        self.octavewise_conv = OctavewiseConvModule(
            n_bins=n_bins,
            bins_per_octave=36,
            d_model=d_model,
            n_filters=octavewise_n_filters
        )

        # Dropout after projection
        self.input_dropout = nn.Dropout(dropout)

        # Conformer layers (with relative positional encoding in attention)
        self.conformer_layers = nn.ModuleList([
            ConformerBlock(d_model, n_heads, d_ff, conv_kernel_size, dropout)
            for _ in range(n_layers)
        ])

        # === Categorical Output Heads (CrossEntropy) ===
        self.key_head = nn.Linear(d_model, num_keys)
        self.degree_head = nn.Linear(d_model, num_degrees)
        self.bass_head = nn.Linear(d_model, num_bass)

        # === Binary Output Heads (BCEWithLogits - 12 dimensions each) ===
        self.pitches_abs_head = nn.Linear(d_model, num_pitches)
        self.intervals_root_head = nn.Linear(d_model, num_pitches)
        self.intervals_bass_head = nn.Linear(d_model, num_pitches)

        # Store head sizes for reference
        self.num_keys = num_keys
        self.num_degrees = num_degrees
        self.num_bass = num_bass
        self.num_pitches = num_pitches

    def forward(self, x, lengths=None):
        """
        Args:
            x: Input CQT features [batch, time, n_bins]
            lengths: Optional sequence lengths for masking [batch]

        Returns:
            dict with keys:
                'key':            [batch, time, 13]  - Key logits
                'degree':         [batch, time, 18]  - Scale degree logits
                'bass':           [batch, time, 13]  - Bass note logits
                'pitches_abs':    [batch, time, 12]  - Absolute pitch presence logits
                'intervals_root': [batch, time, 12]  - Intervals from root logits
                'intervals_bass': [batch, time, 12]  - Intervals from bass logits
        """
        # Octavewise convolution input processing
        x = self.octavewise_conv(x)  # [batch, time, d_model]

        # Apply dropout
        x = self.input_dropout(x)

        # Create attention mask if lengths provided
        mask = None
        if lengths is not None:
            batch_size, max_len = x.shape[:2]
            lengths = lengths.to(x.device)
            mask = torch.arange(max_len, device=x.device).expand(batch_size, max_len)
            mask = mask >= lengths.unsqueeze(1)

        # Conformer layers
        for layer in self.conformer_layers:
            x = layer(x, mask)

        # Output predictions (6 heads)
        outputs = {
            # Categorical heads (for CrossEntropy loss)
            'key': self.key_head(x),
            'degree': self.degree_head(x),
            'bass': self.bass_head(x),
            # Binary heads (for BCEWithLogits loss)
            'pitches_abs': self.pitches_abs_head(x),
            'intervals_root': self.intervals_root_head(x),
            'intervals_bass': self.intervals_bass_head(x),
        }

        return outputs


class MIREXLoss(nn.Module):
    """
    Combined loss for MIREX degree-based chord recognition.

    Loss = ce_weight * (CE_key + CE_degree + CE_bass)
         + bce_weight * (BCE_pitches_abs + BCE_intervals_root + BCE_intervals_bass)

    The loss combines:
    - CrossEntropy for categorical outputs (key, degree, bass)
    - BCEWithLogits for binary pitch presence outputs (3 x 12 dimensions)
    """

    def __init__(self, class_weights=None, ce_weight=1.0, bce_weight=0.5):
        """
        Args:
            class_weights: Optional dict of weight tensors for each head
                          {'key': tensor, 'degree': tensor, 'bass': tensor}
            ce_weight: Weight for categorical (CrossEntropy) losses
            bce_weight: Weight for binary (BCE) losses
        """
        super().__init__()

        # Categorical losses (with optional class weights)
        self.ce_key = nn.CrossEntropyLoss(
            weight=class_weights.get('key') if class_weights else None,
            ignore_index=-1
        )
        self.ce_degree = nn.CrossEntropyLoss(
            weight=class_weights.get('degree') if class_weights else None,
            ignore_index=-1
        )
        self.ce_bass = nn.CrossEntropyLoss(
            weight=class_weights.get('bass') if class_weights else None,
            ignore_index=-1
        )

        # Binary losses (multi-label)
        self.bce = nn.BCEWithLogitsLoss()

        self.ce_weight = ce_weight
        self.bce_weight = bce_weight

    def forward(self, outputs, targets):
        """
        Compute combined loss.

        Args:
            outputs: Dict from model forward
                     {'key': [B,T,13], 'degree': [B,T,18], 'bass': [B,T,13],
                      'pitches_abs': [B,T,12], 'intervals_root': [B,T,12],
                      'intervals_bass': [B,T,12]}
            targets: Dict with same keys, containing label tensors
                     Categorical: [B,T] int64
                     Binary: [B,T,12] float32

        Returns:
            total_loss: Scalar loss value
        """
        # Categorical losses (reshape for CrossEntropy)
        loss_key = self.ce_key(
            outputs['key'].view(-1, outputs['key'].shape[-1]),
            targets['key'].view(-1)
        )
        loss_degree = self.ce_degree(
            outputs['degree'].view(-1, outputs['degree'].shape[-1]),
            targets['degree'].view(-1)
        )
        loss_bass = self.ce_bass(
            outputs['bass'].view(-1, outputs['bass'].shape[-1]),
            targets['bass'].view(-1)
        )

        # Binary losses
        loss_pitches = self.bce(outputs['pitches_abs'], targets['pitches_abs'].float())
        loss_intervals_root = self.bce(outputs['intervals_root'], targets['intervals_root'].float())
        loss_intervals_bass = self.bce(outputs['intervals_bass'], targets['intervals_bass'].float())

        # Combine losses
        total = (
            self.ce_weight * (loss_key + loss_degree + loss_bass) +
            self.bce_weight * (loss_pitches + loss_intervals_root + loss_intervals_bass)
        )

        return total


def test_model():
    """Test BiLSTM, TCN, and ChordFormer models with dummy data."""
    batch_size = 4
    max_len = 200
    num_classes = 25

    # Test BiLSTM model
    print("=" * 60)
    print("Testing ChordRecognitionModel (BiLSTM)...")
    features = torch.randn(batch_size, max_len, 84)
    lengths = torch.tensor([200, 180, 150, 120], dtype=torch.long)
    print("=" * 60)

    model_bilstm = ChordRecognitionModel(num_classes=num_classes, hidden_size=256, dropout=0.2)
    output_bilstm = model_bilstm(features, lengths)

    print(f"Input shape: {features.shape}")
    print(f"Output shape: {output_bilstm.shape}")
    assert output_bilstm.shape == (batch_size, max_len, num_classes), "BiLSTM output shape mismatch!"

    params_bilstm = sum(p.numel() for p in model_bilstm.parameters())
    print(f"Total parameters: {params_bilstm:,}")
    print("BiLSTM model test passed!\n")

    # Test TCN model
    print("=" * 60)
    print("Testing ChordRecognitionModelTCN...")
    print("=" * 60)

    model_tcn = ChordRecognitionModelTCN(num_classes=num_classes, tcn_channels=256, dropout=0.2)
    output_tcn = model_tcn(features, lengths)

    print(f"Input shape: {features.shape}")
    print(f"Output shape: {output_tcn.shape}")
    assert output_tcn.shape == (batch_size, max_len, num_classes), "TCN output shape mismatch!"

    params_tcn = sum(p.numel() for p in model_tcn.parameters())
    print(f"Total parameters: {params_tcn:,}")
    print("TCN model test passed!\n")

    # Test ChordFormer model
    print("=" * 60)
    print("Testing ChordFormerModel...")
    print("=" * 60)

    features_cf = torch.randn(batch_size, max_len, 252)  # 36 bins/octave
    model_chordformer = ChordFormerModel(
        n_bins=252,
        d_model=256,
        n_heads=4,
        d_ff=1024,
        n_layers=4
    )
    outputs_cf = model_chordformer(features_cf, lengths)

    print(f"Input shape: {features_cf.shape}")
    print(f"Output shapes:")
    for key, val in outputs_cf.items():
        print(f"  {key}: {val.shape}")

    params_cf = sum(p.numel() for p in model_chordformer.parameters())
    print(f"Total parameters: {params_cf:,}")
    print("ChordFormer model test passed!\n")

    # Test MIREX ChordFormer model
    print("=" * 60)
    print("Testing MIREXChordFormerModel...")
    print("=" * 60)

    model_mirex = MIREXChordFormerModel(
        n_bins=252,
        d_model=256,
        n_heads=4,
        d_ff=1024,
        n_layers=4
    )
    outputs_mirex = model_mirex(features_cf, lengths)

    print(f"Input shape: {features_cf.shape}")
    print(f"Output shapes:")
    for key, val in outputs_mirex.items():
        print(f"  {key}: {val.shape}")

    params_mirex = sum(p.numel() for p in model_mirex.parameters())
    print(f"Total parameters: {params_mirex:,}")
    print("MIREX ChordFormer model test passed!\n")

    # Compare
    print("=" * 60)
    print("Model Comparison:")
    print("=" * 60)
    print(f"BiLSTM parameters:           {params_bilstm:,}")
    print(f"TCN parameters:              {params_tcn:,}")
    print(f"ChordFormer parameters:      {params_cf:,}")
    print(f"MIREX ChordFormer parameters: {params_mirex:,}")

    return model_bilstm, model_tcn, model_chordformer, model_mirex


if __name__ == '__main__':
    test_model()
