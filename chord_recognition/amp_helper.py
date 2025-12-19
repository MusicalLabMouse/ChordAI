"""
Mixed Precision (AMP) Training Helper

To remove AMP entirely:
1. Delete this file
2. In train.py, remove: from amp_helper import AMPHelper
3. In train.py, remove the amp_helper usage (search for "amp" to find them)
"""

import torch


class AMPHelper:
    """
    Simple wrapper for PyTorch Automatic Mixed Precision.

    Usage:
        amp = AMPHelper(enabled=True)

        with amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        amp.backward(loss, optimizer, model, max_grad_norm=5.0)
    """

    def __init__(self, enabled=True):
        """
        Args:
            enabled: If False, all methods become no-ops (regular FP32 training)
        """
        self.enabled = enabled and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler('cuda') if self.enabled else None

        if self.enabled:
            print("Mixed precision training (AMP) enabled")

    def autocast(self):
        """Context manager for forward pass. Use: with amp.autocast(): ..."""
        return torch.amp.autocast('cuda', enabled=self.enabled)

    def backward(self, loss, optimizer, model, max_grad_norm=5.0):
        """
        Handles backward pass with optional gradient scaling.

        Args:
            loss: Loss tensor
            optimizer: Optimizer
            model: Model (for gradient clipping)
            max_grad_norm: Gradient clipping value
        """
        if self.enabled:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
