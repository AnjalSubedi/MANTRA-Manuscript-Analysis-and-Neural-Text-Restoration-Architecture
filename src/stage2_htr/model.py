"""Convolutional recurrent recognizer for variable-width HTR line images."""

from math import prod

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CRNN(nn.Module):
    """Turn a Stage 2 image batch into CTC log probabilities and input lengths."""

    _pool_sizes = ((2, 2), (2, 2), (2, 1), (2, 1))
    width_stride = prod(width for _, width in _pool_sizes)

    def __init__(self, num_classes: int, *, image_height: int = 64, hidden_size: int = 128):
        super().__init__()
        feature_height = image_height // prod(height for height, _ in self._pool_sizes)
        if feature_height < 1:
            raise ValueError("image_height must be at least 16")
        if num_classes < 2 or hidden_size < 1:
            raise ValueError("num_classes must include a blank and a character, and hidden_size must be positive")
        self.image_height = image_height

        channels = (1, 16, 32, 64, 128)
        self.feature_blocks = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(pool_size),
            )
            for in_channels, out_channels, pool_size in zip(channels, channels[1:], self._pool_sizes)
        )
        feature_size = channels[-1] * feature_height
        self.feature_norm = nn.LayerNorm(feature_size)
        self.recurrent = nn.LSTM(
            feature_size, hidden_size, num_layers=2, batch_first=True,
            bidirectional=True, dropout=0.1,
        )
        self.classifier = nn.Linear(2 * hidden_size, num_classes)

    def output_lengths(self, processed_widths: torch.Tensor) -> torch.Tensor:
        """Compute time steps from resized widths before right-padding."""
        return processed_widths // self.width_stride

    def forward(
        self, images: torch.Tensor, processed_widths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Accept `[B, 1, H, W]` images and `[B]` widths.

        Return time-major log probabilities `[T, B, C]` and CPU int64
        lengths `[B]`. Only time steps before each length belong to the line.
        """
        if images.ndim != 4 or images.shape[0] == 0 or images.shape[1:3] != (1, self.image_height):
            raise ValueError(f"images must have shape [B, 1, {self.image_height}, W] for a nonempty batch")
        if processed_widths.shape != (images.shape[0],) or processed_widths.dtype not in (torch.int32, torch.int64):
            raise ValueError("processed_widths must contain one integer width per image")
        widths = processed_widths.to(device="cpu", dtype=torch.long)
        if ((widths < self.width_stride) | (widths > images.shape[-1])).any():
            raise ValueError(f"processed widths must be between {self.width_stride} and the padded image width")
        input_lengths = self.output_lengths(widths)

        features = images
        valid_widths = widths.to(images.device)
        for block, (_, stride) in zip(self.feature_blocks, self._pool_sizes):
            # Match the convolution's zero padding at the edge of each crop.
            # Mask again after pooling so invalid activations stay out of later blocks.
            columns = torch.arange(features.shape[-1], device=images.device)
            outside_crop = columns >= valid_widths[:, None, None, None]
            features = block(features.masked_fill(outside_crop, 0))
            valid_widths = valid_widths // stride

        columns = self.feature_norm(features.permute(0, 3, 1, 2).flatten(2))
        packed = pack_padded_sequence(columns, input_lengths, batch_first=True, enforce_sorted=False)
        encoded, _ = self.recurrent(packed)
        encoded, _ = pad_packed_sequence(encoded, batch_first=True, total_length=columns.shape[1])
        log_probs = self.classifier(encoded).log_softmax(dim=-1)
        return log_probs.transpose(0, 1).contiguous(), input_lengths
