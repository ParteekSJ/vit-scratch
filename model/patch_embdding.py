import torch
from torch import nn
from typing import Tuple


class PatchEmbeddingModule(nn.Module):
    """
    Image to Patch Embedding Module as described in
    "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"

    This module converts an image into a sequence of patch embeddings:
    1. Split the image into non-overlapping patches
    2. Flatten the patches
    3. Map to embedding dimension with a linear projection (implemented as a Conv2D layer)
    """

    def __init__(self, in_channels: int = 3, patch_size: int = 16, embedding_dim: int = 768):
        """
        Args:
            in_channels: Number of input channels (e.g., 3 for RGB images)
            patch_size: Size of each patch (e.g., 16 means 16x16 pixel patches)
            embedding_dim: Dimension of token embeddings
        """
        super().__init__()

        self.patch_size = patch_size

        # Linear projection of flattened patches (implemented as a Conv2d layer)
        self.embedding_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,  # Kernel size determines patch size
            stride=patch_size,  # Non-overlapping patches
            padding=0,
        )

        # Flatten the spatial dimensions
        self.flattening_layer = nn.Flatten(start_dim=-2, end_dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images [batch_size, channels, height, width]

        Returns:
            Sequence of patch embeddings [batch_size, num_patches, embedding_dim]
        """
        # Validate input dimensions
        batch_size, channels, height, width = x.shape
        assert (
            height % self.patch_size == 0 and width % self.patch_size == 0
        ), f"Input image dimensions ({height}x{width}) must be divisible by patch_size ({self.patch_size})"

        # Project patches to embedding dimension
        # [B, C, H, W] -> [B, embedding_dim, H/patch_size, W/patch_size]
        x_patched = self.embedding_layer(x)

        # Flatten spatial dimensions
        # [B, embedding_dim, H/patch_size, W/patch_size] -> [B, embedding_dim, num_patches]
        x_flattened = self.flattening_layer(x_patched)

        # Rearrange dimensions to [B, num_patches, embedding_dim]
        return x_flattened.permute(0, 2, 1)
