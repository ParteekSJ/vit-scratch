import torch
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dim: int = 768, num_heads: int = 12, attn_dropout: float = 0.0):
        """
        Args:
            embedding_dim: Dimension of token embeddings
            num_heads: Number of attention heads
            attn_dropout: Dropout rate for attention weights
        """
        super().__init__()

        # Pre-normalization layer (LayerNorm)
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, embedding_dim]

        Returns:
            Output tensor with same shape after self-attention + residual connection
        """
        # Store original input for residual connection
        residual = x

        # Pre-normalization
        x_norm = self.layer_norm(x)

        # Self-attention (query, key, and value are the same)
        attn_output, _ = self.multihead_attn(query=x_norm, key=x_norm, value=x_norm, need_weights=False)

        # Residual connection
        return attn_output + residual


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 768,
        mlp_dim: int = 3072,
        dropout: float = 0.1,
    ):
        """
        Args:
            embedding_dim: Dimension of token embeddings
            mlp_dim: Hidden dimension of the MLP
            dropout: Dropout rate
        """
        super().__init__()

        # Pre-normalization layer
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        # MLP with two linear layers, GELU activations, and dropout
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(mlp_dim, embedding_dim),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, embedding_dim]

        Returns:
            Output tensor with same shape after MLP + residual connection
        """
        # Store original input for residual connection
        residual = x

        # Pre-normalization
        x_norm = self.layer_norm(x)

        # MLP
        mlp_output = self.mlp(x_norm)

        # Residual connection
        return mlp_output + residual


class ViTEncoderBlock(nn.Module):
    """
    Transformer encoder block for Vision Transformer with:
    - Multi-head self-attention with pre-norm and residual connection
    - MLP block with pre-norm and residual connection
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        num_heads: int = 12,
        mlp_dim: int = 3072,
        mlp_dropout: float = 0.1,
        attn_dropout: float = 0.0,
    ):
        """
        Args:
            embedding_dim: Dimension of token embeddings
            num_heads: Number of attention heads
            mlp_dim: Hidden dimension of the MLP
            mlp_dropout: Dropout rate for MLP
            attn_dropout: Dropout rate for attention weights
        """
        super().__init__()

        self.mhsa_block = MultiHeadSelfAttention(
            embedding_dim=embedding_dim, num_heads=num_heads, attn_dropout=attn_dropout
        )

        self.mlp_block = MLPBlock(embedding_dim=embedding_dim, mlp_dim=mlp_dim, dropout=mlp_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, embedding_dim]

        Returns:
            Output tensor with same shape after full encoder block
        """
        # Apply attention block (with its own residual connection)
        x = self.mhsa_block(x)

        # Apply MLP block (with its own residual connection)
        x = self.mlp_block(x)

        return x
