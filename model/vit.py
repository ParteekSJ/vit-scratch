import torch
from torch import nn
from .patch_embdding import PatchEmbeddingModule
from .encoder import ViTEncoderBlock


class ViT(nn.Module):
    """
    Implementation of the Vision Transformer (ViT) as described in
    "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
    (Dosovitskiy et al., 2021)

    Paper: https://arxiv.org/abs/2010.11929
    """

    def __init__(self, args, **kwargs):
        """
        Initialize Vision Transformer.

        Args:
            args: Model argsuration object
            **kwargs: Optional parameters that override args values
        """
        super().__init__()

        self.args = args

        # Validate image size and patch size compatibility
        assert (
            args.img_size % args.patch_size == 0
        ), f"Image size ({args.img_size}) must be divisible by patch size ({args.patch_size})"

        # Calculate number of patches
        self.num_patches = (args.img_size // args.patch_size) ** 2

        # Create learnable class token ([CLS])
        self.class_embedding = nn.Parameter(torch.zeros(1, 1, args.embedding_dim))

        # Create learnable position embeddings
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, args.embedding_dim))

        # Patch embedding layer
        self.patch_embedding = PatchEmbeddingModule(
            in_channels=args.in_channels, patch_size=args.patch_size, embedding_dim=args.embedding_dim
        )

        # Dropout for embeddings
        self.embedding_dropout = nn.Dropout(args.dropout_rate)

        # Transformer encoder blocks
        self.encoder_blocks = nn.ModuleList(
            [
                ViTEncoderBlock(
                    embedding_dim=args.embedding_dim,
                    num_heads=args.num_heads,
                    mlp_dim=args.mlp_dim,
                    mlp_dropout=args.dropout_rate,
                    attn_dropout=args.attn_dropout_rate,
                )
                for _ in range(args.depth)
            ]
        )

        # Layer normalization for final output
        self.layer_norm = nn.LayerNorm(args.embedding_dim)

        # Classification head
        self.classifier = nn.Linear(in_features=args.embedding_dim, out_features=args.num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights similar to the original implementation"""
        # Initialize patch embedding like a linear layer
        nn.init.xavier_uniform_(self.patch_embedding.embedding_layer.weight)
        nn.init.zeros_(self.patch_embedding.embedding_layer.bias)

        # Initialize positional embedding
        nn.init.normal_(self.pos_embedding, std=0.02)

        # Initialize class token
        nn.init.normal_(self.class_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Vision Transformer.

        Args:
            x: Input images [batch_size, channels, height, width]

        Returns:
            Classification logits [batch_size, num_classes]
        """
        # Get batch size
        batch_size = x.shape[0]

        # Create patch embeddings
        x = self.patch_embedding(x)  # [B, num_patches, embedding_dim]

        # Expand class token to batch size and prepend to patches
        class_token = self.class_embedding.expand(batch_size, -1, -1)
        x = torch.cat((class_token, x), dim=1)  # [B, num_patches+1, embedding_dim]

        # Add positional embeddings
        x = x + self.pos_embedding

        # Apply embedding dropout
        x = self.embedding_dropout(x)

        # Pass through encoder blocks
        for block in self.encoder_blocks:
            x = block(x)

        # Apply final layer norm
        x = self.layer_norm(x)

        # Use class token ([CLS]) for classification
        x = x[:, 0]

        # Apply classification head
        x = self.classifier(x)

        return x
