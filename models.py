import torch
import torch.nn as nn

class EmbeddingModifierTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers):
        super(EmbeddingModifierTransformer, self).__init__()

        # Multi-Head Self-Attention Layer
        self.self_attn = nn.MultiheadAttention(input_dim, num_heads)

        # Position-wise Feedforward Network
        self.feedforward = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        # Layer Normalization
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)

        # Number of transformer layers
        self.num_layers = num_layers

    def forward(self, input_embeddings):
        # Apply multiple layers of self-attention and feedforward networks
        modified_embeddings = input_embeddings  # initialize
        for _ in range(self.num_layers):
            # Multi-Head Self-Attention
            attn_output, _ = self.self_attn(modified_embeddings, modified_embeddings, modified_embeddings)
            modified_embeddings = self.layer_norm1(attn_output + modified_embeddings)  # Residual connection

            # Position-wise Feedforward Network
            ff_output = self.feedforward(modified_embeddings)
            modified_embeddings = self.layer_norm2(ff_output + modified_embeddings)  # Residual connection

        return modified_embeddings

