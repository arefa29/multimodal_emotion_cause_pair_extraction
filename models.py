import torch
import torch.nn as nn

class EmbeddingModifierTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers):
        super(EmbeddingModifierTransformer, self).__init__()


