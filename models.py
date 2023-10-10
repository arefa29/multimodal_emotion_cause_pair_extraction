import torch
import torch.nn as nn
from utils import get_stacked_tensor

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
            # Multi-Head Self-Attention, giving k, q, v
            attn_output, _ = self.self_attn(modified_embeddings, modified_embeddings, modified_embeddings)
            # Residual connection
            modified_embeddings = self.layer_norm1(attn_output + modified_embeddings) 

            # Position-wise Feedforward Network
            ff_output = self.feedforward(modified_embeddings)
            # Residual connection
            modified_embeddings = self.layer_norm2(ff_output + modified_embeddings)

        return modified_embeddings

class MultipleCauseClassifier(nn.Module):
    """Gives probability for each pair within each conversation whether that utt-pair has a cause"""
    def __init__(self, input_dim, num_utt_tensors, num_labels):
        super(MultipleCauseClassifier, self).__init__()
        self.num_utt_tensors = num_utt_tensors

        # Linear layers for each tensor
        self.linear_layers = nn.ModuleList([nn.Linear(input_dim, num_labels) for _ in range(num_utt_tensors)])
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for tensor, linear in zip(x, self.linear_layers):
            tensor_probs = self.softmax(linear(tensor))
            probabilities.append(tensor_probs)
        return probabilities

class EmotionCausePairClassifierModel(nn.Module):
    def __init__(self, args):
        super(EmotionCausePairClassifierModel, self).__init__()

        self.args = args

        self.transformer_model = EmbeddingModifierTransformer(args.input_dim_transformer, args.hidden_dim_transformer, args.num_heads_transformer, args.num_layers_transformer)
        self.classifier = MultipleCauseClassifier(args.input_dim_transformer * 2, args.max_convo_len, args.max_convo_len) # output dim of transformer as input dim and we concat 2 of them hence *2

    def forward(self, input_embeddings, emotion_idxs):
        modified_embeddings = self.transformer_model(input_embeddings)
        # Create pair of given emotion utt with all other utt in a convo
        utt_pairs = []
        for idx, convo in enumerate(modified_embeddings):
            pairs = []
            emotion_id = emotion_idxs[idx]
            for j in range(len(convo)):
                pair = torch.cat((convo[j], convo[emotion_id]))
                pairs.append(pair)
            utt_pairs.append(pairs)
        # Convert to torch tensor
        utt_pairs = get_stacked_tensor(utt_pairs)
        # Classify each pair as having cause or not 
        probabilities = self.classifier(utt_pairs)
        return probabilities

