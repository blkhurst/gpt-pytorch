"""Based on GPT-2 architecture (with pre-normalization and GELU activation)"""

import torch
from torch import nn
import torch.nn.functional as F

# class SelfAttention(nn.Module):
#     def __init__(self, embedding_dim: int):
#         super().__init__()
#         self.embedding_dim = embedding_dim
#         self.W_q = nn.Linear(embedding_dim, embedding_dim, bias=False)
#         self.W_k = nn.Linear(embedding_dim, embedding_dim, bias=False)
#         self.W_v = nn.Linear(embedding_dim, embedding_dim, bias=False)

#     def forward(self, inputs):
#         B, T, D = inputs.shape
#         Q = self.W_q(inputs)
#         K = self.W_k(inputs)
#         V = self.W_v(inputs)

#         d_k = self.embedding_dim
#         attention_scores = Q @ K.transpose(-2, -1) / d_k**0.5
#         attention_weights = torch.softmax(attention_scores, dim=-1)
#         return attention_weights @ V


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dim: int, n_heads: int):
        super().__init__()
        if n_heads <= 0 or embedding_dim % n_heads != 0:
            raise ValueError(
                "n_heads must be positive and divide embedding_dim evenly."
            )

        self.n_heads = n_heads
        self.head_dim = embedding_dim // n_heads

        # Learned query, key, value and output projections
        self.W_q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_k = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_v = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_o = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, inputs):
        # Batch, Sequence Length, Embedding Dimension
        B, T, D = inputs.shape

        # Project inputs into queries, keys and values
        Q = self.W_q(inputs)
        K = self.W_k(inputs)
        V = self.W_v(inputs)

        # Split into heads - [B, T, D] -> [B, n_heads, T, head_dim]
        Q = Q.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # # Prepare the causal mask - [T, T]
        # mask = torch.triu(torch.ones(T, T, device=inputs.device, dtype=torch.bool), diagonal=1)

        # # Calculate scores by comparing queries and keys within each head
        # # Mask future tokens before softmax - [T, T]
        # # Convert scores to weights and combine values
        # attention_scores = Q @ K.transpose(-2, -1) / self.head_dim**0.5
        # attention_scores = attention_scores.masked_fill(mask, float("-inf"))
        # attention_weights = torch.softmax(attention_scores, dim=-1)
        # context = attention_weights @ V

        # Scaled dot-product attention with causal masking replaces manual block above.
        context = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

        # Combine heads - [B, n_heads, T, head_dim] -> [B, T, D]
        context = context.transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(context)


class FeedForwardNeuralNetwork(nn.Module):
    def __init__(self, embedding_dim: int, hidden_unit_multiplier: int = 4):
        super().__init__()

        self.fc1 = nn.Linear(embedding_dim, embedding_dim * hidden_unit_multiplier)
        self.fc2 = nn.Linear(embedding_dim * hidden_unit_multiplier, embedding_dim)
        self.act = nn.GELU(approximate="tanh")

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim: int, n_heads: int):
        super().__init__()

        self.ln1 = nn.LayerNorm(embedding_dim)
        self.attn = CausalMultiHeadSelfAttention(embedding_dim, n_heads)
        self.ln2 = nn.LayerNorm(embedding_dim)
        self.mlp = FeedForwardNeuralNetwork(embedding_dim)  # Multi-Layer Perceptron

    def forward(self, x):
        # Using pre-normalization
        x = x + self.attn(self.ln1(x))  # Residual connection
        x = x + self.mlp(self.ln2(x))  # Residual connection
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size=50_000,
        embedding_dim=768,
        context_length=1024,
        n_heads=6,
        n_layers=6,
    ):
        super().__init__()
        self.context_length = context_length

        # Embedding Layer & Learned Positional Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(context_length, embedding_dim)

        # Transformer Blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(embedding_dim, n_heads) for _ in range(n_layers)]
        )

        # Final Layer Norm
        self.ln_f = nn.LayerNorm(embedding_dim)

        # Output Linear Layer; Share output weights with token embeddings #? GPT2 design
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, token_ids):
        # Batch, Sequence Length
        B, T = token_ids.shape
        if T > self.context_length:
            raise ValueError("Input exceeds context length")

        # Retrieve Token Embedding and Positional Embedding
        positions = torch.arange(T, device=token_ids.device)
        embeddings = self.token_embedding(token_ids)
        positional_embeddings = self.position_embedding(positions)
        x = embeddings + positional_embeddings

        # Pass through the transformer blocks
        for block in self.blocks:
            x = block(x)

        # Apply final normalization
        x = self.ln_f(x)

        # Produce Logits [B, T, vocab_size]
        return self.lm_head(x)
