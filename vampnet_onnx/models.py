"""
ONNX-compatible VampNet model architectures.

This module provides PyTorch implementations of VampNet models that are designed
to be ONNX-exportable with custom operators.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SimpleRMSNorm(nn.Module):
    """ONNX-compatible RMS normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS normalization
        norm = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x / norm * self.weight


class SimpleFiLM(nn.Module):
    """ONNX-compatible Feature-wise Linear Modulation."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.shift


class VerySimpleCodebookEmbedding(nn.Module):
    """ONNX-compatible codebook embedding layer."""
    
    def __init__(self, n_codebooks: int, vocab_size: int, dim: int):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.vocab_size = vocab_size
        self.dim = dim
        
        # Embedding weights
        self.weight = nn.Parameter(torch.randn(n_codebooks, vocab_size, dim))
        
    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            indices: [batch_size, seq_len, n_codebooks]
            
        Returns:
            embeddings: [batch_size, seq_len, dim]
        """
        batch_size, seq_len, n_codebooks = indices.shape
        
        # Gather embeddings for each codebook
        embeddings = []
        for i in range(n_codebooks):
            idx = indices[:, :, i]  # [batch_size, seq_len]
            emb = torch.nn.functional.embedding(idx, self.weight[i])  # [batch_size, seq_len, dim]
            embeddings.append(emb)
            
        # Sum across codebooks
        output = torch.stack(embeddings, dim=0).sum(dim=0)
        return output


class OnnxMultiheadAttention(nn.Module):
    """ONNX-compatible multi-head attention."""
    
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
        # Combined QKV projection
        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, n_heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.dim)
        output = self.o_proj(attn_output)
        
        return output


class GatedFFN(nn.Module):
    """Gated feed-forward network matching VampNet's architecture."""
    
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w_gated = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gated GELU activation
        gate = torch.nn.functional.gelu(self.w_gated(x))
        up = self.w_up(x)
        return self.w_down(gate * up)


class TransformerLayer(nn.Module):
    """Single transformer layer with ONNX-compatible components."""
    
    def __init__(self, dim: int, n_heads: int, hidden_dim: int):
        super().__init__()
        self.norm1 = SimpleRMSNorm(dim)
        self.self_attn = OnnxMultiheadAttention(dim, n_heads)
        self.norm2 = SimpleRMSNorm(dim)
        self.ffn = GatedFFN(dim, hidden_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, mask)
        x = residual + x
        
        # FFN with residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        
        return x


class VampNetTransformer(nn.Module):
    """ONNX-compatible VampNet transformer model."""
    
    def __init__(
        self,
        n_codebooks: int,
        vocab_size: int,
        dim: int,
        n_heads: int,
        n_layers: int,
        hidden_dim: int,
        n_classes: int
    ):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.n_classes = n_classes
        
        # Token embeddings
        self.token_embedding = VerySimpleCodebookEmbedding(n_codebooks, vocab_size, dim)
        
        # Positional embedding
        self.positional_embedding = nn.Parameter(torch.randn(1, 100, dim))
        
        # Transformer layers
        self.transformer = nn.ModuleList([
            TransformerLayer(dim, n_heads, hidden_dim)
            for _ in range(n_layers)
        ])
        self.final_norm = SimpleRMSNorm(dim)
        
        # Output projections (one per codebook)
        self.output_proj = nn.ModuleList([
            nn.Linear(dim, n_classes, bias=False)
            for _ in range(n_codebooks)
        ])
        
    def forward(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            tokens: [batch_size, seq_len, n_codebooks]
            mask: Optional attention mask
            
        Returns:
            logits: [batch_size, seq_len, n_codebooks, n_classes]
        """
        batch_size, seq_len, _ = tokens.shape
        
        # Token embeddings
        x = self.token_embedding(tokens)
        
        # Add positional embeddings
        x = x + self.positional_embedding[:, :seq_len, :]
        
        # Apply transformer layers
        for layer in self.transformer:
            x = layer(x, mask)
            
        x = self.final_norm(x)
        
        # Project to output logits for each codebook
        logits = []
        for i in range(self.n_codebooks):
            logits.append(self.output_proj[i](x))
            
        # Stack logits
        logits = torch.stack(logits, dim=2)  # [batch, seq_len, n_codebooks, n_classes]
        
        return logits


class CoarseTransformer(VampNetTransformer):
    """Coarse transformer model (4 codebooks)."""
    
    def __init__(self, vocab_size: int = 1025, dim: int = 1280, n_heads: int = 20, 
                 n_layers: int = 48, n_classes: int = 1025):
        super().__init__(
            n_codebooks=4,
            vocab_size=vocab_size,
            dim=dim,
            n_heads=n_heads,
            n_layers=n_layers,
            hidden_dim=2560,  # VampNet uses 2x dim for FFN
            n_classes=n_classes
        )


class C2FTransformer(VampNetTransformer):
    """Coarse-to-Fine transformer model (10 fine codebooks)."""
    
    def __init__(self, vocab_size: int = 1025, dim: int = 768, n_heads: int = 12,
                 n_layers: int = 24, n_classes: int = 1025):
        super().__init__(
            n_codebooks=10,  # 10 fine codebooks
            vocab_size=vocab_size,
            dim=dim,
            n_heads=n_heads,
            n_layers=n_layers,
            hidden_dim=1536,  # 2x dim for FFN
            n_classes=n_classes
        )