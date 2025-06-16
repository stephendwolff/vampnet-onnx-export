"""
ONNX-compatible implementation of VampNet's MultiHeadRelativeAttention.
This includes the relative position bias computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from einops import rearrange


class OnnxMultiheadRelativeAttention(nn.Module):
    """
    Multi-head attention with relative position bias, compatible with ONNX export.
    Matches VampNet's MultiHeadRelativeAttention implementation.
    """
    
    def __init__(self, 
                 d_model: int, 
                 n_heads: int, 
                 dropout: float = 0.0,
                 bidirectional: bool = True,
                 has_relative_attention_bias: bool = True,
                 attention_num_buckets: int = 32,
                 attention_max_distance: int = 128):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.bidirectional = bidirectional
        self.has_relative_attention_bias = has_relative_attention_bias
        self.attention_num_buckets = attention_num_buckets
        self.attention_max_distance = attention_max_distance
        
        # Linear projections (matching VampNet's naming)
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.fc = nn.Linear(d_model, d_model, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Relative attention bias
        if has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(
                self.attention_num_buckets, self.n_heads
            )
    
    def _relative_position_bucket(self, relative_position, num_buckets=32, max_distance=128):
        """
        Adapted from VampNet's implementation.
        Translates relative position to a bucket number for relative attention.
        """
        relative_buckets = 0
        
        if self.bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        
        # Half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        
        # The other half of the buckets are for logarithmically bigger bins
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact) 
            / math.log(max_distance / max_exact) 
            * (num_buckets - max_exact)
        ).to(torch.long)
        
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1),
        )
        
        relative_buckets += torch.where(
            is_small, relative_position, relative_position_if_large
        )
        
        return relative_buckets
    
    def compute_bias(self, query_length, key_length):
        """Compute position bias for each position in query_length x key_length."""
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        
        relative_position = memory_position - context_position
        
        rp_bucket = self._relative_position_bucket(
            relative_position,
            num_buckets=self.attention_num_buckets,
            max_distance=self.attention_max_distance,
        )
        
        # Get bias values from embedding
        values = self.relative_attention_bias(rp_bucket)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # [1, n_heads, T_q, T_kv]
        
        return values
    
    def forward(self, query, key, value, mask=None, position_bias=None):
        """
        Forward pass matching VampNet's MultiHeadRelativeAttention.
        
        Args:
            query: [batch, seq_len, d_model]
            key: [batch, seq_len, d_model]
            value: [batch, seq_len, d_model]
            mask: Optional attention mask
            position_bias: Optional pre-computed position bias
            
        Returns:
            output: [batch, seq_len, d_model]
            position_bias: The computed position bias (for caching)
        """
        batch_size, seq_len_q = query.shape[:2]
        seq_len_kv = key.shape[1]
        
        # Linear projections and reshape for multi-head
        q = self.w_qs(query).view(batch_size, seq_len_q, self.n_heads, self.d_head)
        k = self.w_ks(key).view(batch_size, seq_len_kv, self.n_heads, self.d_head)
        v = self.w_vs(value).view(batch_size, seq_len_kv, self.n_heads, self.d_head)
        
        # Transpose for attention: [batch, n_heads, seq_len, d_head]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        
        # Add relative position bias
        if position_bias is None and self.has_relative_attention_bias:
            position_bias = self.compute_bias(seq_len_q, seq_len_kv)
            position_bias = position_bias.to(scores.device)
        
        if position_bias is not None:
            scores = scores + position_bias
        
        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)
        
        # Transpose back: [batch, seq_len, n_heads, d_head]
        output = output.transpose(1, 2).contiguous()
        
        # Concatenate heads
        output = output.view(batch_size, seq_len_q, self.d_model)
        
        # Final linear projection
        output = self.fc(output)
        
        return output, position_bias