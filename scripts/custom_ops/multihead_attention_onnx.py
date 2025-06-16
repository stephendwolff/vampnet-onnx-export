"""
ONNX-compatible Multi-Head Attention implementation.
PyTorch's nn.MultiheadAttention has issues with dynamic shapes in ONNX.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class OnnxMultiheadAttention(nn.Module):
    """
    ONNX-friendly implementation of multi-head attention.
    
    This avoids the dynamic shape issues in PyTorch's nn.MultiheadAttention.
    """
    
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_k)
        
        # Linear projections (VampNet uses bias=False)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        
        # Dropout (disabled for ONNX)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, query, key, value, attn_mask=None):
        """
        Args:
            query: [batch, seq_len, d_model]
            key: [batch, seq_len, d_model]
            value: [batch, seq_len, d_model]
            attn_mask: Optional attention mask
            
        Returns:
            output: [batch, seq_len, d_model]
            attn_weights: [batch, n_heads, seq_len, seq_len]
        """
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        # 1. Linear projections
        Q = self.w_q(query)  # [batch, seq_len, d_model]
        K = self.w_k(key)    # [batch, seq_len, d_model]
        V = self.w_v(value)  # [batch, seq_len, d_model]
        
        # 2. Reshape for multi-head attention
        # [batch, seq_len, n_heads, d_k]
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k)
        
        # 3. Transpose for attention computation
        # [batch, n_heads, seq_len, d_k]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # 4. Compute attention scores
        # [batch, n_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # 5. Apply attention mask if provided
        if attn_mask is not None:
            scores = scores + attn_mask
        
        # 6. Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 7. Apply attention to values
        # [batch, n_heads, seq_len, d_k]
        context = torch.matmul(attn_weights, V)
        
        # 8. Concatenate heads
        # [batch, seq_len, n_heads, d_k]
        context = context.transpose(1, 2).contiguous()
        # [batch, seq_len, d_model]
        context = context.view(batch_size, seq_len, self.d_model)
        
        # 9. Final linear projection
        output = self.w_o(context)
        
        return output, attn_weights


def test_onnx_attention():
    """Test the ONNX-compatible attention."""
    
    print("Testing ONNX Multi-Head Attention...")
    
    # Create model
    d_model = 512
    n_heads = 8
    model = OnnxMultiheadAttention(d_model, n_heads)
    model.eval()
    
    # Test with different sequence lengths
    for seq_len in [10, 50, 100]:
        x = torch.randn(2, seq_len, d_model)
        
        with torch.no_grad():
            output, weights = model(x, x, x)
            
        print(f"Seq length {seq_len}: output shape = {output.shape}, weights shape = {weights.shape}")
    
    # Export to ONNX
    print("\nExporting to ONNX...")
    dummy_input = torch.randn(1, 50, d_model)
    
    torch.onnx.export(
        model,
        (dummy_input, dummy_input, dummy_input),
        "test_attention.onnx",
        input_names=['query', 'key', 'value'],
        output_names=['output', 'attn_weights'],
        dynamic_axes={
            'query': {0: 'batch', 1: 'sequence'},
            'key': {0: 'batch', 1: 'sequence'},
            'value': {0: 'batch', 1: 'sequence'},
            'output': {0: 'batch', 1: 'sequence'},
            'attn_weights': {0: 'batch', 2: 'sequence', 3: 'sequence'}
        },
        opset_version=14,
        verbose=False
    )
    
    print("✓ Export successful!")
    
    # Clean up
    import os
    if os.path.exists("test_attention.onnx"):
        os.remove("test_attention.onnx")


if __name__ == "__main__":
    test_onnx_attention()