#!/usr/bin/env python3
"""
Check VampNet's relative position computation.
"""

import torch
from vampnet.modules.transformer import MultiHeadRelativeAttention
import inspect

# Get the source
print("VampNet MultiHeadRelativeAttention._relative_position_bucket:")
print(inspect.getsource(MultiHeadRelativeAttention._relative_position_bucket))

print("\n\nVampNet MultiHeadRelativeAttention.compute_bias:")
print(inspect.getsource(MultiHeadRelativeAttention.compute_bias))

# Test it
print("\n\nTesting VampNet's relative attention:")
# Check the actual __init__ signature
print("MultiHeadRelativeAttention.__init__ signature:")
print(inspect.signature(MultiHeadRelativeAttention.__init__))

attn = MultiHeadRelativeAttention(
    dim=1280,
    heads=20,
    dropout=0.1,
    bidirectional=True,
    has_relative_attention_bias=True,
    attention_num_buckets=32,
    attention_max_distance=128
)

# Test compute_bias
bias = attn.compute_bias(10, 10)
print(f"Bias shape: {bias.shape}")
print(f"Bias stats: mean={bias.mean():.4f}, std={bias.std():.4f}")

# Test with our implementation
from scripts.custom_ops.relative_attention_onnx import OnnxMultiheadRelativeAttention

our_attn = OnnxMultiheadRelativeAttention(
    d_model=1280,
    n_heads=20,
    dropout=0.1,
    bidirectional=True,
    has_relative_attention_bias=True,
    attention_num_buckets=32,
    attention_max_distance=128
)

# Copy weights
our_attn.relative_attention_bias.weight.data = attn.relative_attention_bias.weight.data

# Test our compute_bias
our_bias = our_attn.compute_bias(10, 10)
print(f"\nOur bias shape: {our_bias.shape}")
print(f"Our bias stats: mean={our_bias.mean():.4f}, std={our_bias.std():.4f}")

# Compare
diff = (bias - our_bias).abs()
print(f"\nDifference: mean={diff.mean():.6f}, max={diff.max():.6f}")