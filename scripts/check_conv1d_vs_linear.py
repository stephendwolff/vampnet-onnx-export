#!/usr/bin/env python3
"""
Check Conv1d vs Linear operations.
"""

import torch
import torch.nn as nn
from einops import rearrange

# Test shapes
batch = 1
seq_len = 10
in_features = 1280
out_features = 1024

# Create test input
x = torch.randn(batch, seq_len, in_features)

# Conv1d approach (like VampNet)
conv = nn.Conv1d(in_features, out_features, kernel_size=1)
# Conv1d expects [batch, channels, length]
x_conv = rearrange(x, "b n d -> b d n")
conv_out = conv(x_conv)
print(f"Conv1d input: {x_conv.shape}")
print(f"Conv1d output: {conv_out.shape}")
print(f"Conv1d weight shape: {conv.weight.shape}")

# Linear approach (like V10)
linear = nn.Linear(in_features, out_features)
# Linear expects [batch, seq_len, features]
linear_out = linear(x)
print(f"\nLinear input: {x.shape}")
print(f"Linear output: {linear_out.shape}")
print(f"Linear weight shape: {linear.weight.shape}")

# Copy Conv1d weights to Linear
print("\n\nTesting weight transfer...")

# Method 1: Direct copy
linear.weight.data = conv.weight.squeeze(-1)
linear.bias.data = conv.bias
linear_out1 = linear(x)
conv_out_rearranged = rearrange(conv_out, "b d n -> b n d")
diff1 = (linear_out1 - conv_out_rearranged).abs().max()
print(f"Direct copy - Max diff: {diff1:.6f}")

# Method 2: Transpose
linear.weight.data = conv.weight.squeeze(-1).t()
linear_out2 = linear(x)
diff2 = (linear_out2 - conv_out_rearranged).abs().max()
print(f"Transpose - Max diff: {diff2:.6f}")

# Let's understand the math
print("\n\nUnderstanding the operations:")
print("Conv1d: y = conv(x) where x is [batch, in_channels, length]")
print("        For kernel_size=1, this is: y[b,o,l] = sum_i(W[o,i] * x[b,i,l]) + b[o]")
print("Linear: y = x @ W.T + b where x is [batch, seq, in_features]")
print("        This is: y[b,s,o] = sum_i(x[b,s,i] * W[o,i]) + b[o]")
print("\nSo if we rearrange Conv1d input/output properly, the weight matrices are the same!")