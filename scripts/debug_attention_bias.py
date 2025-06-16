#!/usr/bin/env python3
"""
Check if VampNet attention layers use bias.
"""

import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from vampnet.modules.transformer import VampNet

# Load VampNet
vampnet = VampNet.load("models/vampnet/coarse.pth", map_location='cpu')

print("Checking VampNet attention layer biases...")

# Check first layer (relative attention)
layer0 = vampnet.transformer.layers[0]
print("\nLayer 0 (relative attention):")
print(f"  w_qs has bias: {layer0.self_attn.w_qs.bias is not None}")
print(f"  w_ks has bias: {layer0.self_attn.w_ks.bias is not None}")
print(f"  w_vs has bias: {layer0.self_attn.w_vs.bias is not None}")
print(f"  fc has bias: {layer0.self_attn.fc.bias is not None}")

# Check second layer (standard attention)
layer1 = vampnet.transformer.layers[1]
print("\nLayer 1 (standard attention):")
print(f"  w_qs has bias: {layer1.self_attn.w_qs.bias is not None}")
print(f"  w_ks has bias: {layer1.self_attn.w_ks.bias is not None}")
print(f"  w_vs has bias: {layer1.self_attn.w_vs.bias is not None}")
print(f"  fc has bias: {layer1.self_attn.fc.bias is not None}")

# Check FFN biases
print("\nFFN biases:")
print(f"  w_1 has bias: {layer0.feed_forward.w_1.bias is not None}")
print(f"  w_2 has bias: {layer0.feed_forward.w_2.bias is not None}")

# Check our attention implementation
print("\n\nChecking our attention implementations...")
from scripts.custom_ops.multihead_attention_onnx import OnnxMultiheadAttention
from scripts.custom_ops.relative_attention_onnx import OnnxMultiheadRelativeAttention

our_std_attn = OnnxMultiheadAttention(1280, 20)
print("\nOur standard attention:")
print(f"  w_q has bias: {our_std_attn.w_q.bias is not None}")
print(f"  w_k has bias: {our_std_attn.w_k.bias is not None}")
print(f"  w_v has bias: {our_std_attn.w_v.bias is not None}")
print(f"  w_o has bias: {our_std_attn.w_o.bias is not None}")

our_rel_attn = OnnxMultiheadRelativeAttention(1280, 20, 0.1)
print("\nOur relative attention:")
print(f"  w_qs has bias: {our_rel_attn.w_qs.bias is not None}")
print(f"  w_ks has bias: {our_rel_attn.w_ks.bias is not None}")
print(f"  w_vs has bias: {our_rel_attn.w_vs.bias is not None}")
print(f"  fc has bias: {our_rel_attn.fc.bias is not None}")

if our_std_attn.w_q.bias is not None:
    print("\n⚠️ Our attention has bias but VampNet doesn't! This needs to be fixed.")