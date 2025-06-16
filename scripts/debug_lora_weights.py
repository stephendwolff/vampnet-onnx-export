#!/usr/bin/env python3
"""
Debug LoRA weights in VampNet.
"""

import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

# Load VampNet
from vampnet.modules.transformer import VampNet
vampnet = VampNet.load("models/vampnet/coarse.pth", map_location='cpu')
vampnet.eval()

print("Checking LoRA layers in VampNet...")

# Check first transformer layer
layer = vampnet.transformer.layers[0]

# Check self-attention
print("\n1. Self-attention:")
for name, module in layer.self_attn.named_modules():
    if hasattr(module, 'lora_A'):
        print(f"  {name} has LoRA adaptation")
        print(f"    Base weight shape: {module.weight.shape}")
        print(f"    LoRA A shape: {module.lora_A.shape}")
        print(f"    LoRA B shape: {module.lora_B.shape}")
        print(f"    LoRA rank: {module.r}")
        
        # Check if LoRA is merged
        if hasattr(module, 'merged'):
            print(f"    Merged: {module.merged}")

# Check FFN
print("\n2. Feed-forward network:")
for name, module in layer.feed_forward.named_modules():
    if hasattr(module, 'lora_A'):
        print(f"  {name} has LoRA adaptation")
        print(f"    Base weight shape: {module.weight.shape}")
        print(f"    LoRA A shape: {module.lora_A.shape}")
        print(f"    LoRA B shape: {module.lora_B.shape}")
        print(f"    LoRA rank: {module.r}")
        
        # Check if LoRA is merged
        if hasattr(module, 'merged'):
            print(f"    Merged: {module.merged}")

# Test LoRA forward pass
print("\n3. Testing LoRA forward pass...")
x = torch.randn(1, 10, 1280)

# Get w_qs output with LoRA
w_qs = layer.self_attn.w_qs
with torch.no_grad():
    # Standard linear forward
    out_standard = torch.nn.functional.linear(x, w_qs.weight, w_qs.bias)
    
    # Check if LoRA modifies this
    out_lora = w_qs(x)
    
    diff = (out_standard - out_lora).abs().max()
    print(f"  Difference between standard and LoRA forward: {diff:.6f}")
    
    if diff > 0.001:
        print("  ⚠️ LoRA is modifying the output!")
        
        # Check LoRA contribution
        if hasattr(w_qs, 'lora_A') and hasattr(w_qs, 'lora_B'):
            lora_out = x @ w_qs.lora_A.T @ w_qs.lora_B.T * w_qs.scaling
            print(f"  LoRA contribution magnitude: {lora_out.abs().mean():.4f}")

# Check if we need to merge LoRA weights
print("\n4. Checking if LoRA weights need merging...")
total_lora_layers = 0
for name, module in vampnet.named_modules():
    if hasattr(module, 'lora_A'):
        total_lora_layers += 1

print(f"Total LoRA layers found: {total_lora_layers}")

if total_lora_layers > 0:
    print("\n⚠️ VampNet uses LoRA adaptation!")
    print("You may need to merge LoRA weights before exporting.")