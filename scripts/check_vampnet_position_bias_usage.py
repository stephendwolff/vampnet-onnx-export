#!/usr/bin/env python3
"""
Check how VampNet actually uses position bias.
"""

import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from vampnet.modules.transformer import VampNet
from lac.model.lac import LAC as DAC

# Load VampNet
vampnet = VampNet.load("models/vampnet/coarse.pth", map_location='cpu')
vampnet.eval()
codec = DAC.load(Path("models/vampnet/codec.pth"))
codec.eval()

# Check transformer forward
import inspect
print("VampNet Transformer forward signature:")
print(inspect.signature(vampnet.transformer.forward))

# Test
torch.manual_seed(42)
codes = torch.randint(0, 1024, (1, 4, 10))
masked_codes = codes.clone()
masked_codes[:, :, 5:] = 1024

with torch.no_grad():
    latents = vampnet.embedding.from_codes(masked_codes, codec)
    
    # Full forward
    print("\nRunning VampNet forward...")
    out = vampnet(latents)
    print(f"Output shape: {out.shape}")
    
    # Check if VampNet uses position bias throughout
    print("\n\nChecking each layer:")
    for i, layer in enumerate(vampnet.transformer.layers):
        print(f"\nLayer {i}:")
        print(f"  Type: {type(layer.self_attn).__name__}")
        if hasattr(layer.self_attn, 'relative_attention_bias'):
            print(f"  Has relative_attention_bias: True")
            print(f"  Bias shape: {layer.self_attn.relative_attention_bias.weight.shape}")
        else:
            print(f"  Has relative_attention_bias: False")
    
    # Check the actual transformer code
    print("\n\nVampNet Transformer class:")
    print(f"Type: {type(vampnet.transformer)}")
    
    # Look for position_bias in the source
    transformer_source = inspect.getsource(vampnet.transformer.__class__.forward)
    if "position_bias" in transformer_source:
        print("\nTransformer forward() uses position_bias")
        # Find the relevant lines
        lines = transformer_source.split('\n')
        for i, line in enumerate(lines):
            if "position_bias" in line:
                print(f"  Line {i}: {line.strip()}")