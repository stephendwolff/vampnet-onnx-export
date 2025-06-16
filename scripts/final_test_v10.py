#!/usr/bin/env python3
"""
Final test of V10 model.
"""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from vampnet.modules.transformer import VampNet
from lac.model.lac import LAC as DAC
from scripts.export_vampnet_transformer_v10_all_relative import VampNetTransformerV10, transfer_weights_v10
from einops import rearrange

# Deterministic
torch.manual_seed(42)
np.random.seed(42)

print("Final V10 test...")

# Load models
vampnet = VampNet.load("models/vampnet/coarse.pth", map_location='cpu')
vampnet.eval()
codec = DAC.load(Path("models/vampnet/codec.pth"))
codec.eval()

# Create V10
model = VampNetTransformerV10()
transfer_weights_v10("models/vampnet/coarse.pth", model, "models/vampnet/codec.pth")
model.eval()

# Test
codes = torch.randint(0, 1024, (1, 4, 10))
masked_codes = codes.clone()
masked_codes[:, :, 5:] = 1024

with torch.no_grad():
    latents = vampnet.embedding.from_codes(masked_codes, codec)
    
    # Run both models
    vampnet_out = vampnet(latents)
    v10_out = model(latents)
    
    print(f"VampNet output: {vampnet_out.shape}")
    print(f"V10 output: {v10_out.shape}")
    
    # Correct reshape for VampNet
    vampnet_reshaped = rearrange(vampnet_out, "b p (t c) -> b c t p", c=4, t=10)
    v10_truncated = v10_out[:, :, :, :1024]
    
    # Compare
    diff = (vampnet_reshaped - v10_truncated).abs()
    corr = np.corrcoef(vampnet_reshaped.flatten(), v10_truncated.flatten())[0,1]
    
    print(f"\nMean absolute difference: {diff.mean():.6f}")
    print(f"Max absolute difference: {diff.max():.6f}")
    print(f"Correlation: {corr:.4f}")
    
    if corr > 0.99:
        print("\n✅ SUCCESS! V10 model matches VampNet!")
        
        # Export to ONNX
        print("\nExporting V10 to ONNX...")
        dummy_latents = torch.randn(1, 32, 100)  # 100 sequence length
        
        torch.onnx.export(
            model,
            dummy_latents,
            "vampnet_transformer_v10.onnx",
            input_names=['latents'],
            output_names=['logits'],
            dynamic_axes={
                'latents': {0: 'batch', 2: 'sequence'},
                'logits': {0: 'batch', 2: 'sequence'}
            },
            opset_version=14,
            verbose=False
        )
        print("✓ Exported to vampnet_transformer_v10.onnx")
        
    else:
        print("\n❌ Still not matching...")
        
        # Debug the issue
        print("\nDebugging the mismatch...")
        
        # Check a few specific logits
        print("\nSample logits:")
        for i in range(4):
            print(f"\nCodebook {i}, position 0:")
            print(f"VampNet: {vampnet_reshaped[0, i, 0, :5]}")
            print(f"V10:     {v10_truncated[0, i, 0, :5]}")
            
        # Check if it's a systematic offset
        mean_diff_per_codebook = []
        for i in range(4):
            cb_diff = (vampnet_reshaped[0, i] - v10_truncated[0, i]).mean()
            mean_diff_per_codebook.append(cb_diff.item())
            print(f"\nCodebook {i} mean difference: {cb_diff:.6f}")
            
        # Export anyway for debugging
        print("\nExporting V10 to ONNX for debugging...")
        dummy_latents = torch.randn(1, 32, 100)
        torch.onnx.export(
            model,
            dummy_latents,
            "vampnet_transformer_v10_debug.onnx",
            input_names=['latents'],
            output_names=['logits'],
            dynamic_axes={
                'latents': {0: 'batch', 2: 'sequence'},
                'logits': {0: 'batch', 2: 'sequence'}
            },
            opset_version=14,
            verbose=False
        )
        print("✓ Exported to vampnet_transformer_v10_debug.onnx")