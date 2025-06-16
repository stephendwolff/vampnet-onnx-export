#!/usr/bin/env python3
"""
Simple test to compare VampNet and V9 outputs.
"""

import torch
from pathlib import Path
import sys
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from vampnet.modules.transformer import VampNet
from lac.model.lac import LAC as DAC
from scripts.export_vampnet_transformer_v9_proper_flow import VampNetTransformerV9, transfer_weights_v9

# Create a deterministic test
torch.manual_seed(42)
np.random.seed(42)

# Load models
print("Loading models...")
vampnet = VampNet.load("models/vampnet/coarse.pth", map_location='cpu')
vampnet.eval()
codec = DAC.load(Path("models/vampnet/codec.pth"))
codec.eval()

# Create V9 model
model = VampNetTransformerV9(
    n_codebooks=4,
    n_conditioning_codebooks=0,
    vocab_size=1024,
    latent_dim=8,
    d_model=1280,
    n_heads=20,
    n_layers=20
)

transfer_weights_v9("models/vampnet/coarse.pth", model, "models/vampnet/codec.pth")
model.eval()

# Test with smaller input to reduce noise
codes = torch.randint(0, 1024, (1, 4, 10))
mask = torch.zeros((1, 4, 10), dtype=torch.bool)
mask[:, :, 5:] = True
masked_codes = codes.clone()
masked_codes[mask] = 1024

print("\nComparing outputs...")

with torch.no_grad():
    # Get latents
    latents = vampnet.embedding.from_codes(masked_codes, codec)
    print(f"Latents shape: {latents.shape}")
    print(f"Latents stats: mean={latents.mean():.4f}, std={latents.std():.4f}")
    
    # Full forward pass
    vampnet_out = vampnet(latents)
    v9_out = model(latents)
    
    print(f"\nVampNet output: {vampnet_out.shape}")
    print(f"V9 output: {v9_out.shape}")
    
    # Reshape for comparison
    vampnet_reshaped = vampnet_out.reshape(1, 4, 10, 1024)
    v9_truncated = v9_out[:, :, :, :1024]
    
    # Statistics
    print(f"\nVampNet stats: mean={vampnet_reshaped.mean():.4f}, std={vampnet_reshaped.std():.4f}")
    print(f"V9 stats: mean={v9_truncated.mean():.4f}, std={v9_truncated.std():.4f}")
    
    # Correlation
    corr = np.corrcoef(vampnet_reshaped.flatten(), v9_truncated.flatten())[0,1]
    print(f"\nCorrelation: {corr:.4f}")
    
    # Difference
    diff = (vampnet_reshaped - v9_truncated).abs()
    print(f"Mean absolute difference: {diff.mean():.6f}")
    print(f"Max absolute difference: {diff.max():.6f}")
    
    # Check a few specific values
    print(f"\nSample logits at position [0, 0, 0, :5]:")
    print(f"VampNet: {vampnet_reshaped[0, 0, 0, :5]}")
    print(f"V9:      {v9_truncated[0, 0, 0, :5]}")
    
    # Try with fresh random input to see if it's input-specific
    print("\n\nTrying with different random seed...")
    torch.manual_seed(123)
    codes2 = torch.randint(0, 1024, (1, 4, 10))
    masked_codes2 = codes2.clone()
    masked_codes2[:, :, 5:] = 1024
    
    latents2 = vampnet.embedding.from_codes(masked_codes2, codec)
    vampnet_out2 = vampnet(latents2)
    v9_out2 = model(latents2)
    
    vampnet_reshaped2 = vampnet_out2.reshape(1, 4, 10, 1024)
    v9_truncated2 = v9_out2[:, :, :, :1024]
    
    corr2 = np.corrcoef(vampnet_reshaped2.flatten(), v9_truncated2.flatten())[0,1]
    print(f"Correlation with different input: {corr2:.4f}")
    
    # Check if it's a scaling issue
    scale_factor = vampnet_reshaped.std() / v9_truncated.std()
    print(f"\nScale factor (VampNet std / V9 std): {scale_factor:.4f}")
    
    # Try scaling V9 output
    v9_scaled = v9_truncated * scale_factor
    diff_scaled = (vampnet_reshaped - v9_scaled).abs()
    print(f"After scaling - Mean difference: {diff_scaled.mean():.6f}")
    
    corr_scaled = np.corrcoef(vampnet_reshaped.flatten(), v9_scaled.flatten())[0,1]
    print(f"After scaling - Correlation: {corr_scaled:.4f}")