#!/usr/bin/env python3
"""
Debug C2F value ranges to understand the NaN issue.
"""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from vampnet.modules.transformer import VampNet
from lac.model.lac import LAC as DAC
from torch.nn.utils import remove_weight_norm


def debug_value_ranges():
    """Debug value ranges in C2F to understand explosion."""
    print("="*80)
    print("DEBUGGING C2F VALUE RANGES")
    print("="*80)
    
    # Load VampNet C2F
    vampnet_c2f = VampNet.load("models/vampnet/c2f.pth", map_location='cpu')
    try:
        remove_weight_norm(vampnet_c2f.classifier.layers[0])
    except:
        pass
    vampnet_c2f.eval()
    
    # Codec
    codec = DAC.load(Path("models/vampnet/codec.pth"))
    codec.eval()
    
    # Create different test inputs
    print("\n1. Testing with different inputs...")
    
    # Try different seeds
    for seed in [42, 123, 999]:
        print(f"\n--- Seed {seed} ---")
        torch.manual_seed(seed)
        
        # Create test input
        test_codes = torch.randint(0, 1024, (1, 14, 10))
        z = codec.quantizer.from_codes(test_codes)
        test_latents = z[1]
        
        print(f"Latents range: [{test_latents.min():.4f}, {test_latents.max():.4f}]")
        
        # Forward pass
        with torch.no_grad():
            try:
                output = vampnet_c2f(test_latents)
                print(f"Output shape: {output.shape}")
                print(f"Has NaN: {torch.isnan(output).any()}")
                if not torch.isnan(output).any():
                    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
            except Exception as e:
                print(f"Error: {e}")
    
    # Test with zero input
    print("\n2. Testing with zero latents...")
    zero_latents = torch.zeros(1, 112, 10)
    with torch.no_grad():
        output = vampnet_c2f(zero_latents)
        print(f"Output shape: {output.shape}")
        print(f"Has NaN: {torch.isnan(output).any()}")
        if not torch.isnan(output).any():
            print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    # Test with small random values
    print("\n3. Testing with small random latents...")
    small_latents = torch.randn(1, 112, 10) * 0.1
    with torch.no_grad():
        output = vampnet_c2f(small_latents)
        print(f"Output shape: {output.shape}")
        print(f"Has NaN: {torch.isnan(output).any()}")
        if not torch.isnan(output).any():
            print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    # Check if it's specific to the test codes
    print("\n4. Testing with mask token codes...")
    mask_codes = torch.full((1, 14, 10), 1024)  # All mask tokens
    z = codec.quantizer.from_codes(mask_codes)
    mask_latents = z[1]
    
    print(f"Mask latents range: [{mask_latents.min():.4f}, {mask_latents.max():.4f}]")
    
    with torch.no_grad():
        output = vampnet_c2f(mask_latents)
        print(f"Output shape: {output.shape}")
        print(f"Has NaN: {torch.isnan(output).any()}")
        if not torch.isnan(output).any():
            print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    print("="*80)


if __name__ == "__main__":
    debug_value_ranges()