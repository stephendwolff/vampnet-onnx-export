#!/usr/bin/env python3
"""
Fix C2F NaN issues by checking activation and normalization.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from vampnet.modules.transformer import VampNet
from lac.model.lac import LAC as DAC
from torch.nn.utils import remove_weight_norm


def check_gated_gelu():
    """Check if GatedGELU implementation matches VampNet."""
    print("="*80)
    print("CHECKING GATED GELU IMPLEMENTATION")
    print("="*80)
    
    # Load VampNet to check activation
    vampnet = VampNet.load("models/vampnet/c2f.pth", map_location='cpu')
    vampnet.eval()
    
    # Check FFN structure
    print("\n1. Checking FFN structure...")
    layer0_ffn = vampnet.transformer.layers[0].feed_forward
    print(f"   FFN type: {type(layer0_ffn)}")
    print(f"   FFN modules: {list(layer0_ffn._modules.keys())}")
    
    # Check activation
    if hasattr(layer0_ffn, 'activation'):
        print(f"   Activation type: {type(layer0_ffn.activation)}")
    
    # Check dimensions
    print(f"\n2. Checking FFN dimensions...")
    print(f"   w_1: {layer0_ffn.w_1.weight.shape}")
    print(f"   w_2: {layer0_ffn.w_2.weight.shape}")
    
    # The issue might be the GatedGELU expecting 2x the input dimension
    d_model = 1280
    d_inner_expected = layer0_ffn.w_1.weight.shape[0]
    print(f"   d_inner from weights: {d_inner_expected}")
    print(f"   Expected for GatedGELU: d_model * 2 = {d_model * 2}")
    print(f"   Actual: {d_inner_expected}")
    
    # Test activation
    print(f"\n3. Testing activation...")
    test_input = torch.randn(1, 10, d_model)
    
    with torch.no_grad():
        # Through w_1
        x = layer0_ffn.w_1(test_input)
        print(f"   After w_1: {x.shape}")
        print(f"   Has NaN: {torch.isnan(x).any()}")
        
        # Through activation
        x_act = layer0_ffn.act(x)
        print(f"   After activation: {x_act.shape}")
        print(f"   Has NaN: {torch.isnan(x_act).any()}")
        
        # Check activation type
        print(f"   Activation type: {type(layer0_ffn.act)}")
        
    # Check if we need to adjust the FFN dimension
    print(f"\n4. Analyzing FFN dimension mismatch...")
    if d_inner_expected == d_model * 4:  # 5120
        print("   ✓ FFN uses 4x expansion with GatedGELU (input gets split in half)")
        print("   Effective dimension after split: 2560")
    elif d_inner_expected == d_model * 2:  # 2560
        print("   ✓ FFN uses 2x expansion (no split needed)")
    
    print("="*80)


def test_fixed_implementation():
    """Test a fixed implementation."""
    print("\n\nTESTING FIXED IMPLEMENTATION")
    print("="*80)
    
    # Create a fixed FFN that matches VampNet
    class FixedFeedForward(nn.Module):
        def __init__(self, d_model, d_inner, dropout=0.0):
            super().__init__()
            self.w_1 = nn.Linear(d_model, d_inner, bias=False)
            self.w_2 = nn.Linear(d_inner // 2, d_model, bias=False)  # Note: d_inner // 2
            self.dropout = nn.Dropout(dropout)
            
            # Custom activation
            self.activation = nn.GELU()  # Use standard GELU for testing
            
        def forward(self, x):
            # Project to d_inner
            x = self.w_1(x)
            
            # Split for gating
            x, gate = x.chunk(2, dim=-1)
            
            # Apply activation to gate and multiply
            x = x * self.activation(gate)
            
            # Dropout and project back
            x = self.dropout(x)
            x = self.w_2(x)
            return x
    
    # Test it
    ffn = FixedFeedForward(1280, 5120, 0.1)
    ffn.eval()
    
    test_input = torch.randn(1, 10, 1280)
    with torch.no_grad():
        output = ffn(test_input)
        print(f"Input shape: {test_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Has NaN: {torch.isnan(output).any()}")
    
    print("="*80)


if __name__ == "__main__":
    check_gated_gelu()
    test_fixed_implementation()