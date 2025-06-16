#!/usr/bin/env python3
"""
Debug the FFN issue in detail.
"""

import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from vampnet.modules.transformer import VampNet
from scripts.export_vampnet_transformer_v9_proper_flow import GatedGELU, NewGELU

# Load VampNet
vampnet = VampNet.load("models/vampnet/coarse.pth", map_location='cpu')
vampnet.eval()

# Get VampNet's FFN
vamp_ffn = vampnet.transformer.layers[0].feed_forward

print("Checking FFN structure...")
print(f"VampNet FFN type: {type(vamp_ffn)}")
print(f"VampNet FFN modules: {list(vamp_ffn.named_children())}")

# Check dimensions
print(f"\nVampNet FFN weights:")
print(f"  w_1: {vamp_ffn.w_1.weight.shape}")
print(f"  w_2: {vamp_ffn.w_2.weight.shape}")

# Create our FFN
our_ffn = GatedGELU(1280)

# Transfer weights
our_ffn.w_1.weight.data = vamp_ffn.w_1.weight.data
our_ffn.w_2.weight.data = vamp_ffn.w_2.weight.data

print(f"\nOur FFN weights:")
print(f"  w_1: {our_ffn.w_1.weight.shape}")
print(f"  w_2: {our_ffn.w_2.weight.shape}")

# Test with same input
torch.manual_seed(42)
x = torch.randn(1, 10, 1280)

with torch.no_grad():
    # VampNet forward
    vamp_out = vamp_ffn(x)
    
    # Our forward
    our_out = our_ffn(x)
    
    # Compare
    diff = (vamp_out - our_out).abs()
    print(f"\nFFN output comparison:")
    print(f"  Mean diff: {diff.mean():.6f}")
    print(f"  Max diff: {diff.max():.6f}")
    
    # Debug intermediate steps
    print(f"\nDebug intermediate steps:")
    
    # VampNet's forward
    vamp_hidden = vamp_ffn.w_1(x)
    print(f"  VampNet w_1 output shape: {vamp_hidden.shape}")
    
    # Check activation
    print(f"  VampNet activation: {vamp_ffn.act}")
    
    # Our forward step by step
    our_hidden = our_ffn.w_1(x)
    print(f"  Our w_1 output shape: {our_hidden.shape}")
    
    # Check if issue is in chunking
    our_h1, our_h2 = our_hidden.chunk(2, dim=-1)
    print(f"  Our chunk shapes: {our_h1.shape}, {our_h2.shape}")
    
    # Apply activation
    our_activated = our_h1 * our_ffn.activation(our_h2)
    print(f"  Our activated shape: {our_activated.shape}")
    
    # Final projection
    our_final = our_ffn.w_2(our_activated)
    
    manual_diff = (vamp_out - our_final).abs()
    print(f"\nManual forward diff: {manual_diff.max():.6f}")

# Check VampNet's actual activation
print("\n\nChecking VampNet's activation implementation...")
print(f"VampNet act type: {type(vamp_ffn.act)}")

# Check if it's really GatedGELU
if hasattr(vamp_ffn.act, 'forward'):
    # Test the activation (VampNet's GatedGELU expects even dimension for chunking)
    test_input = torch.randn(1, 10)
    
    # VampNet's GatedGELU will chunk this, so output will be half size
    vamp_act_out = vamp_ffn.act(test_input)
    print(f"VampNet activation output shape: {vamp_act_out.shape}")
    
    # For fair comparison, we need to test with NewGELU on half the input
    p1, p2 = test_input.chunk(2, dim=-1)
    our_gelu = NewGELU()
    our_gated_out = p1 * our_gelu(p2)
    print(f"Our gated output shape: {our_gated_out.shape}")
    
    if vamp_act_out.shape == our_gated_out.shape:
        act_diff = (vamp_act_out - our_gated_out).abs().max()
        print(f"Activation diff: {act_diff:.6f}")
    
    # Test gated activation properly
    test_gated = torch.randn(1, 20)
    
    # VampNet's gated activation (it chunks internally)
    vamp_gated = vamp_ffn.act(test_gated)
    print(f"VampNet gated output shape: {vamp_gated.shape}")
    
    # Our gated computation
    p1, p2 = test_gated.chunk(2, dim=-1)
    our_gated = p1 * our_gelu(p2)
    print(f"Our gated output shape: {our_gated.shape}")
    
    if vamp_gated.shape == our_gated.shape:
        gated_diff = (vamp_gated - our_gated).abs().max()
        print(f"Gated activation diff: {gated_diff:.6f}")

# Check VampNet's actual FFN forward implementation
print("\n\nTracing VampNet FFN forward...")
import inspect
print(inspect.getsource(vamp_ffn.forward))