#!/usr/bin/env python3
"""
Debug the classifier/output projection difference.
"""

import torch
from pathlib import Path
import sys
from einops import rearrange

sys.path.append(str(Path(__file__).parent.parent))

from vampnet.modules.transformer import VampNet
from lac.model.lac import LAC as DAC
from scripts.export_vampnet_transformer_v10_all_relative import VampNetTransformerV10, transfer_weights_v10

# Load models
vampnet = VampNet.load("models/vampnet/coarse.pth", map_location='cpu')
vampnet.eval()
codec = DAC.load(Path("models/vampnet/codec.pth"))
codec.eval()

model = VampNetTransformerV10()
transfer_weights_v10("models/vampnet/coarse.pth", model, "models/vampnet/codec.pth")
model.eval()

print("Debugging classifier...")

# Test input
torch.manual_seed(42)
test_input = torch.randn(1, 10, 1280)  # After transformer

with torch.no_grad():
    # VampNet classifier
    print("1. VampNet classifier:")
    print(f"   Type: {type(vampnet.classifier)}")
    print(f"   Modules: {list(vampnet.classifier.named_children())}")
    
    # VampNet needs rearrange before classifier
    vamp_input = rearrange(test_input, "b n d -> b d n")
    print(f"   Input shape: {vamp_input.shape}")
    
    vamp_out = vampnet.classifier(vamp_input, None)
    print(f"   Output shape: {vamp_out.shape}")
    print(f"   Output stats: mean={vamp_out.mean():.4f}, std={vamp_out.std():.4f}")
    
    # Get the Conv1d layer
    conv_layer = vampnet.classifier.layers[0]
    print(f"\n   Conv1d details:")
    print(f"   In channels: {conv_layer.in_channels}")
    print(f"   Out channels: {conv_layer.out_channels}")
    print(f"   Kernel size: {conv_layer.kernel_size}")
    print(f"   Weight shape: {conv_layer.weight.shape}")
    print(f"   Bias shape: {conv_layer.bias.shape}")
    
    # V10 output projections
    print("\n2. V10 output projections:")
    v10_outputs = []
    for i, proj in enumerate(model.output_projs):
        out = proj(test_input)
        v10_outputs.append(out)
        print(f"   Proj {i}: {out.shape}, mean={out.mean():.4f}, std={out.std():.4f}")
    
    # Stack and compare
    v10_stacked = torch.stack(v10_outputs, dim=1)  # [batch, n_codebooks, seq_len, vocab_size+1]
    print(f"\n   V10 stacked: {v10_stacked.shape}")
    
    # To match VampNet, we need to rearrange
    # VampNet: [batch, out_channels, seq_len] where out_channels = n_codebooks * vocab_size
    # V10: [batch, n_codebooks, seq_len, vocab_size+1]
    
    # Truncate to vocab_size and rearrange
    v10_truncated = v10_stacked[:, :, :, :1024]  # Remove mask token
    v10_rearranged = rearrange(v10_truncated, "b c t v -> b (c v) t")
    print(f"   V10 rearranged: {v10_rearranged.shape}")
    
    # Compare
    print(f"\n3. Comparison:")
    print(f"   Shapes match: {vamp_out.shape == v10_rearranged.shape}")
    
    if vamp_out.shape == v10_rearranged.shape:
        diff = (vamp_out - v10_rearranged).abs()
        print(f"   Mean diff: {diff.mean():.6f}")
        print(f"   Max diff: {diff.max():.6f}")
        
        # Check specific channels
        print(f"\n   Checking specific channels:")
        for i in range(4):
            start = i * 1024
            end = (i + 1) * 1024
            vamp_cb = vamp_out[:, start:end, :]
            v10_cb = v10_rearranged[:, start:end, :]
            cb_diff = (vamp_cb - v10_cb).abs()
            print(f"   Codebook {i}: mean diff={cb_diff.mean():.6f}, max diff={cb_diff.max():.6f}")
    
    # Check weight transfer
    print("\n4. Checking weight transfer:")
    
    # VampNet Conv1d weight is [out_channels, in_channels, kernel_size]
    # We split it into separate Linear layers
    vamp_weight = conv_layer.weight.squeeze(-1)  # Remove kernel dimension
    vamp_bias = conv_layer.bias
    
    for i in range(4):
        start = i * 1024
        end = (i + 1) * 1024
        
        # Compare weights
        vamp_w = vamp_weight[start:end]
        v10_w = model.output_projs[i].weight[:1024]
        
        w_match = torch.allclose(vamp_w, v10_w, atol=1e-5)
        print(f"   Codebook {i} weight match: {w_match}")
        
        if not w_match:
            w_diff = (vamp_w - v10_w).abs()
            print(f"     Weight diff: mean={w_diff.mean():.6f}, max={w_diff.max():.6f}")
        
        # Compare bias
        vamp_b = vamp_bias[start:end]
        v10_b = model.output_projs[i].bias[:1024]
        
        b_match = torch.allclose(vamp_b, v10_b, atol=1e-5)
        print(f"   Codebook {i} bias match: {b_match}")
        
        if not b_match:
            b_diff = (vamp_b - v10_b).abs()
            print(f"     Bias diff: mean={b_diff.mean():.6f}, max={b_diff.max():.6f}")