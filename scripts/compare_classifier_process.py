#!/usr/bin/env python3
"""
Compare VampNet classifier process step-by-step with V10.
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

print("Comparing classifier process step-by-step...")

# Test input - use the same transformer output
torch.manual_seed(42)
test_input = torch.randn(1, 10, 1280)  # [batch, seq_len, d_model]

with torch.no_grad():
    # 1. VampNet classifier process
    print("\n1. VampNet classifier process:")
    print(f"   Input shape: {test_input.shape}")
    
    # VampNet rearranges: "b n d -> b d n"
    vamp_rearranged = rearrange(test_input, "b n d -> b d n")
    print(f"   After rearrange: {vamp_rearranged.shape}")
    
    # Pass through classifier (SequentialWithFiLM)
    print(f"   Classifier type: {type(vampnet.classifier)}")
    vamp_classifier_out = vampnet.classifier(vamp_rearranged, None)
    print(f"   Classifier output: {vamp_classifier_out.shape}")
    
    # Final rearrange: "b (p c) t -> b p (t c)"
    vamp_final = rearrange(vamp_classifier_out, "b (p c) t -> b p (t c)", c=vampnet.n_predict_codebooks)
    print(f"   Final output: {vamp_final.shape}")
    print(f"   n_predict_codebooks: {vampnet.n_predict_codebooks}")
    
    # 2. V10 process
    print("\n2. V10 classifier process:")
    print(f"   Input shape: {test_input.shape}")
    
    # V10 applies projections directly
    v10_outputs = []
    for i, proj in enumerate(model.output_projs):
        out = proj(test_input)
        v10_outputs.append(out)
        print(f"   Projection {i}: {out.shape}")
    
    # Stack outputs
    v10_stacked = torch.stack(v10_outputs, dim=1)
    print(f"   Stacked output: {v10_stacked.shape}")
    
    # 3. Compare intermediate values
    print("\n3. Comparing intermediate values:")
    
    # Let's manually apply VampNet's Conv1d
    conv_weight = vampnet.classifier.layers[0].weight.data  # [4096, 1280, 1]
    conv_bias = vampnet.classifier.layers[0].bias.data      # [4096]
    
    # Manual Conv1d computation
    manual_conv_out = torch.nn.functional.conv1d(vamp_rearranged, conv_weight, conv_bias)
    print(f"   Manual Conv1d output: {manual_conv_out.shape}")
    print(f"   Matches VampNet classifier: {torch.allclose(manual_conv_out, vamp_classifier_out)}")
    
    # Now let's replicate this with Linear operations
    print("\n4. Replicating Conv1d with Linear:")
    
    # For each position in sequence
    for pos in range(test_input.shape[1]):
        if pos > 2:  # Just check first 3 positions
            break
            
        print(f"\n   Position {pos}:")
        
        # Get input at this position
        input_vec = test_input[0, pos, :]  # [1280]
        
        # VampNet Conv1d output at this position
        vamp_pos = vamp_classifier_out[0, :, pos]  # [4096]
        
        # V10 Linear outputs at this position
        v10_pos = []
        for i in range(4):
            proj_out = model.output_projs[i](input_vec.unsqueeze(0))  # Add batch dim
            v10_pos.append(proj_out[0, :1024])  # Remove batch dim and mask token
        v10_pos = torch.cat(v10_pos)  # [4096]
        
        # Compare
        diff = (vamp_pos - v10_pos).abs()
        print(f"   VampNet vs V10: mean diff = {diff.mean():.6f}, max diff = {diff.max():.6f}")
        
        # Check specific codebook
        for cb in range(4):
            start = cb * 1024
            end = (cb + 1) * 1024
            cb_diff = (vamp_pos[start:end] - v10_pos[start:end]).abs()
            print(f"   Codebook {cb}: mean = {cb_diff.mean():.6f}, max = {cb_diff.max():.6f}")
    
    # 5. Check weight application
    print("\n5. Checking weight application manually:")
    
    # Take first codebook as example
    cb0_weight = conv_weight[:1024, :, 0]  # [1024, 1280]
    cb0_bias = conv_bias[:1024]            # [1024]
    
    # Manual computation for first position
    input_vec = test_input[0, 0, :]  # [1280]
    
    # Conv1d style: weight @ input + bias
    manual_out = cb0_weight @ input_vec + cb0_bias
    
    # V10 Linear
    v10_out = model.output_projs[0](input_vec.unsqueeze(0))[0, :1024]
    
    print(f"   Manual computation shape: {manual_out.shape}")
    print(f"   V10 computation shape: {v10_out.shape}")
    print(f"   Outputs match: {torch.allclose(manual_out, v10_out, atol=1e-5)}")
    
    if not torch.allclose(manual_out, v10_out, atol=1e-5):
        diff = (manual_out - v10_out).abs()
        print(f"   Difference: mean = {diff.mean():.6f}, max = {diff.max():.6f}")
        
        # Check weights
        v10_weight = model.output_projs[0].weight.data[:1024]
        v10_bias = model.output_projs[0].bias.data[:1024]
        
        weight_match = torch.allclose(cb0_weight, v10_weight)
        bias_match = torch.allclose(cb0_bias, v10_bias)
        
        print(f"   Weights match: {weight_match}")
        print(f"   Bias match: {bias_match}")
        
        if not weight_match:
            w_diff = (cb0_weight - v10_weight).abs()
            print(f"   Weight diff: mean = {w_diff.mean():.6f}, max = {w_diff.max():.6f}")
            
            # Check a specific row
            print(f"\n   First weight row comparison:")
            print(f"   VampNet: {cb0_weight[0, :5]}")
            print(f"   V10:     {v10_weight[0, :5]}")