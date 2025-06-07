"""
Debug FFN weight transfer issues, specifically the second layer.
"""

import torch
import vampnet
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.export_vampnet_transformer import VampNetTransformerONNX


def analyze_ffn_weights():
    """Analyze FFN weight shapes and names in both models."""
    
    print("=== FFN Weight Analysis ===\n")
    
    # Load VampNet
    interface = vampnet.interface.Interface(
        codec_ckpt="../models/vampnet/codec.pth",
        coarse_ckpt="../models/vampnet/coarse.pth",
        wavebeat_ckpt="../models/vampnet/wavebeat.pth",
    )
    
    vampnet_model = interface.coarse
    if hasattr(vampnet_model, '_orig_mod'):
        vampnet_model = vampnet_model._orig_mod
    
    vampnet_state = vampnet_model.state_dict()
    
    # Create ONNX model
    onnx_model = VampNetTransformerONNX(
        n_codebooks=4,
        vocab_size=1024,
        d_model=1280,
        n_heads=20,
        n_layers=20
    )
    
    print("=== VampNet FFN Weights (Layer 0) ===")
    print("\nAll FFN-related weights:")
    ffn_weights = {}
    for name, param in vampnet_state.items():
        if 'layers.0' in name and ('feed_forward' in name or 'w_1' in name or 'w_2' in name):
            ffn_weights[name] = param
            print(f"{name}: {param.shape}")
    
    print("\n=== ONNX FFN Structure (Layer 0) ===")
    ffn_module = onnx_model.layers[0]['ffn']
    print(f"FFN module type: {type(ffn_module)}")
    print(f"FFN submodules: {list(ffn_module.children())}")
    
    print("\nONNX FFN weights:")
    print(f"ffn[0] (first linear): {ffn_module[0].weight.shape}")
    print(f"ffn[0] bias: {ffn_module[0].bias.shape if ffn_module[0].bias is not None else 'None'}")
    print(f"ffn[1] (activation): {ffn_module[1]}")
    print(f"ffn[2] (second linear): {ffn_module[2].weight.shape}")
    print(f"ffn[2] bias: {ffn_module[2].bias.shape if ffn_module[2].bias is not None else 'None'}")
    
    # Analyze weight dimensions
    print("\n=== Dimension Analysis ===")
    
    # Expected dimensions
    d_model = 1280
    d_ff = d_model * 4  # 5120
    
    print(f"\nExpected dimensions:")
    print(f"d_model: {d_model}")
    print(f"d_ff (d_model * 4): {d_ff}")
    print(f"First linear: ({d_ff}, {d_model}) -> maps {d_model} to {d_ff}")
    print(f"Second linear: ({d_model}, {d_ff}) -> maps {d_ff} to {d_model}")
    
    # Check VampNet w_2 weight
    print("\n=== VampNet w_2 Analysis ===")
    w2_candidates = []
    for name, param in vampnet_state.items():
        if 'w_2' in name and 'weight' in name:
            w2_candidates.append((name, param.shape))
            if 'layers.0' in name:
                print(f"\nLayer 0 w_2: {name}")
                print(f"Shape: {param.shape}")
                print(f"Expected ONNX shape: ({d_model}, {d_ff})")
                
                # Check if dimensions match or need transpose
                if param.shape == (d_model, d_ff):
                    print("✓ Dimensions match exactly!")
                elif param.shape == (d_ff, d_model):
                    print("⚠️ Dimensions are transposed")
                elif param.shape[0] == d_model:
                    print(f"⚠️ First dimension matches, but second is {param.shape[1]} instead of {d_ff}")
                else:
                    print("✗ Dimensions don't match")
    
    # Show all w_2 weights found
    print("\nAll w_2 weights found:")
    for name, shape in w2_candidates[:5]:
        print(f"  {name}: {shape}")
    
    # Check for LoRA weights
    print("\n=== LoRA Analysis ===")
    lora_weights = []
    for name, param in vampnet_state.items():
        if 'lora' in name.lower() and 'layers.0' in name:
            lora_weights.append((name, param.shape))
    
    if lora_weights:
        print("Found LoRA weights in layer 0:")
        for name, shape in lora_weights:
            print(f"  {name}: {shape}")
        print("\nNote: LoRA weights suggest low-rank adaptation is used")
        print("The actual weight might need to be reconstructed from base + LoRA")
    
    return vampnet_state, onnx_model


def check_w2_shapes_all_layers(vampnet_state):
    """Check w_2 shapes across all layers."""
    
    print("\n\n=== W_2 Shapes Across All Layers ===")
    
    for layer_idx in range(20):
        w2_name = f'transformer.layers.{layer_idx}.feed_forward.w_2.weight'
        if w2_name in vampnet_state:
            shape = vampnet_state[w2_name].shape
            print(f"Layer {layer_idx}: {shape}")
            
            # Check for inconsistencies
            if layer_idx == 0:
                first_shape = shape
            elif shape != first_shape:
                print(f"  ⚠️ Shape differs from layer 0!")


def test_weight_assignment(vampnet_state, onnx_model):
    """Test different weight assignment strategies."""
    
    print("\n\n=== Testing Weight Assignment ===")
    
    # Get w_2 weight from VampNet layer 0
    w2_name = 'transformer.layers.0.feed_forward.w_2.weight'
    if w2_name not in vampnet_state:
        print(f"✗ {w2_name} not found in VampNet state")
        return
    
    w2_vampnet = vampnet_state[w2_name]
    print(f"\nVampNet w_2 shape: {w2_vampnet.shape}")
    
    # Get ONNX FFN module
    ffn_module = onnx_model.layers[0]['ffn']
    w2_onnx = ffn_module[2].weight
    print(f"ONNX ffn[2] shape: {w2_onnx.shape}")
    
    # Test assignment
    print("\nTesting assignment strategies:")
    
    # Strategy 1: Direct assignment
    if w2_vampnet.shape == w2_onnx.shape:
        print("✓ Strategy 1: Direct assignment would work")
        print(f"  Shapes match: {w2_vampnet.shape}")
    else:
        print("✗ Strategy 1: Direct assignment won't work")
        print(f"  VampNet: {w2_vampnet.shape}, ONNX: {w2_onnx.shape}")
    
    # Strategy 2: Transpose
    if w2_vampnet.T.shape == w2_onnx.shape:
        print("✓ Strategy 2: Transpose would work")
        print(f"  VampNet.T: {w2_vampnet.T.shape} matches ONNX: {w2_onnx.shape}")
    else:
        print("✗ Strategy 2: Transpose won't work")
    
    # Strategy 3: Check actual dimensions
    print("\nDimension analysis:")
    print(f"VampNet w_2: in_features={w2_vampnet.shape[1]}, out_features={w2_vampnet.shape[0]}")
    print(f"ONNX ffn[2]: in_features={w2_onnx.shape[1]}, out_features={w2_onnx.shape[0]}")
    
    # Check if it's a LoRA issue
    print("\n=== Checking for LoRA in w_2 ===")
    lora_a = f'transformer.layers.0.feed_forward.w_2.lora_A'
    lora_b = f'transformer.layers.0.feed_forward.w_2.lora_B'
    
    if lora_a in vampnet_state and lora_b in vampnet_state:
        print("Found LoRA weights for w_2:")
        print(f"  LoRA A: {vampnet_state[lora_a].shape}")
        print(f"  LoRA B: {vampnet_state[lora_b].shape}")
        print("  Note: The effective weight might be base + LoRA_B @ LoRA_A")


def suggest_fix():
    """Suggest how to fix the weight transfer."""
    
    print("\n\n=== Suggested Fix ===")
    
    print("\nThe issue appears to be with the FFN second layer dimensions.")
    print("\nVampNet FFN structure:")
    print("  w_1: (5120, 1280) - expands from 1280 to 5120")
    print("  w_2: (1280, 2560) - reduces from 2560 to 1280")
    print("\nONNX FFN structure:")
    print("  ffn[0]: (5120, 1280) - expands from 1280 to 5120")
    print("  ffn[2]: (1280, 5120) - reduces from 5120 to 1280")
    
    print("\nThe mismatch is in the intermediate dimension:")
    print("  VampNet uses 2560 as intermediate size")
    print("  ONNX expects 5120 (d_model * 4)")
    
    print("\nPossible solutions:")
    print("1. VampNet might use a different intermediate size (2*d_model instead of 4*d_model)")
    print("2. There might be grouped/factorized layers")
    print("3. LoRA decomposition might be affecting the dimensions")
    
    print("\nTo fix in transfer_weights_improved.py:")
    print("- Check the actual intermediate dimension used by VampNet")
    print("- Adjust the ONNX model creation to match")
    print("- Or handle the dimension mismatch in weight transfer")


if __name__ == "__main__":
    # Change to scripts directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Analyze FFN weights
    vampnet_state, onnx_model = analyze_ffn_weights()
    
    # Check all layers
    check_w2_shapes_all_layers(vampnet_state)
    
    # Test weight assignment
    test_weight_assignment(vampnet_state, onnx_model)
    
    # Suggest fix
    suggest_fix()