"""
Analyze VampNet's FFN structure to understand the dimension mismatch.
"""

import torch
import vampnet
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def inspect_vampnet_model():
    """Inspect the actual VampNet model structure."""
    
    print("=== Inspecting VampNet Model Structure ===\n")
    
    # Load VampNet
    interface = vampnet.interface.Interface(
        codec_ckpt="../models/vampnet/codec.pth",
        coarse_ckpt="../models/vampnet/coarse.pth",
        wavebeat_ckpt="../models/vampnet/wavebeat.pth",
    )
    
    vampnet_model = interface.coarse
    if hasattr(vampnet_model, '_orig_mod'):
        vampnet_model = vampnet_model._orig_mod
    
    # Print model structure
    print("=== Full Model Structure ===")
    print(vampnet_model)
    
    # Focus on FFN structure
    print("\n\n=== FFN Module Analysis ===")
    
    # Try to access the FFN module directly
    if hasattr(vampnet_model, 'transformer'):
        transformer = vampnet_model.transformer
        if hasattr(transformer, 'layers'):
            layer0 = transformer.layers[0]
            print(f"\nLayer 0 type: {type(layer0)}")
            print(f"Layer 0 modules: {list(layer0._modules.keys())}")
            
            if hasattr(layer0, 'feed_forward'):
                ff = layer0.feed_forward
                print(f"\nFeed-forward type: {type(ff)}")
                print(f"Feed-forward structure:")
                print(ff)
                
                # Check for specific attributes
                if hasattr(ff, 'w_1'):
                    print(f"\nw_1 shape: {ff.w_1.weight.shape}")
                if hasattr(ff, 'w_2'):
                    print(f"w_2 shape: {ff.w_2.weight.shape}")
                if hasattr(ff, 'w_3'):
                    print(f"w_3 shape: {ff.w_3.weight.shape if hasattr(ff, 'w_3') else 'No w_3'}")
    
    # Check if it's using SwiGLU or similar
    print("\n\n=== Checking for SwiGLU/Gated Linear Units ===")
    
    # Look for any w_3 weights in state dict
    state_dict = vampnet_model.state_dict()
    w3_weights = [name for name in state_dict.keys() if 'w_3' in name]
    
    if w3_weights:
        print("Found w_3 weights (suggests gated activation):")
        for name in w3_weights[:5]:
            print(f"  {name}: {state_dict[name].shape}")
        
        # Check layer 0 specifically
        w3_layer0 = [name for name in w3_weights if 'layers.0' in name]
        if w3_layer0:
            print(f"\nLayer 0 w_3: {w3_layer0[0]}")
            print(f"Shape: {state_dict[w3_layer0[0]].shape}")
    else:
        print("No w_3 weights found")
    
    # Analyze activation pattern
    print("\n\n=== FFN Activation Analysis ===")
    
    # Get all weights for layer 0 FFN
    layer0_ffn = {}
    for name, param in state_dict.items():
        if 'layers.0.feed_forward' in name:
            layer0_ffn[name] = param.shape
    
    print("All Layer 0 FFN weights:")
    for name, shape in sorted(layer0_ffn.items()):
        print(f"  {name}: {shape}")
    
    # Deduce the architecture
    print("\n\n=== Architecture Deduction ===")
    
    if 'transformer.layers.0.feed_forward.w_3.weight' in state_dict:
        w1_shape = state_dict['transformer.layers.0.feed_forward.w_1.weight'].shape
        w2_shape = state_dict['transformer.layers.0.feed_forward.w_2.weight'].shape
        w3_shape = state_dict['transformer.layers.0.feed_forward.w_3.weight'].shape
        
        print("VampNet appears to use SwiGLU or similar gated activation:")
        print(f"  w_1: {w1_shape} - Linear projection")
        print(f"  w_2: {w2_shape} - Down projection")
        print(f"  w_3: {w3_shape} - Gate projection")
        print("\nTypical SwiGLU pattern:")
        print("  hidden = x")
        print("  gate = w_3(x)")
        print("  hidden = w_1(x)")
        print("  hidden = swish(gate) * hidden  # or silu(gate) * hidden")
        print("  output = w_2(hidden)")
        print("\nThis explains why w_2 expects 2560 input instead of 5120!")
    else:
        print("Standard FFN pattern detected")
    
    return vampnet_model, state_dict


def create_fixed_onnx_ffn():
    """Show how to create an ONNX model with correct FFN dimensions."""
    
    print("\n\n=== Suggested ONNX FFN Fix ===")
    
    print("\nOption 1: Modify ONNX model to match VampNet")
    print("```python")
    print("# In VampNetTransformerONNX.__init__")
    print("# Replace the FFN with:")
    print("'ffn': nn.Sequential(")
    print("    nn.Linear(d_model, d_model * 2),  # 1280 -> 2560")
    print("    nn.GELU(),")
    print("    nn.Linear(d_model * 2, d_model)   # 2560 -> 1280")
    print(")")
    print("```")
    
    print("\nOption 2: Handle in weight transfer")
    print("```python")
    print("# Skip w_2 if dimensions don't match")
    print("# Or pad/truncate the weight matrix")
    print("```")
    
    print("\nOption 3: Implement SwiGLU in ONNX")
    print("```python")
    print("class SwiGLU(nn.Module):")
    print("    def __init__(self, d_model):")
    print("        super().__init__()")
    print("        self.w_1 = nn.Linear(d_model, d_model * 4)")
    print("        self.w_2 = nn.Linear(d_model * 2, d_model)")
    print("        self.w_3 = nn.Linear(d_model, d_model * 2)")
    print("    ")
    print("    def forward(self, x):")
    print("        gate = self.w_3(x)")
    print("        hidden = self.w_1(x)")
    print("        # Note: Need to handle dimension mismatch here")
    print("        hidden = F.silu(gate) * hidden[:, :gate.size(1)]")
    print("        return self.w_2(hidden)")
    print("```")


if __name__ == "__main__":
    # Change to scripts directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Inspect model
    model, state_dict = inspect_vampnet_model()
    
    # Suggest fixes
    create_fixed_onnx_ffn()