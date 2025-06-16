#!/usr/bin/env python3
"""
Debug C2F NaN issues.
"""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from scripts.export_c2f_transformer_v13_final import C2FTransformerV13, transfer_weights_c2f_v13
from vampnet.modules.transformer import VampNet
from lac.model.lac import LAC as DAC
from torch.nn.utils import remove_weight_norm


def debug_c2f_nan():
    """Debug why C2F produces NaN values."""
    print("="*80)
    print("DEBUGGING C2F NAN ISSUES")
    print("="*80)
    
    # Load models
    print("\n1. Loading models...")
    
    # Our model
    model = C2FTransformerV13()
    transfer_weights_c2f_v13("models/vampnet/c2f.pth", model, "models/vampnet/codec.pth")
    model.eval()
    
    # VampNet C2F
    vampnet_c2f = VampNet.load("models/vampnet/c2f.pth", map_location='cpu')
    try:
        remove_weight_norm(vampnet_c2f.classifier.layers[0])
    except:
        pass
    vampnet_c2f.eval()
    
    # Codec
    codec = DAC.load(Path("models/vampnet/codec.pth"))
    codec.eval()
    
    # Create test input
    print("\n2. Creating test input...")
    batch_size = 1
    seq_len = 10
    test_codes = torch.randint(0, 1024, (batch_size, 14, seq_len))
    
    # Convert to latents
    z = codec.quantizer.from_codes(test_codes)
    test_latents = z[1]  # Get latents
    print(f"   Input latents shape: {test_latents.shape}")
    print(f"   Latents min: {test_latents.min():.4f}, max: {test_latents.max():.4f}")
    print(f"   Latents has NaN: {torch.isnan(test_latents).any()}")
    
    # Forward pass through both models
    print("\n3. Forward pass...")
    with torch.no_grad():
        # VampNet
        vampnet_output = vampnet_c2f(test_latents)
        print(f"   VampNet output shape: {vampnet_output.shape}")
        print(f"   VampNet has NaN: {torch.isnan(vampnet_output).any()}")
        print(f"   VampNet has Inf: {torch.isinf(vampnet_output).any()}")
        print(f"   VampNet min: {vampnet_output.min():.4f}, max: {vampnet_output.max():.4f}")
        
        # Our model
        our_output = model(test_latents)
        print(f"\n   Our output shape: {our_output.shape}")
        print(f"   Our has NaN: {torch.isnan(our_output).any()}")
        print(f"   Our has Inf: {torch.isinf(our_output).any()}")
        if not torch.isnan(our_output).any():
            print(f"   Our min: {our_output.min():.4f}, max: {our_output.max():.4f}")
        
        # Check intermediate values
        print("\n4. Checking intermediate values...")
        
        # Embedding output
        x = model.embedding(test_latents)
        print(f"   After embedding - has NaN: {torch.isnan(x).any()}, has Inf: {torch.isinf(x).any()}")
        
        # Check each layer
        from einops import rearrange
        x = rearrange(x, "b d n -> b n d")
        
        for i, layer in enumerate(model.transformer_layers):
            residual = x
            x = layer['norm'](x)
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"   Layer {i} after norm - has NaN/Inf!")
                break
                
            x = layer['self_attn'](x)
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"   Layer {i} after attention - has NaN/Inf!")
                break
                
            x = residual + x
            x = layer['film'](x)
            
            residual = x
            x = layer['ffn'](x)
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"   Layer {i} after FFN - has NaN/Inf!")
                break
                
            x = residual + x
        
        if not (torch.isnan(x).any() or torch.isinf(x).any()):
            print("   All transformer layers OK")
            
            # Check final layers
            x = model.norm_out(x)
            print(f"   After final norm - has NaN: {torch.isnan(x).any()}, has Inf: {torch.isinf(x).any()}")
            
            x = rearrange(x, "b n d -> b d n")
            out = model.classifier(x)
            print(f"   After classifier - has NaN: {torch.isnan(out).any()}, has Inf: {torch.isinf(out).any()}")
            
            # Check the rearrange
            out_rearranged = rearrange(out, "b (p c) t -> b p (t c)", c=model.n_predict_codebooks)
            print(f"   After rearrange - has NaN: {torch.isnan(out_rearranged).any()}, has Inf: {torch.isinf(out_rearranged).any()}")
    
    # Compare outputs where both are valid
    print("\n5. Comparing valid outputs...")
    if not (torch.isnan(vampnet_output).any() or torch.isnan(our_output).any()):
        diff = torch.abs(vampnet_output - our_output).max()
        print(f"   Max difference: {diff:.6f}")
        
        # Correlation
        correlation = torch.corrcoef(torch.stack([
            vampnet_output.flatten(),
            our_output.flatten()
        ]))[0, 1]
        print(f"   Correlation: {correlation:.4f}")
    
    print("="*80)


if __name__ == "__main__":
    debug_c2f_nan()