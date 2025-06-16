#!/usr/bin/env python3
"""
Debug C2F NaN issues more carefully - check each layer.
"""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from scripts.export_c2f_transformer_v14_fixed import C2FTransformerV14, transfer_weights_c2f_v14
from vampnet.modules.transformer import VampNet
from lac.model.lac import LAC as DAC
from torch.nn.utils import remove_weight_norm
from einops import rearrange


def debug_layer_by_layer():
    """Debug C2F layer by layer to find where NaN appears."""
    print("="*80)
    print("DEBUGGING C2F LAYER BY LAYER")
    print("="*80)
    
    # Load models
    print("\n1. Loading models...")
    
    # Our model
    model = C2FTransformerV14()
    transfer_weights_c2f_v14("models/vampnet/c2f.pth", model, "models/vampnet/codec.pth")
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
    
    # Use a specific seed for reproducibility
    torch.manual_seed(42)
    test_codes = torch.randint(0, 1024, (batch_size, 14, seq_len))
    
    # Convert to latents
    z = codec.quantizer.from_codes(test_codes)
    test_latents = z[1]  # Get latents
    print(f"   Input latents shape: {test_latents.shape}")
    
    # Check FFN weights
    print("\n3. Checking FFN weight dimensions...")
    for i in range(min(3, model.n_layers)):
        ffn = model.transformer_layers[i]['ffn']
        vampnet_ffn = vampnet_c2f.transformer.layers[i].feed_forward
        
        print(f"\n   Layer {i} FFN:")
        print(f"   Our w_1: {ffn.w_1.weight.shape}")
        print(f"   VampNet w_1: {vampnet_ffn.w_1.weight.shape}")
        print(f"   Our w_2: {ffn.w_2.weight.shape}")
        print(f"   VampNet w_2: {vampnet_ffn.w_2.weight.shape}")
        
        # Check if weights match
        w1_match = torch.allclose(ffn.w_1.weight, vampnet_ffn.w_1.weight)
        w2_match = torch.allclose(ffn.w_2.weight, vampnet_ffn.w_2.weight)
        print(f"   w_1 weights match: {w1_match}")
        print(f"   w_2 weights match: {w2_match}")
    
    # Test layer by layer
    print("\n4. Testing layer by layer...")
    with torch.no_grad():
        # Embedding
        x = model.embedding(test_latents)
        x_vampnet = vampnet_c2f.embedding(test_latents)
        
        print(f"\nAfter embedding:")
        print(f"   Our shape: {x.shape}, has NaN: {torch.isnan(x).any()}")
        print(f"   VampNet shape: {x_vampnet.shape}, has NaN: {torch.isnan(x_vampnet).any()}")
        print(f"   Embedding match: {torch.allclose(x, x_vampnet, atol=1e-5)}")
        
        # Rearrange
        x = rearrange(x, "b d n -> b n d")
        x_vampnet = rearrange(x_vampnet, "b d n -> b n d")
        
        # Test first few layers
        for i in range(min(5, model.n_layers)):
            print(f"\n--- Layer {i} ---")
            
            # Our model
            layer = model.transformer_layers[i]
            residual = x
            x_norm = layer['norm'](x)
            
            # VampNet
            vampnet_layer = vampnet_c2f.transformer.layers[i]
            residual_vampnet = x_vampnet
            x_norm_vampnet = vampnet_layer.norm_1(x_vampnet)
            
            print(f"After norm: Our NaN: {torch.isnan(x_norm).any()}, VampNet NaN: {torch.isnan(x_norm_vampnet).any()}")
            
            # Attention
            x_attn = layer['self_attn'](x_norm)
            # VampNet's attention expects (q, k, v) separately and returns (output, attention_weights)
            x_attn_vampnet_tuple = vampnet_layer.self_attn(x_norm_vampnet, x_norm_vampnet, x_norm_vampnet)
            x_attn_vampnet = x_attn_vampnet_tuple[0] if isinstance(x_attn_vampnet_tuple, tuple) else x_attn_vampnet_tuple
            
            print(f"After attn: Our NaN: {torch.isnan(x_attn).any()}, VampNet NaN: {torch.isnan(x_attn_vampnet).any()}")
            
            # Add residual
            x = residual + x_attn
            x_vampnet = residual_vampnet + x_attn_vampnet
            
            # FiLM
            x = layer['film'](x)
            x_vampnet = vampnet_layer.film_1(x_vampnet, None)  # C2F doesn't use conditioning
            
            # FFN
            residual = x
            residual_vampnet = x_vampnet
            
            # Test FFN step by step
            print(f"\nFFN debugging:")
            
            # w_1
            x_ffn = layer['ffn'].w_1(x)
            x_ffn_vampnet = vampnet_layer.feed_forward.w_1(x_vampnet)
            print(f"  After w_1: Our shape: {x_ffn.shape}, VampNet shape: {x_ffn_vampnet.shape}")
            print(f"  After w_1: Our NaN: {torch.isnan(x_ffn).any()}, VampNet NaN: {torch.isnan(x_ffn_vampnet).any()}")
            
            # Check values
            if not torch.isnan(x_ffn).any():
                print(f"  Our w_1 output range: [{x_ffn.min():.4f}, {x_ffn.max():.4f}]")
            if not torch.isnan(x_ffn_vampnet).any():
                print(f"  VampNet w_1 output range: [{x_ffn_vampnet.min():.4f}, {x_ffn_vampnet.max():.4f}]")
            
            # Activation
            x_act = layer['ffn'].activation(x_ffn)
            x_act_vampnet = vampnet_layer.feed_forward.act(x_ffn_vampnet)
            print(f"  After activation: Our shape: {x_act.shape}, VampNet shape: {x_act_vampnet.shape}")
            print(f"  After activation: Our NaN: {torch.isnan(x_act).any()}, VampNet NaN: {torch.isnan(x_act_vampnet).any()}")
            
            # If NaN appears, stop
            if torch.isnan(x_act).any():
                print("\n  !!! NaN detected in our model after activation !!!")
                print(f"  Checking activation input more carefully...")
                print(f"  x_ffn min: {x_ffn.min()}, max: {x_ffn.max()}")
                
                # Check if it's the split that causes issues
                x1, x2 = x_ffn.chunk(2, dim=-1)
                print(f"  After split - x1: {x1.shape}, x2: {x2.shape}")
                print(f"  x1 has NaN: {torch.isnan(x1).any()}, x2 has NaN: {torch.isnan(x2).any()}")
                
                # Test NewGELU on x2
                gelu = model.transformer_layers[i]['ffn'].activation.activation
                gelu_out = gelu(x2)
                print(f"  GELU(x2) has NaN: {torch.isnan(gelu_out).any()}")
                
                break
            
            # w_2
            x_ffn = layer['ffn'].w_2(x_act)
            x_ffn_vampnet = vampnet_layer.feed_forward.w_2(x_act_vampnet)
            print(f"  After w_2: Our NaN: {torch.isnan(x_ffn).any()}, VampNet NaN: {torch.isnan(x_ffn_vampnet).any()}")
            
            # Add residual
            x = residual + x_ffn
            x_vampnet = residual_vampnet + x_ffn_vampnet
            
            print(f"After FFN: Our NaN: {torch.isnan(x).any()}, VampNet NaN: {torch.isnan(x_vampnet).any()}")
            
            if torch.isnan(x).any():
                print(f"\n!!! NaN detected in layer {i} !!!")
                break
    
    print("="*80)


if __name__ == "__main__":
    debug_layer_by_layer()