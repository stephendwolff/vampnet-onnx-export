#!/usr/bin/env python3
"""
Export VampNet transformer V11 with proper weight normalization handling.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
import os
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all the components from V10
from scripts.export_vampnet_transformer_v10_all_relative import (
    NewGELU, VampNetGatedGELU, FeedForwardGatedGELU, 
    VampNetEmbeddingLayer, VampNetTransformerV10
)

# Use V10 as base
VampNetTransformerV11 = VampNetTransformerV10


def transfer_weights_v11(vampnet_checkpoint, model, codec_path):
    """Transfer weights to V11 model with proper weight norm handling."""
    from vampnet.modules.transformer import VampNet
    from lac.model.lac import LAC as DAC
    from torch.nn.utils import remove_weight_norm
    
    # Set seed for reproducible weight normalization
    torch.manual_seed(42)
    
    vampnet = VampNet.load(vampnet_checkpoint, map_location='cpu')
    
    # Remove weight normalization from classifier to get stable weights
    try:
        remove_weight_norm(vampnet.classifier.layers[0])
        print("✓ Removed weight normalization from classifier")
    except:
        print("⚠ No weight normalization to remove")
    
    vampnet.eval()  # Set to eval mode
    codec = DAC.load(Path(codec_path)) if codec_path else None
    
    print("Transferring weights to V11 model...")
    
    # 1. Transfer projection weights
    model.embedding.out_proj.weight.data = vampnet.embedding.out_proj.weight.data
    model.embedding.out_proj.bias.data = vampnet.embedding.out_proj.bias.data
    print("✓ Transferred projection weights")
    
    # 2. Transfer transformer layers
    for i, (onnx_layer, vamp_layer) in enumerate(zip(model.layers, vampnet.transformer.layers)):
        # RMSNorm
        onnx_layer['norm_1'].weight.data = vamp_layer.norm_1.weight.data
        onnx_layer['norm_3'].weight.data = vamp_layer.norm_3.weight.data
        
        # Attention - all layers use MultiHeadRelativeAttention
        onnx_layer['self_attn'].w_qs.weight.data = vamp_layer.self_attn.w_qs.weight.data
        onnx_layer['self_attn'].w_ks.weight.data = vamp_layer.self_attn.w_ks.weight.data
        onnx_layer['self_attn'].w_vs.weight.data = vamp_layer.self_attn.w_vs.weight.data
        onnx_layer['self_attn'].fc.weight.data = vamp_layer.self_attn.fc.weight.data
        
        # Only layer 0 has relative attention bias
        if i == 0 and hasattr(vamp_layer.self_attn, 'relative_attention_bias'):
            onnx_layer['self_attn'].relative_attention_bias.weight.data = \
                vamp_layer.self_attn.relative_attention_bias.weight.data
        
        # FiLM
        if vamp_layer.film_3.input_dim > 0:
            onnx_layer['film_3'].gamma_weight.data = vamp_layer.film_3.gamma.weight.data.t()
            onnx_layer['film_3'].gamma_bias.data = vamp_layer.film_3.gamma.bias.data
            onnx_layer['film_3'].beta_weight.data = vamp_layer.film_3.beta.weight.data.t()
            onnx_layer['film_3'].beta_bias.data = vamp_layer.film_3.beta.bias.data
        else:
            # Identity FiLM
            onnx_layer['film_3'].gamma_weight.data.fill_(0)
            onnx_layer['film_3'].gamma_bias.data.fill_(1)
            onnx_layer['film_3'].beta_weight.data.fill_(0)
            onnx_layer['film_3'].beta_bias.data.fill_(0)
        
        # FFN
        onnx_layer['feed_forward'].w_1.weight.data = vamp_layer.feed_forward.w_1.weight.data
        onnx_layer['feed_forward'].w_2.weight.data = vamp_layer.feed_forward.w_2.weight.data
    
    # 3. Transfer final norm
    if hasattr(vampnet.transformer, 'norm'):
        model.final_norm.weight.data = vampnet.transformer.norm.weight.data
    
    # 4. Transfer output projections with stable weights
    classifier_weight = vampnet.classifier.layers[0].weight.data.squeeze(-1)
    classifier_bias = vampnet.classifier.layers[0].bias.data
    n_output_codebooks = model.n_codebooks - model.n_conditioning_codebooks
    
    for i in range(n_output_codebooks):
        vamp_start = i * vampnet.vocab_size
        vamp_end = (i + 1) * vampnet.vocab_size
        
        # Transfer weights
        model.output_projs[i].weight.data[:vampnet.vocab_size] = classifier_weight[vamp_start:vamp_end]
        model.output_projs[i].weight.data[vampnet.vocab_size] = 0
        
        # Transfer bias
        model.output_projs[i].bias.data[:vampnet.vocab_size] = classifier_bias[vamp_start:vamp_end]
        model.output_projs[i].bias.data[vampnet.vocab_size] = 0
    
    print("✓ Weight transfer complete!")


def test_v11_model():
    """Test V11 model with weight norm fix."""
    print("\n" + "="*80)
    print("TESTING V11 MODEL WITH WEIGHT NORM FIX")
    print("="*80)
    
    from vampnet.modules.transformer import VampNet
    from lac.model.lac import LAC as DAC
    from torch.nn.utils import remove_weight_norm
    from einops import rearrange
    
    checkpoint_path = "models/vampnet/coarse.pth"
    codec_path = "models/vampnet/codec.pth"
    
    # Load VampNet with weight norm removed
    torch.manual_seed(42)
    vampnet = VampNet.load(checkpoint_path, map_location='cpu')
    remove_weight_norm(vampnet.classifier.layers[0])
    vampnet.eval()
    codec = DAC.load(Path(codec_path))
    codec.eval()
    
    # Create V11 model
    model = VampNetTransformerV11(
        n_codebooks=4,
        n_conditioning_codebooks=0,
        vocab_size=1024,
        latent_dim=8,
        d_model=1280,
        n_heads=20,
        n_layers=20
    )
    
    # Transfer weights
    transfer_weights_v11(checkpoint_path, model, codec_path)
    model.eval()
    
    # Test
    torch.manual_seed(42)
    codes = torch.randint(0, 1024, (1, 4, 10))
    mask = torch.zeros((1, 4, 10), dtype=torch.bool)
    mask[:, :, 5:] = True
    masked_codes = codes.clone()
    masked_codes[mask] = 1024
    
    print(f"\nTest input codes: {masked_codes.shape}")
    
    with torch.no_grad():
        # Get latents
        vampnet_latents = vampnet.embedding.from_codes(masked_codes, codec)
        
        # Run both models
        vampnet_out = vampnet(vampnet_latents)
        v11_out = model(vampnet_latents)
        
        print(f"\nVampNet output: {vampnet_out.shape}")
        print(f"V11 output: {v11_out.shape}")
        
        # Correct comparison
        vampnet_reshaped = rearrange(vampnet_out, "b p (t c) -> b c t p", c=4, t=10)
        v11_truncated = v11_out[:, :, :, :1024]
        
        diff = (vampnet_reshaped - v11_truncated).abs()
        corr = np.corrcoef(vampnet_reshaped.flatten(), v11_truncated.flatten())[0,1]
        
        print(f"\nMean absolute difference: {diff.mean():.6f}")
        print(f"Max absolute difference: {diff.max():.6f}")
        print(f"Correlation: {corr:.4f}")
        
        if corr > 0.99:
            print("\n✅ SUCCESS! V11 model matches VampNet!")
            
            # Export to ONNX
            print("\nExporting to ONNX...")
            dummy_latents = torch.randn(1, 32, 100)
            
            torch.onnx.export(
                model,
                dummy_latents,
                "vampnet_transformer_v11.onnx",
                input_names=['latents'],
                output_names=['logits'],
                dynamic_axes={
                    'latents': {0: 'batch', 2: 'sequence'},
                    'logits': {0: 'batch', 2: 'sequence'}
                },
                opset_version=14,
                verbose=False
            )
            print("✓ Exported to vampnet_transformer_v11.onnx")
        else:
            print("\n❌ Still not matching...")
            
            # Debug info
            print(f"\nSample logits at [0, 0, 0, :5]:")
            print(f"VampNet: {vampnet_reshaped[0, 0, 0, :5]}")
            print(f"V11:     {v11_truncated[0, 0, 0, :5]}")


if __name__ == "__main__":
    test_v11_model()