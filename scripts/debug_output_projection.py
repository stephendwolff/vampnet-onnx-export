#!/usr/bin/env python3
"""
Debug the output projection difference.
"""

import torch
from pathlib import Path
import sys
from einops import rearrange

sys.path.append(str(Path(__file__).parent.parent))

from vampnet.modules.transformer import VampNet
from lac.model.lac import LAC as DAC
from scripts.export_vampnet_transformer_v9_proper_flow import VampNetTransformerV9, transfer_weights_v9

# Load models
vampnet = VampNet.load("models/vampnet/coarse.pth", map_location='cpu')
vampnet.eval()
codec = DAC.load(Path("models/vampnet/codec.pth"))
codec.eval()

# Create V9 model
model = VampNetTransformerV9(
    n_codebooks=4,
    n_conditioning_codebooks=0,
    vocab_size=1024,
    latent_dim=8,
    d_model=1280,
    n_heads=20,
    n_layers=20
)

transfer_weights_v9("models/vampnet/coarse.pth", model, "models/vampnet/codec.pth")
model.eval()

# Test input
torch.manual_seed(42)
codes = torch.randint(0, 1024, (1, 4, 10))
mask = torch.zeros((1, 4, 10), dtype=torch.bool)
mask[:, :, 5:] = True
masked_codes = codes.clone()
masked_codes[mask] = 1024

print("Debugging output projection...")

with torch.no_grad():
    # Get latents
    latents = vampnet.embedding.from_codes(masked_codes, codec)
    
    # Get embeddings
    vampnet_emb = vampnet.embedding(latents)
    vampnet_x = rearrange(vampnet_emb, "b d n -> b n d")
    
    # Pass through VampNet transformer
    vampnet_transformer_out = vampnet.transformer(x=vampnet_x, x_mask=torch.ones(1, 10, dtype=torch.bool))
    print(f"VampNet transformer output shape: {vampnet_transformer_out.shape}")
    
    # Rearrange back
    vampnet_out_rearranged = rearrange(vampnet_transformer_out, "b n d -> b d n")
    print(f"VampNet rearranged shape: {vampnet_out_rearranged.shape}")
    
    # Classifier
    vampnet_classifier_out = vampnet.classifier(vampnet_out_rearranged, None)
    print(f"VampNet classifier output shape: {vampnet_classifier_out.shape}")
    print(f"VampNet classifier output stats: mean={vampnet_classifier_out.mean():.4f}, std={vampnet_classifier_out.std():.4f}")
    
    # Final rearrange
    vampnet_final = rearrange(vampnet_classifier_out, "b (p c) t -> b p (t c)", c=vampnet.n_predict_codebooks)
    print(f"VampNet final shape: {vampnet_final.shape}")
    print(f"n_predict_codebooks: {vampnet.n_predict_codebooks}")
    
    # Now V9 model
    print("\n\nV9 Model:")
    v9_emb = model.embedding(latents)
    print(f"V9 embeddings shape: {v9_emb.shape}")
    
    # Pass through layers manually
    x = v9_emb
    position_bias = None
    
    # Just check after all transformer layers
    for i, layer in enumerate(model.layers):
        x_norm = layer['norm_1'](x)
        if i == 0:
            attn_out, position_bias = layer['self_attn'](x_norm, x_norm, x_norm, None, position_bias)
        else:
            attn_out, _ = layer['self_attn'](x_norm, x_norm, x_norm)
        x = x + attn_out
        x_norm = layer['norm_3'](x)
        x_norm = layer['film_3'](x_norm, x_norm)
        ffn_out = layer['feed_forward'](x_norm)
        x = x + ffn_out
    
    # Final norm
    x = model.final_norm(x)
    print(f"After transformer + final norm: {x.shape}")
    print(f"Stats: mean={x.mean():.4f}, std={x.std():.4f}")
    
    # Compare with VampNet after final norm
    vampnet_after_norm = vampnet.transformer.norm(vampnet_transformer_out) if hasattr(vampnet.transformer, 'norm') else vampnet_transformer_out
    norm_diff = (x - vampnet_after_norm).abs().max()
    print(f"\nDifference after final norm: {norm_diff:.6f}")
    
    # Output projections
    print("\n\nOutput projections:")
    
    # VampNet classifier
    print("VampNet classifier:")
    print(f"  Type: {type(vampnet.classifier)}")
    print(f"  Modules: {list(vampnet.classifier.named_children())}")
    
    # Check VampNet classifier weights
    vamp_classifier_weight = vampnet.classifier.layers[0].weight
    print(f"  Weight shape: {vamp_classifier_weight.shape}")
    print(f"  Weight stats: mean={vamp_classifier_weight.mean():.4f}, std={vamp_classifier_weight.std():.4f}")
    
    # Our output projections
    print("\nV9 output projections:")
    for i, proj in enumerate(model.output_projs):
        print(f"  Proj {i} weight shape: {proj.weight.shape}")
        print(f"  Proj {i} weight stats: mean={proj.weight.mean():.4f}, std={proj.weight.std():.4f}")
    
    # Apply our projections
    v9_logits = []
    for i in range(4):
        cb_logits = model.output_projs[i](x)
        v9_logits.append(cb_logits)
        print(f"  Codebook {i} logits shape: {cb_logits.shape}")
    
    # Stack
    v9_logits_stacked = torch.stack(v9_logits, dim=1)
    print(f"\nV9 stacked logits shape: {v9_logits_stacked.shape}")
    
    # Compare specific logit values
    print("\n\nComparing specific logit values at position [0, 0, 0]:")
    
    # VampNet logits at position 0
    vamp_pos0 = vampnet_classifier_out[0, :4, 0]  # First 4 values (codebook 0)
    print(f"VampNet: {vamp_pos0}")
    
    # V9 logits at position 0
    v9_pos0 = v9_logits_stacked[0, 0, 0, :4]  # First 4 values
    print(f"V9: {v9_pos0}")
    
    print(f"Difference: {(vamp_pos0 - v9_pos0).abs()}")