#!/usr/bin/env python3
"""
Debug the EXACT flow through VampNet to understand what's happening.
"""

import torch
from pathlib import Path
import sys
from einops import rearrange

sys.path.append(str(Path(__file__).parent.parent))

# Load models
from vampnet.modules.transformer import VampNet
from lac.model.lac import LAC as DAC

vampnet = VampNet.load("models/vampnet/coarse.pth", map_location='cpu')
vampnet.eval()
codec = DAC.load(Path("models/vampnet/codec.pth"))
codec.eval()

# Test input
torch.manual_seed(42)
codes = torch.randint(0, 1024, (1, 4, 10))
mask = torch.zeros((1, 4, 10), dtype=torch.bool)
mask[:, :, 5:] = True

print("Detailed VampNet flow analysis...")
print(f"Input codes shape: {codes.shape}")

with torch.no_grad():
    # Step 1: Apply mask
    x = codes.clone()
    x[mask] = vampnet.mask_token
    print(f"\n1. After masking:")
    print(f"  Masked codes: {x}")
    
    # Step 2: Get latents via from_codes
    latents = vampnet.embedding.from_codes(x, codec)
    print(f"\n2. from_codes output:")
    print(f"  Shape: {latents.shape}")
    print(f"  Type: {latents.dtype}")
    print(f"  Device: {latents.device}")
    print(f"  Mean: {latents.mean():.4f}, Std: {latents.std():.4f}")
    
    # Check what from_codes actually does
    print(f"\n3. Checking from_codes internals:")
    print(f"  n_codebooks: {vampnet.n_codebooks}")
    print(f"  latent_dim: {vampnet.latent_dim}")
    print(f"  Expected latent shape: [1, {vampnet.n_codebooks * vampnet.latent_dim}, 10]")
    print(f"  Actual latent shape: {latents.shape}")
    
    # Step 3: Check VampNet's forward method directly
    print(f"\n4. Calling VampNet forward:")
    output = vampnet(latents)
    print(f"  Output shape: {output.shape}")
    print(f"  Output mean: {output.mean():.4f}, std: {output.std():.4f}")
    
    # Let's trace through VampNet's forward manually
    print(f"\n5. Manual trace through VampNet forward:")
    
    # VampNet's forward does:
    x = vampnet.embedding(latents)
    print(f"  After embedding: {x.shape}")
    
    x_mask = torch.ones_like(x, dtype=torch.bool)[:, :1, :].squeeze(1)
    print(f"  x_mask shape: {x_mask.shape}")
    
    x = rearrange(x, "b d n -> b n d")
    print(f"  After rearrange: {x.shape}")
    
    # Pass through transformer
    out = vampnet.transformer(x=x, x_mask=x_mask)
    print(f"  After transformer: {out.shape}")
    
    out = rearrange(out, "b n d -> b d n")
    print(f"  After rearrange back: {out.shape}")
    
    out = vampnet.classifier(out, None)
    print(f"  After classifier: {out.shape}")
    
    out = rearrange(out, "b (p c) t -> b p (t c)", c=vampnet.n_predict_codebooks)
    print(f"  Final output: {out.shape}")
    print(f"  n_predict_codebooks: {vampnet.n_predict_codebooks}")
    
    # Check if the outputs match
    manual_out = out
    auto_out = output
    
    print(f"\n6. Comparing manual vs auto forward:")
    diff = (manual_out - auto_out).abs().max()
    print(f"  Max difference: {diff:.8f}")
    
    if diff < 1e-6:
        print("  ✓ Manual trace matches VampNet forward!")
    else:
        print("  ✗ Manual trace differs from VampNet forward!")

# Now let's check what our V9 model expects
print("\n\n" + "="*60)
print("V9 Model Analysis:")

from scripts.export_vampnet_transformer_v9_proper_flow import VampNetTransformerV9, transfer_weights_v9

model = VampNetTransformerV9(
    n_codebooks=4,
    n_conditioning_codebooks=0,
    vocab_size=1024,
    latent_dim=8,
    d_model=1280,
    n_heads=20,
    n_layers=1  # Just 1 layer for quick test
)

transfer_weights_v9("models/vampnet/coarse.pth", model, "models/vampnet/codec.pth")
model.eval()

with torch.no_grad():
    # Get same latents (recreate masked codes)
    x_masked = codes.clone()
    x_masked[mask] = vampnet.mask_token
    latents = vampnet.embedding.from_codes(x_masked, codec)
    
    # Pass to V9
    v9_out = model(latents)
    print(f"V9 output shape: {v9_out.shape}")
    
    # Check first layer embeddings
    emb = model.embedding(latents)
    print(f"V9 embeddings shape: {emb.shape}")
    print(f"V9 embeddings mean: {emb.mean():.4f}, std: {emb.std():.4f}")
    
    # Compare with VampNet embeddings
    vampnet_emb = vampnet.embedding(latents)
    vampnet_emb_rearranged = rearrange(vampnet_emb, "b d n -> b n d")
    
    emb_diff = (emb - vampnet_emb_rearranged).abs()
    print(f"\nEmbedding difference:")
    print(f"  Mean: {emb_diff.mean():.6f}")
    print(f"  Max: {emb_diff.max():.6f}")
    
    if emb_diff.max() < 1e-5:
        print("  ✓ Embeddings match!")
    else:
        print("  ✗ Embeddings differ!")