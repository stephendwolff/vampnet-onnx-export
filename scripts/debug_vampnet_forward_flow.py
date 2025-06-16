#!/usr/bin/env python3
"""
Debug the exact forward flow in VampNet.
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

print("Testing VampNet forward flow...")
print(f"Input codes shape: {codes.shape}")
print(f"Mask shape: {mask.shape}")

# Step through forward pass
with torch.no_grad():
    # 1. Apply mask
    x = codes.clone()
    x[mask] = vampnet.mask_token
    print(f"\n1. After masking:")
    print(f"  Masked codes: {x}")
    print(f"  Mask token: {vampnet.mask_token}")
    
    # 2. Get latents from codes
    latents = vampnet.embedding.from_codes(x, codec)
    print(f"\n2. After from_codes:")
    print(f"  Latents shape: {latents.shape}")
    print(f"  Latents mean: {latents.mean():.4f}, Std: {latents.std():.4f}")
    
    # 3. Get embeddings from latents
    x = vampnet.embedding(latents)
    print(f"\n3. After embedding:")
    print(f"  Shape: {x.shape}")
    print(f"  Mean: {x.mean():.4f}, Std: {x.std():.4f}")
    
    # Create mask for transformer
    x_mask = torch.ones_like(x, dtype=torch.bool)[:, :1, :].squeeze(1)
    print(f"  x_mask shape: {x_mask.shape}")
    
    # 4. Rearrange
    x = rearrange(x, "b d n -> b n d")
    print(f"\n4. After rearrange:")
    print(f"  Shape: {x.shape}")
    
    # 5. Pass through transformer
    out = vampnet.transformer(x=x, x_mask=x_mask)
    print(f"\n5. After transformer:")
    print(f"  Shape: {out.shape}")
    print(f"  Mean: {out.mean():.4f}, Std: {out.std():.4f}")
    
    # 6. Rearrange back
    out = rearrange(out, "b n d -> b d n")
    print(f"\n6. After rearrange back:")
    print(f"  Shape: {out.shape}")
    
    # 7. Classifier
    out = vampnet.classifier(out, None)
    print(f"\n7. After classifier:")
    print(f"  Shape: {out.shape}")
    print(f"  Mean: {out.mean():.4f}, Std: {out.std():.4f}")
    
    # 8. Final rearrange
    out = rearrange(out, "b (p c) t -> b p (t c)", c=vampnet.n_predict_codebooks)
    print(f"\n8. Final output:")
    print(f"  Shape: {out.shape}")
    print(f"  n_predict_codebooks: {vampnet.n_predict_codebooks}")
    
    # Check a specific output value
    print(f"\n8. Sample output values:")
    print(f"  Position [0, 0, 0]: {out[0, 0, 0]:.4f}")
    print(f"  Position [0, 100, 0]: {out[0, 100, 0]:.4f}")
    print(f"  Position [0, 200, 0]: {out[0, 200, 0]:.4f}")

# Now check what our ONNX model expects
print("\n\nONNX Model Expected Flow:")
print("1. Input: codes [B, n_codebooks, seq_len]")
print("2. Input: mask [B, n_codebooks, seq_len]")
print("3. Apply mask internally")
print("4. Get embeddings")
print("5. Pass through transformer")  
print("6. Output: logits [B, n_codebooks, seq_len, vocab_size+1]")

# The key difference might be in the embedding function
print("\n\nChecking VampNet embedding function...")
print(f"VampNet embedding type: {type(vampnet.embedding)}")
print(f"Has from_codes: {hasattr(vampnet.embedding, 'from_codes')}")

# Check if VampNet's forward expects latents or codes
print("\n\nChecking VampNet forward signature...")
import inspect
sig = inspect.signature(vampnet.forward)
print(f"VampNet forward parameters: {list(sig.parameters.keys())}")