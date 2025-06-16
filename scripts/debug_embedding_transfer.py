#!/usr/bin/env python3
"""
Debug the embedding weight transfer issue.
"""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

# Load models
print("Loading models...")
from vampnet.modules.transformer import VampNet
from lac.model.lac import LAC as DAC

vampnet = VampNet.load("../models/vampnet/coarse.pth", map_location='cpu')
vampnet.eval()
codec = DAC.load(Path("../models/vampnet/codec.pth"))
codec.eval()

# Test tokens
torch.manual_seed(42)
test_tokens = torch.tensor([[[100, 200, 300, 1024],
                             [150, 250, 350, 1024],
                             [50, 150, 250, 1024],
                             [80, 180, 280, 1024]]])  # 4 codebooks

print("\n1. Testing VampNet embedding process...")
with torch.no_grad():
    # Step by step through VampNet embedding
    codes = test_tokens
    
    print(f"Input codes: {codes}")
    
    # VampNet's from_codes method
    n_codebooks = codes.shape[1]
    latent = []
    for i in range(n_codebooks):
        c = codes[:, i, :]
        print(f"\nCodebook {i}:")
        print(f"  Code indices: {c}")
        
        # Get lookup table
        lookup_table = codec.quantizer.quantizers[i].codebook.weight
        print(f"  Lookup table shape: {lookup_table.shape}")
        
        # Add special tokens
        if hasattr(vampnet.embedding, "special"):
            special_lookup = torch.cat(
                [vampnet.embedding.special[tkn][i : i + 1] for tkn in vampnet.embedding.special], dim=0
            )
            print(f"  Special tokens shape: {special_lookup.shape}")
            print(f"  MASK token embedding: {vampnet.embedding.special['MASK'][i]}")
            lookup_table = torch.cat([lookup_table, special_lookup], dim=0)
            print(f"  Combined lookup table shape: {lookup_table.shape}")
        
        # Embed
        l = torch.nn.functional.embedding(c, lookup_table).transpose(1, 2)
        print(f"  Embedded shape: {l.shape}")
        print(f"  Embedded mean: {l.mean():.4f}, std: {l.std():.4f}")
        latent.append(l)
    
    latent = torch.cat(latent, dim=1)
    print(f"\nConcatenated latents shape: {latent.shape}")
    print(f"Concatenated latents mean: {latent.mean():.4f}, std: {latent.std():.4f}")
    
    # Project
    embeddings = vampnet.embedding.out_proj(latent)
    print(f"\nProjected embeddings shape: {embeddings.shape}")
    print(f"Projected embeddings mean: {embeddings.mean():.4f}, std: {embeddings.std():.4f}")

print("\n2. Checking our embedding implementation...")
from scripts.codebook_embedding_correct_v2 import CodebookEmbeddingCorrectV2

our_embedding = CodebookEmbeddingCorrectV2(
    n_codebooks=1,
    vocab_size=1024,
    latent_dim=8,
    d_model=1280
)

# Transfer weights for single codebook
ckpt = torch.load("../models/vampnet/coarse.pth", map_location='cpu')
if 'codec' in ckpt:
    codec_state = ckpt['codec']
    key = f'quantizer.quantizers.0.codebook.weight'
    if key in codec_state:
        codec_emb = codec_state[key]
        our_embedding.embeddings[0].weight.data[:1024] = codec_emb
        print(f"Transferred codec embeddings shape: {codec_emb.shape}")

# Transfer special token
if hasattr(vampnet.embedding, 'special') and 'MASK' in vampnet.embedding.special:
    mask_emb = vampnet.embedding.special['MASK'][0]
    our_embedding.embeddings[0].weight.data[1024] = mask_emb
    print(f"Transferred MASK embedding: {mask_emb}")

# Test single codebook
single_code = test_tokens[:, 0:1, :]  # Just first codebook
print(f"\nTesting single codebook: {single_code}")

with torch.no_grad():
    our_latent = our_embedding.embeddings[0](single_code[0, 0])
    print(f"Our embedding output shape: {our_latent.shape}")
    print(f"Our embedding mean: {our_latent.mean():.4f}, std: {our_latent.std():.4f}")
    
    # Compare with VampNet's first codebook
    vampnet_cb0 = latent[:, :8, :]  # First 8 channels
    print(f"\nVampNet codebook 0 latent mean: {vampnet_cb0.mean():.4f}, std: {vampnet_cb0.std():.4f}")
    
    # Direct comparison
    our_latent_reshaped = our_latent.transpose(0, 1).unsqueeze(0)
    diff = (vampnet_cb0 - our_latent_reshaped).abs()
    print(f"Difference - mean: {diff.mean():.6f}, max: {diff.max():.6f}")

print("\n3. Testing projection layer...")
# Create projection with correct dimensions
our_proj = torch.nn.Conv1d(8, 1280, 1)  # Just for single codebook
our_proj.weight.data = vampnet.embedding.out_proj.weight.data[:, :8, :]
our_proj.bias.data = vampnet.embedding.out_proj.bias.data

with torch.no_grad():
    our_projected = our_proj(our_latent_reshaped)
    vampnet_projected = vampnet.embedding.out_proj(vampnet_cb0)
    
    print(f"Our projection shape: {our_projected.shape}")
    print(f"VampNet projection shape: {vampnet_projected.shape}")
    
    proj_diff = (our_projected - vampnet_projected).abs()
    print(f"Projection difference - mean: {proj_diff.mean():.6f}, max: {proj_diff.max():.6f}")