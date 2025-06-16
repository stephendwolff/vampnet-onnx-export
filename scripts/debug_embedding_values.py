#!/usr/bin/env python3
"""
Debug the actual embedding values to understand the discrepancy.
"""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

# Load models
print("Loading VampNet...")
from vampnet.modules.transformer import VampNet
from lac.model.lac import LAC as DAC

vampnet = VampNet.load("../models/vampnet/coarse.pth", map_location='cpu')
codec = DAC.load(Path("../models/vampnet/codec.pth"))

# Simple test - just get embedding for token 100 in codebook 0
print("\n1. Direct codec embedding lookup:")
codec_weight = codec.quantizer.quantizers[0].codebook.weight
print(f"Codec weight shape: {codec_weight.shape}")
print(f"Codec weight stats - mean: {codec_weight.mean():.4f}, std: {codec_weight.std():.4f}")
print(f"Embedding for token 100: {codec_weight[100]}")

# Test VampNet's embedding process
print("\n2. VampNet embedding process:")
test_code = torch.tensor([[[100]]])  # Single token, single codebook

with torch.no_grad():
    # Manual lookup
    lookup_table = codec.quantizer.quantizers[0].codebook.weight
    special_lookup = vampnet.embedding.special['MASK'][0:1]
    full_lookup = torch.cat([lookup_table, special_lookup], dim=0)
    
    embedded = torch.nn.functional.embedding(test_code[0, 0], full_lookup)
    print(f"Embedded value: {embedded}")
    print(f"Embedded shape: {embedded.shape}")

# Now check our embedding
print("\n3. Our embedding implementation:")
from scripts.codebook_embedding_correct_v2 import CodebookEmbeddingCorrectV2

our_embedding = CodebookEmbeddingCorrectV2(
    n_codebooks=1,
    vocab_size=1024,
    latent_dim=8,
    d_model=1280
)

# Check initial values
print(f"Our embedding initial weight stats - mean: {our_embedding.embeddings[0].weight.mean():.4f}, std: {our_embedding.embeddings[0].weight.std():.4f}")

# Transfer weights
ckpt = torch.load("../models/vampnet/coarse.pth", map_location='cpu')
if 'codec' in ckpt:
    codec_state = ckpt['codec']
    key = f'quantizer.quantizers.0.codebook.weight'
    if key in codec_state:
        codec_emb = codec_state[key]
        print(f"\nCodec embedding from checkpoint shape: {codec_emb.shape}")
        print(f"Codec embedding from checkpoint stats - mean: {codec_emb.mean():.4f}, std: {codec_emb.std():.4f}")
        our_embedding.embeddings[0].weight.data[:1024] = codec_emb
        print("Transferred codec embeddings")

# Check after transfer
print(f"\nOur embedding after transfer stats - mean: {our_embedding.embeddings[0].weight[:1024].mean():.4f}, std: {our_embedding.embeddings[0].weight[:1024].std():.4f}")
print(f"Our embedding for token 100: {our_embedding.embeddings[0].weight[100]}")

# Test embedding
with torch.no_grad():
    our_embedded = our_embedding.embeddings[0](test_code[0, 0])
    print(f"\nOur embedded value: {our_embedded}")
    
# Compare
print("\n4. Comparison:")
print(f"VampNet embedding: {embedded[0]}")
print(f"Our embedding: {our_embedded[0]}")
diff = (embedded[0] - our_embedded[0]).abs()
print(f"Difference: {diff}")
print(f"Max difference: {diff.max():.6f}")

# Check if the issue is in how VampNet uses the codec
print("\n5. Checking VampNet's from_codes method:")
with torch.no_grad():
    # Full 4-codebook test
    test_tokens_4cb = torch.tensor([[[100], [150], [50], [80]]])
    latents = vampnet.embedding.from_codes(test_tokens_4cb, codec)
    print(f"VampNet from_codes output shape: {latents.shape}")
    print(f"VampNet from_codes stats - mean: {latents.mean():.4f}, std: {latents.std():.4f}")
    
    # Check individual codebook outputs
    for i in range(4):
        cb_latent = latents[:, i*8:(i+1)*8, :]
        print(f"  Codebook {i} latent stats - mean: {cb_latent.mean():.4f}, std: {cb_latent.std():.4f}")