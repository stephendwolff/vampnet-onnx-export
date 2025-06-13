"""
Check vocabulary sizes in VampNet models and codecs.

This script investigates the actual vocabulary sizes used in VampNet
to understand the discrepancy mentioned in the technical notes.
"""

import torch
import numpy as np
from pathlib import Path
import audiotools as at
from vampnet import interface as vampnet


def check_vampnet_vocab_sizes():
    """Check vocabulary sizes in VampNet models."""
    
    print("=== Checking VampNet Vocabulary Sizes ===\n")
    
    # Load VampNet interface
    print("Loading VampNet models...")
    interface = vampnet.Interface(
        coarse_ckpt="models/vampnet/coarse.pth",
        coarse2fine_ckpt="models/vampnet/c2f.pth",
        codec_ckpt="models/vampnet/codec.pth",
        device="cpu"
    )
    
    # Check codec vocabulary sizes
    print("\n=== Codec Information ===")
    codec = interface.codec
    print(f"Codec type: {type(codec)}")
    print(f"Number of quantizers: {len(codec.quantizer.quantizers)}")
    
    for i, quantizer in enumerate(codec.quantizer.quantizers):
        codebook_size = quantizer.codebook.weight.shape[0]
        latent_dim = quantizer.codebook.weight.shape[1]
        print(f"Quantizer {i}: codebook_size={codebook_size}, latent_dim={latent_dim}")
    
    # Check coarse model
    print("\n=== Coarse Model ===")
    coarse = interface.coarse
    
    print(f"Model vocab_size attribute: {coarse.vocab_size}")
    print(f"Model n_codebooks: {coarse.n_codebooks}")
    print(f"Model n_conditioning_codebooks: {coarse.n_conditioning_codebooks}")
    print(f"Model n_predict_codebooks: {coarse.n_predict_codebooks}")
    
    # Check embedding details
    print("\nEmbedding details:")
    if hasattr(coarse.embedding, 'special_idxs'):
        print(f"Special tokens: {coarse.embedding.special_idxs}")
    
    # Check classifier output dimensions
    classifier = coarse.classifier.layers[0]  # WNConv1d
    if hasattr(classifier, 'weight_v'):
        out_channels = classifier.weight_v.shape[0]
        in_channels = classifier.weight_v.shape[1]
        print(f"\nClassifier dimensions:")
        print(f"  Input channels: {in_channels}")
        print(f"  Output channels: {out_channels}")
        print(f"  Output per codebook: {out_channels // coarse.n_predict_codebooks}")
    
    # Check C2F model
    print("\n=== C2F Model ===")
    c2f = interface.c2f
    
    print(f"Model vocab_size attribute: {c2f.vocab_size}")
    print(f"Model n_codebooks: {c2f.n_codebooks}")
    print(f"Model n_conditioning_codebooks: {c2f.n_conditioning_codebooks}")
    print(f"Model n_predict_codebooks: {c2f.n_predict_codebooks}")
    
    # Check output dimensions
    classifier = c2f.classifier.layers[0]
    if hasattr(classifier, 'weight_v'):
        out_channels = classifier.weight_v.shape[0]
        print(f"\nClassifier output channels: {out_channels}")
        print(f"Output per codebook: {out_channels // c2f.n_predict_codebooks}")
    
    # Test token generation
    print("\n=== Testing Token Generation ===")
    
    # Generate some dummy audio with correct sample rate
    signal = at.AudioSignal(torch.randn(1, 1, codec.sample_rate), codec.sample_rate)
    
    # Encode with codec
    with torch.no_grad():
        encoded = codec.encode(signal.samples, signal.sample_rate)
        if isinstance(encoded, tuple):
            tokens = encoded[0]
        else:
            tokens = encoded
    
    print(f"\nEncoded token shape: {tokens.shape}")
    print(f"Token value range: [{tokens.min().item()}, {tokens.max().item()}]")
    print(f"Unique token values: {len(torch.unique(tokens))}")
    
    # Sample some tokens
    sample_tokens = tokens[0, :4, :10]  # First 4 codebooks, first 10 timesteps
    print(f"\nSample tokens:\n{sample_tokens}")
    
    return interface


def analyze_vocab_discrepancy():
    """Analyze the 1024 vs 4096 vocabulary discrepancy."""
    
    print("\n\n=== Analyzing Vocabulary Size Discrepancy ===\n")
    
    print("The technical notes mention:")
    print("- VampNet: 4096 vocabulary size")
    print("- ONNX: 1024 + 1 (mask token)")
    
    print("\nPossible explanations:")
    print("1. VampNet might use different vocab sizes for different codebooks")
    print("2. The 4096 might refer to total output dimensions (vocab_size * n_codebooks)")
    print("3. Different models (coarse vs c2f) might use different sizes")
    print("4. The codec might support larger vocabulary than what's used")
    
    # Check if 4096 = 1024 * 4
    print(f"\n1024 * 4 codebooks = {1024 * 4}")
    print("This matches the 4096 mentioned in the notes!")
    
    print("\nConclusion:")
    print("The '4096' likely refers to the total output dimensions of the classifier")
    print("(1024 vocabulary * 4 codebooks), not the vocabulary size per codebook.")
    print("Each codebook still uses vocabulary size 1024.")


def check_model_outputs():
    """Check the actual output dimensions of models."""
    
    print("\n\n=== Checking Model Output Dimensions ===\n")
    
    interface = vampnet.Interface(
        coarse_ckpt="models/vampnet/coarse.pth",
        coarse2fine_ckpt="models/vampnet/c2f.pth",
        codec_ckpt="models/vampnet/codec.pth",
        device="cpu"
    )
    
    # Test coarse model
    print("Testing coarse model forward pass...")
    coarse = interface.coarse
    coarse.eval()
    
    # Create dummy input
    batch_size = 1
    seq_len = 10
    n_codebooks = coarse.n_codebooks
    
    # Create latent input (after embedding)
    latent_dim = coarse.latent_dim
    embedding_dim = coarse.embedding_dim
    
    dummy_latent = torch.randn(batch_size, n_codebooks * latent_dim, seq_len)
    
    with torch.no_grad():
        output = coarse(dummy_latent)
    
    print(f"\nCoarse model:")
    print(f"  Input shape: {dummy_latent.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output interpretation: [batch, n_predict_codebooks, seq * vocab_size]")
    
    # Reshape to see per-codebook vocabulary
    output_per_cb = output.shape[-1] // seq_len
    print(f"  Vocabulary per codebook: {output_per_cb}")
    
    # Test embedding dimensions
    print("\nChecking embedding dimensions...")
    
    # Check embedding module structure
    print(f"  Embedding latent_dim: {coarse.embedding.latent_dim}")
    print(f"  Embedding emb_dim: {coarse.embedding.emb_dim}")
    print(f"  Embedding n_codebooks: {coarse.embedding.n_codebooks}")
    
    # Check output projection
    if hasattr(coarse.embedding, 'out_proj'):
        out_proj = coarse.embedding.out_proj
        print(f"  Output projection: {out_proj.in_channels} -> {out_proj.out_channels}")


if __name__ == "__main__":
    # Check vocabulary sizes
    interface = check_vampnet_vocab_sizes()
    
    # Analyze discrepancy
    analyze_vocab_discrepancy()
    
    # Check outputs
    check_model_outputs()