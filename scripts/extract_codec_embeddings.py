"""
Extract codec embeddings from VampNet model for ONNX initialization.

This script extracts the actual codec quantizer weights that VampNet uses
as embeddings, along with special token embeddings, and saves them in a format
that can be used to initialize ONNX models.
"""

import torch
import numpy as np
from pathlib import Path
import argparse
import audiotools as at
from vampnet import interface as vampnet


def extract_codec_embeddings(
    model_path: str = "models/vampnet/coarse.pth",
    save_path: str = "models/codec_embeddings.pth",
    n_codebooks: int = 4,
    vocab_size: int = 1024,
):
    """Extract codec embeddings from VampNet model."""
    
    print("=== Extracting Codec Embeddings from VampNet ===")
    
    # Load VampNet interface
    print(f"\nLoading VampNet model from {model_path}")
    interface = vampnet.Interface(
        coarse_ckpt=model_path,
        coarse2fine_ckpt=None,  # We only need coarse for embeddings
        codec_ckpt="models/vampnet/codec.pth",
        device="cpu"
    )
    
    # Get the model and codec
    model = interface.coarse
    codec = interface.codec
    
    print(f"Model type: {type(model)}")
    print(f"Codec type: {type(codec)}")
    
    # Extract embeddings for each codebook
    embeddings = {}
    
    print(f"\nExtracting embeddings for {n_codebooks} codebooks")
    for i in range(n_codebooks):
        # Get the quantizer codebook weights
        quantizer = codec.quantizer.quantizers[i]
        codebook_weights = quantizer.codebook.weight.data.clone()
        
        print(f"\nCodebook {i}:")
        print(f"  Codebook shape: {codebook_weights.shape}")
        print(f"  Expected: ({vocab_size}, latent_dim)")
        
        # Get special token embeddings if they exist
        if hasattr(model.embedding, "special"):
            # Extract MASK token embedding for this codebook
            mask_embedding = model.embedding.special["MASK"][i].data.clone()
            print(f"  MASK embedding shape: {mask_embedding.shape}")
            
            # Concatenate codebook weights with special tokens
            # This matches what VampNet does in from_codes()
            full_embeddings = torch.cat([codebook_weights, mask_embedding.unsqueeze(0)], dim=0)
        else:
            full_embeddings = codebook_weights
            
        embeddings[f"codebook_{i}"] = full_embeddings
        print(f"  Final embedding shape: {full_embeddings.shape}")
    
    # Also save the output projection weights
    if hasattr(model.embedding, "out_proj"):
        print("\nExtracting output projection weights")
        out_proj_weight = model.embedding.out_proj.weight.data.clone()
        out_proj_bias = model.embedding.out_proj.bias.data.clone()
        
        embeddings["out_proj_weight"] = out_proj_weight
        embeddings["out_proj_bias"] = out_proj_bias
        
        print(f"  Projection weight shape: {out_proj_weight.shape}")
        print(f"  Projection bias shape: {out_proj_bias.shape}")
    
    # Save embeddings
    print(f"\nSaving embeddings to {save_path}")
    torch.save(embeddings, save_path)
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Extracted embeddings for {n_codebooks} codebooks")
    print(f"Vocabulary size: {vocab_size} + special tokens")
    print(f"Saved to: {save_path}")
    
    # Verify by loading
    loaded = torch.load(save_path, map_location="cpu")
    print("\nVerification - loaded keys:")
    for k, v in loaded.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")
    
    return embeddings


def create_onnx_embeddings_from_codec(
    codec_embeddings_path: str = "models/codec_embeddings.pth",
    save_path: str = "models/onnx_embeddings.pth",
    d_model: int = 1280,
    n_codebooks: int = 4,
):
    """Convert codec embeddings to ONNX-compatible format."""
    
    print("\n=== Creating ONNX Embeddings from Codec ===")
    
    # Load codec embeddings
    codec_embeddings = torch.load(codec_embeddings_path, map_location="cpu")
    
    # Create ONNX-compatible embeddings
    # The ONNX model uses VerySimpleCodebookEmbedding which sums embeddings
    onnx_embeddings = {}
    
    for i in range(n_codebooks):
        key = f"codebook_{i}"
        if key in codec_embeddings:
            # Get the codec embeddings
            codec_emb = codec_embeddings[key]
            latent_dim = codec_emb.shape[-1]
            vocab_size = codec_emb.shape[0]
            
            print(f"\nProcessing codebook {i}:")
            print(f"  Input shape: {codec_emb.shape}")
            
            # Apply the output projection to match d_model
            if "out_proj_weight" in codec_embeddings:
                # The projection in VampNet goes from n_codebooks*latent_dim to d_model
                # We need to simulate this for each codebook
                proj_weight = codec_embeddings["out_proj_weight"]
                proj_bias = codec_embeddings["out_proj_bias"]
                
                # Extract the relevant portion of the projection for this codebook
                start_idx = i * latent_dim
                end_idx = (i + 1) * latent_dim
                cb_proj_weight = proj_weight[:, start_idx:end_idx]  # [d_model, latent_dim]
                
                # Project the embeddings
                projected_emb = torch.matmul(codec_emb, cb_proj_weight.t())
                if proj_bias is not None:
                    # Add bias (distributed across codebooks)
                    projected_emb = projected_emb + proj_bias / n_codebooks
                
                onnx_embeddings[f"embedding.embeddings.{i}.weight"] = projected_emb
                print(f"  Output shape: {projected_emb.shape}")
            else:
                # If no projection, we need to pad to d_model
                if latent_dim < d_model:
                    padding = torch.zeros(vocab_size, d_model - latent_dim)
                    padded_emb = torch.cat([codec_emb, padding], dim=-1)
                    onnx_embeddings[f"embedding.embeddings.{i}.weight"] = padded_emb
                    print(f"  Padded to shape: {padded_emb.shape}")
                else:
                    onnx_embeddings[f"embedding.embeddings.{i}.weight"] = codec_emb[:, :d_model]
                    print(f"  Truncated to shape: {codec_emb[:, :d_model].shape}")
    
    # Save ONNX embeddings
    print(f"\nSaving ONNX embeddings to {save_path}")
    torch.save(onnx_embeddings, save_path)
    
    return onnx_embeddings


def test_embedding_extraction():
    """Test the embedding extraction process."""
    
    print("\n=== Testing Embedding Extraction ===")
    
    # First extract codec embeddings
    codec_emb = extract_codec_embeddings()
    
    # Then convert to ONNX format
    onnx_emb = create_onnx_embeddings_from_codec()
    
    # Test with dummy input
    print("\nTesting with dummy input:")
    codes = torch.randint(0, 1024, (1, 4, 10))
    codes[0, 0, 5] = 1024  # Add a mask token
    
    print(f"Input codes shape: {codes.shape}")
    print(f"Codes with mask token at position [0, 0, 5]")
    
    # Simulate ONNX embedding lookup
    embedded = torch.zeros(1, 10, 1280)
    for i in range(4):
        cb_codes = codes[0, i, :]
        cb_weight = onnx_emb[f"embedding.embeddings.{i}.weight"]
        cb_embed = torch.embedding(cb_codes, cb_weight)
        embedded += cb_embed
    
    print(f"Output embedding shape: {embedded.shape}")
    print(f"Output mean: {embedded.mean().item():.6f}")
    print(f"Output std: {embedded.std().item():.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract codec embeddings from VampNet")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/vampnet/coarse.pth",
        help="Path to VampNet model checkpoint",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="models/codec_embeddings.pth",
        help="Path to save extracted embeddings",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test extraction",
    )
    
    args = parser.parse_args()
    
    if args.test:
        test_embedding_extraction()
    else:
        extract_codec_embeddings(
            model_path=args.model_path,
            save_path=args.save_path,
        )