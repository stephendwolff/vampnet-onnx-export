"""
Complete weight transfer from VampNet to ONNX including codec embeddings.

This script performs a complete weight transfer:
1. Transfers transformer weights (attention, FFN, norms)
2. Extracts and transfers codec embeddings
3. Handles special tokens (MASK)
4. Adjusts output projections if needed
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
import os
import argparse
import audiotools as at
from vampnet import interface as vampnet

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.export_vampnet_transformer_v2 import VampNetTransformerV2


def extract_codec_embeddings_for_transfer(interface, n_codebooks=4, vocab_size=1024, model_type="coarse"):
    """Extract codec embeddings from VampNet interface."""
    
    print("\n=== Extracting Codec Embeddings ===")
    
    # Get the appropriate model
    if model_type == "coarse":
        model = interface.coarse
    else:
        model = interface.c2f
    
    codec = interface.codec
    
    embeddings = {}
    
    # For C2F, we need embeddings for all 14 codebooks
    actual_n_codebooks = min(n_codebooks, len(codec.quantizer.quantizers))
    print(f"Extracting embeddings for {actual_n_codebooks} codebooks")
    
    # Extract embeddings for each codebook
    for i in range(actual_n_codebooks):
        # Get the quantizer codebook weights
        quantizer = codec.quantizer.quantizers[i]
        codebook_weights = quantizer.codebook.weight.data.clone()
        
        print(f"Codebook {i}: shape {codebook_weights.shape}")
        
        # Get special token embeddings (MASK)
        if hasattr(model.embedding, "special") and "MASK" in model.embedding.special:
            mask_embedding = model.embedding.special["MASK"][i].data.clone()
            # Concatenate codebook weights with mask token
            full_embeddings = torch.cat([codebook_weights, mask_embedding.unsqueeze(0)], dim=0)
            print(f"  Added MASK token, new shape: {full_embeddings.shape}")
        else:
            full_embeddings = codebook_weights
            
        embeddings[f"codebook_{i}"] = full_embeddings
    
    # Get output projection if it exists
    if hasattr(model.embedding, "out_proj"):
        embeddings["out_proj_weight"] = model.embedding.out_proj.weight.data.clone()
        embeddings["out_proj_bias"] = model.embedding.out_proj.bias.data.clone()
        print(f"\nOutput projection: {embeddings['out_proj_weight'].shape}")
    
    # Get special mask embeddings
    if hasattr(model.embedding, "special") and hasattr(model.embedding, "special_idxs"):
        embeddings["special_mask"] = model.embedding.special["MASK"].data.clone()
        embeddings["mask_token_idx"] = model.embedding.special_idxs["MASK"]
        print(f"\nSpecial MASK embedding: {embeddings['special_mask'].shape}")
        print(f"MASK token index: {embeddings['mask_token_idx']}")
    
    return embeddings


def transfer_codec_embeddings_to_onnx(codec_embeddings, onnx_model, n_codebooks=4, d_model=1280):
    """Transfer codec embeddings to ONNX model."""
    
    print("\n=== Transferring Codec Embeddings to ONNX ===")
    
    for i in range(n_codebooks):
        key = f"codebook_{i}"
        if key in codec_embeddings:
            codec_emb = codec_embeddings[key]
            latent_dim = codec_emb.shape[-1]
            vocab_size = codec_emb.shape[0]
            
            print(f"\nCodebook {i}:")
            print(f"  Codec embedding shape: {codec_emb.shape}")
            print(f"  ONNX embedding shape: {onnx_model.embedding.embeddings[i].weight.shape}")
            
            # Apply projection if available
            if "out_proj_weight" in codec_embeddings:
                proj_weight = codec_embeddings["out_proj_weight"]
                proj_bias = codec_embeddings["out_proj_bias"]
                
                # Handle Conv1d weight shape [out_channels, in_channels, kernel_size]
                if proj_weight.dim() == 3:
                    proj_weight = proj_weight.squeeze(-1)  # Remove kernel dimension
                
                # Extract relevant portion for this codebook
                start_idx = i * latent_dim
                end_idx = (i + 1) * latent_dim
                cb_proj_weight = proj_weight[:, start_idx:end_idx]
                
                # Project embeddings
                projected_emb = torch.matmul(codec_emb, cb_proj_weight.t())
                if proj_bias is not None:
                    projected_emb = projected_emb + proj_bias / n_codebooks
                
                # Assign to ONNX model
                with torch.no_grad():
                    onnx_model.embedding.embeddings[i].weight.data[:vocab_size] = projected_emb[:vocab_size]
                    
                print(f"  Transferred {vocab_size} embeddings with projection")
            else:
                # Direct transfer (may need padding)
                with torch.no_grad():
                    if latent_dim == d_model:
                        onnx_model.embedding.embeddings[i].weight.data[:vocab_size] = codec_emb[:vocab_size]
                    elif latent_dim < d_model:
                        # Pad with zeros
                        onnx_model.embedding.embeddings[i].weight.data[:vocab_size, :latent_dim] = codec_emb[:vocab_size]
                        onnx_model.embedding.embeddings[i].weight.data[:vocab_size, latent_dim:] = 0
                    else:
                        # Truncate
                        onnx_model.embedding.embeddings[i].weight.data[:vocab_size] = codec_emb[:vocab_size, :d_model]
                
                print(f"  Transferred {vocab_size} embeddings directly")


def transfer_transformer_weights(vampnet_model, onnx_model, model_type="coarse"):
    """Transfer transformer weights from VampNet to ONNX model."""
    
    print(f"\n=== Transferring {model_type} Transformer Weights ===")
    
    transferred = 0
    skipped = 0
    
    # Map VampNet transformer layers to ONNX layers
    for i in range(len(onnx_model.layers)):
        print(f"\nLayer {i}:")
        
        # Self-attention weights
        mappings = [
            # VampNet Q weights are combined, ONNX uses separate
            (f"transformer.layers.{i}.self_attn.w_qs.weight", 
             f"layers.{i}.self_attn.w_q.weight",
             lambda x: x),  # Direct mapping
            
            # K and V weights
            (f"transformer.layers.{i}.self_attn.w_ks.weight",
             f"layers.{i}.self_attn.w_k.weight",
             lambda x: x),
            
            (f"transformer.layers.{i}.self_attn.w_vs.weight",
             f"layers.{i}.self_attn.w_v.weight",
             lambda x: x),
            
            # Output projection
            (f"transformer.layers.{i}.self_attn.fc.weight",
             f"layers.{i}.self_attn.w_o.weight",
             lambda x: x),
            
            # Layer norms
            (f"transformer.layers.{i}.norm_1.weight",
             f"layers.{i}.norm1.weight",
             lambda x: x),
            
            (f"transformer.layers.{i}.norm_3.weight",
             f"layers.{i}.norm2.weight",
             lambda x: x),
            
            # FFN weights - VampNet uses feed_forward, not ff
            (f"transformer.layers.{i}.feed_forward.w_1.weight",
             f"layers.{i}.ffn.w_1.weight",
             lambda x: x),
            
            (f"transformer.layers.{i}.feed_forward.w_2.weight",
             f"layers.{i}.ffn.w_2.weight",
             lambda x: x),
        ]
        
        # Add relative attention bias for layer 0
        if i == 0 and hasattr(vampnet_model.transformer.layers[0].self_attn, 'relative_attention_bias'):
            mappings.append((
                f"transformer.layers.0.self_attn.relative_attention_bias.weight",
                f"layers.0.self_attn.relative_attention_bias.weight",
                lambda x: x
            ))
        
        # Apply mappings
        for vampnet_key, onnx_key, transform in mappings:
            if hasattr(vampnet_model, vampnet_key.split('.')[0]):
                try:
                    # Navigate through the model structure
                    vampnet_param = vampnet_model
                    for part in vampnet_key.split('.'):
                        vampnet_param = getattr(vampnet_param, part)
                    
                    onnx_param = onnx_model
                    for part in onnx_key.split('.'):
                        onnx_param = getattr(onnx_param, part)
                    
                    # Transfer weight
                    transformed = transform(vampnet_param.data)
                    onnx_param.data.copy_(transformed)
                    
                    transferred += 1
                    print(f"  ✓ {vampnet_key} -> {onnx_key}")
                    
                except Exception as e:
                    print(f"  ✗ Failed to transfer {vampnet_key}: {e}")
                    skipped += 1
            else:
                skipped += 1
    
    # Final norm
    try:
        onnx_model.final_norm.weight.data.copy_(vampnet_model.transformer.norm.weight.data)
        print("\n✓ Transferred final norm")
        transferred += 1
    except:
        print("\n✗ Failed to transfer final norm")
        skipped += 1
    
    print(f"\n=== Transfer Summary ===")
    print(f"Transferred: {transferred} weights")
    print(f"Skipped: {skipped} weights")
    
    return transferred, skipped


def transfer_output_projections(vampnet_model, onnx_model, n_codebooks=4, n_conditioning_codebooks=0):
    """Transfer output projection weights with dimension adjustment."""
    
    print("\n=== Transferring Output Projections ===")
    
    # VampNet uses a single large classifier
    # ONNX uses separate projections per codebook
    
    vampnet_classifier = vampnet_model.classifier.layers[0]  # WNConv1d
    
    # Get weight_v and weight_g for weight normalization
    if hasattr(vampnet_classifier, 'weight_v'):
        # Reconstruct the actual weight
        weight_v = vampnet_classifier.weight_v
        weight_g = vampnet_classifier.weight_g
        
        # Compute normalized weight
        norm = weight_v.norm(2, 1, keepdim=True)
        weight = weight_v * (weight_g / norm)
        
        print(f"VampNet classifier weight shape: {weight.shape}")
        
        # For C2F, only transfer weights for non-conditioning codebooks
        n_predict_codebooks = n_codebooks - n_conditioning_codebooks
        
        # Split the weight across codebooks
        vocab_size = weight.shape[0] // n_predict_codebooks
        
        # For C2F, we only have output projections for non-conditioning codebooks
        # ONNX model has projections for all codebooks, but C2F only generates for some
        
        for i in range(n_codebooks):
            # Skip conditioning codebooks for C2F
            if n_conditioning_codebooks > 0 and i < n_conditioning_codebooks:
                print(f"\nCodebook {i}: Skipping (conditioning codebook)")
                continue
                
            # Map from output index to VampNet classifier index
            vampnet_idx = i - n_conditioning_codebooks if n_conditioning_codebooks > 0 else i
            
            if vampnet_idx < 0 or vampnet_idx >= n_predict_codebooks:
                print(f"\nCodebook {i}: Out of range")
                continue
                
            start_idx = vampnet_idx * vocab_size
            end_idx = (vampnet_idx + 1) * vocab_size
            
            # Extract slice for this codebook
            cb_weight = weight[start_idx:end_idx]
            
            # ONNX expects different dimensions, may need to adjust
            # For C2F, output projections are indexed differently
            onnx_proj_idx = i - n_conditioning_codebooks if n_conditioning_codebooks > 0 else i
            
            if onnx_proj_idx >= len(onnx_model.output_projs):
                print(f"\nCodebook {i}: No corresponding output projection in ONNX model")
                continue
                
            onnx_proj = onnx_model.output_projs[onnx_proj_idx]
            
            print(f"\nCodebook {i} (VampNet idx {vampnet_idx}):")
            print(f"  VampNet slice shape: {cb_weight.shape}")
            print(f"  ONNX projection shape: {onnx_proj.weight.shape}")
            
            # Transfer what we can
            with torch.no_grad():
                if cb_weight.shape[0] <= onnx_proj.weight.shape[0]:
                    # Squeeze the conv dimension if needed
                    if cb_weight.dim() == 3:
                        cb_weight = cb_weight.squeeze(-1)
                    onnx_proj.weight.data[:cb_weight.shape[0]] = cb_weight
                    print(f"  ✓ Transferred {cb_weight.shape[0]} output dimensions")
                else:
                    print(f"  ✗ Shape mismatch, skipping")
    else:
        print("✗ VampNet classifier doesn't have weight_v attribute")


def complete_weight_transfer(
    checkpoint_path: str,
    output_path: str,
    model_type: str = "coarse",
    n_codebooks: int = None,
    n_conditioning_codebooks: int = None,
):
    """Perform complete weight transfer including embeddings."""
    
    print(f"\n{'='*60}")
    print(f"Complete Weight Transfer for {model_type.upper()} Model")
    print(f"{'='*60}")
    
    # Set default parameters based on model type
    if model_type == "coarse":
        n_codebooks = n_codebooks or 4
        n_conditioning_codebooks = n_conditioning_codebooks or 0
    else:  # c2f
        n_codebooks = n_codebooks or 14
        n_conditioning_codebooks = n_conditioning_codebooks or 4
    
    # Load VampNet interface
    print(f"\nLoading VampNet from {checkpoint_path}")
    
    if model_type == "coarse":
        interface = vampnet.Interface(
            coarse_ckpt=checkpoint_path,
            coarse2fine_ckpt=None,
            codec_ckpt="models/vampnet/codec.pth",
            device="cpu"
        )
        vampnet_model = interface.coarse
    else:  # c2f
        interface = vampnet.Interface(
            coarse_ckpt="models/vampnet/coarse.pth",  # Need coarse for codec
            coarse2fine_ckpt=checkpoint_path,
            codec_ckpt="models/vampnet/codec.pth",
            device="cpu"
        )
        vampnet_model = interface.c2f
    
    # Create ONNX model
    print(f"\nCreating ONNX {model_type} model")
    
    if model_type == "coarse":
        onnx_model = VampNetTransformerV2(
            n_codebooks=n_codebooks,
            n_conditioning_codebooks=n_conditioning_codebooks,
            vocab_size=1024,
            d_model=1280,
            n_heads=20,
            n_layers=20,
            use_gated_ffn=True
        )
    else:  # c2f
        onnx_model = VampNetTransformerV2(
            n_codebooks=n_codebooks,
            n_conditioning_codebooks=n_conditioning_codebooks,
            vocab_size=1024,
            d_model=1280,
            n_heads=20,
            n_layers=16,  # C2F uses 16 layers
            use_gated_ffn=True
        )
    
    # 1. Extract codec embeddings
    codec_embeddings = extract_codec_embeddings_for_transfer(interface, n_codebooks, model_type=model_type)
    
    # 2. Transfer codec embeddings to ONNX
    transfer_codec_embeddings_to_onnx(codec_embeddings, onnx_model, n_codebooks)
    
    # 3. Transfer transformer weights
    transferred, skipped = transfer_transformer_weights(vampnet_model, onnx_model, model_type)
    
    # 4. Transfer output projections
    transfer_output_projections(vampnet_model, onnx_model, n_codebooks, n_conditioning_codebooks)
    
    # 5. Save the complete model
    print(f"\n=== Saving Complete Model ===")
    torch.save(onnx_model.state_dict(), output_path)
    print(f"✓ Saved to {output_path}")
    
    # Test the model
    print("\n=== Testing Model ===")
    try:
        test_complete_model(onnx_model, n_codebooks, n_conditioning_codebooks)
    except Exception as e:
        print(f"Warning: Test failed with error: {e}")
        print("Model saved successfully despite test failure.")
    
    return onnx_model


def test_complete_model(model, n_codebooks=4, n_conditioning_codebooks=0):
    """Test the model with dummy input."""
    
    model.eval()
    
    # Create test input
    batch_size = 1
    seq_len = 50
    codes = torch.randint(0, 1024, (batch_size, n_codebooks, seq_len))
    mask = torch.zeros(batch_size, n_codebooks, seq_len).bool()
    
    # Only mask non-conditioning codebooks
    if n_conditioning_codebooks > 0:
        mask[:, n_conditioning_codebooks:, 20:30] = True  # Mask middle portion of non-conditioning
    else:
        mask[:, :, 20:30] = True  # Mask middle portion
    
    temperature = torch.tensor(1.0)
    
    # Add some mask tokens
    codes[mask] = 1024
    
    print(f"Input shape: {codes.shape}")
    print(f"Masked positions: {mask.sum().item()}")
    
    # Forward pass
    with torch.no_grad():
        output = model(codes, mask, temperature)
    
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.2f}, {output.max().item():.2f}]")
    
    # Check diversity
    unique_tokens = len(torch.unique(output))
    print(f"Unique tokens in output: {unique_tokens}")
    
    # Check if embeddings are working
    if hasattr(model, 'embedding'):
        # Get embedding statistics
        for i in range(n_codebooks):
            emb_weight = model.embedding.embeddings[i].weight
            print(f"\nCodebook {i} embedding stats:")
            print(f"  Mean: {emb_weight.mean().item():.6f}")
            print(f"  Std: {emb_weight.std().item():.6f}")
            print(f"  Non-zero: {(emb_weight != 0).sum().item()} / {emb_weight.numel()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Complete weight transfer from VampNet to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/vampnet/coarse.pth",
        help="Path to VampNet checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/coarse_complete_v3.pth",
        help="Output path for ONNX weights",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["coarse", "c2f"],
        default="coarse",
        help="Model type to transfer",
    )
    
    args = parser.parse_args()
    
    complete_weight_transfer(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        model_type=args.model_type,
    )