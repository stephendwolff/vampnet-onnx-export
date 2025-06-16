#!/usr/bin/env python3
"""
Export C2F (Coarse-to-Fine) Transformer with V11 architecture fixes.
Matches the exact architecture used in the coarse model.
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

# Import the same V11 components we used for coarse
from scripts.export_vampnet_transformer_v11_fixed import (
    VampNetEmbeddingLayer,
    OnnxRMSNorm,
    OnnxMultiheadRelativeAttention,
    VampNetGatedGELU,
    FeedForward,
    FiLMLayer,
    TransformerLayer,
    VampNetTransformerV11
)

# C2F model is the same architecture but with different parameters
class VampNetC2FTransformerV11(VampNetTransformerV11):
    """C2F Transformer with V11 architecture - expects latents as input."""
    
    def __init__(self, 
                 n_codebooks: int = 14,  # C2F uses all 14 codebooks
                 n_conditioning_codebooks: int = 4,  # First 4 are conditioning
                 vocab_size: int = 1024,
                 d_model: int = 1280,
                 n_heads: int = 20,
                 n_layers: int = 16,  # C2F has 16 layers instead of 20
                 dropout: float = 0.1,
                 latent_dim: int = 8):
        
        # Initialize parent with C2F parameters
        super().__init__(
            n_codebooks=n_codebooks,
            n_conditioning_codebooks=n_conditioning_codebooks,
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            latent_dim=latent_dim
        )


def transfer_weights_c2f(checkpoint_path, model, codec_path):
    """Transfer weights from VampNet C2F checkpoint to our model."""
    print("Transferring weights to C2F V11 model...")
    
    # Load checkpoints
    vampnet_ckpt = torch.load(checkpoint_path, map_location='cpu')
    codec_ckpt = torch.load(codec_path, map_location='cpu')
    
    # Load VampNet C2F model
    from vampnet.modules.transformer import VampNet
    from torch.nn.utils import remove_weight_norm
    
    vampnet = VampNet.load(checkpoint_path, map_location='cpu')
    
    # Remove weight normalization from classifier
    try:
        remove_weight_norm(vampnet.classifier.layers[0])
        print("✓ Removed weight normalization from classifier")
    except:
        print("⚠ No weight normalization to remove")
    
    vampnet.eval()
    
    # 1. Transfer embedding projection
    with torch.no_grad():
        model.embedding.out_proj.weight.copy_(vampnet.embedding.out_proj.weight)
        model.embedding.out_proj.bias.copy_(vampnet.embedding.out_proj.bias)
    print("✓ Transferred embedding projection")
    
    # 2. Transfer transformer layers
    for i in range(model.n_layers):
        src_layer = vampnet.transformer.layers[i]
        dst_layer = model.transformer_layers[i]
        
        # RMSNorm
        dst_layer.norm.weight.copy_(src_layer[0].weight)
        
        # Attention
        dst_layer.self_attn.in_proj_weight.copy_(src_layer[1].in_proj_weight)
        dst_layer.self_attn.in_proj_bias.copy_(src_layer[1].in_proj_bias)
        dst_layer.self_attn.out_proj.weight.copy_(src_layer[1].out_proj.weight)
        
        # Relative attention bias (only layer 0)
        if i == 0 and hasattr(src_layer[1], 'relative_attention_bias'):
            dst_layer.self_attn.relative_attention_bias.copy_(
                src_layer[1].relative_attention_bias
            )
        
        # FiLM
        if hasattr(src_layer[2], 'input_proj') and src_layer[2].input_proj is not None:
            dst_layer.film.input_proj.weight.copy_(src_layer[2].input_proj.weight)
            dst_layer.film.input_proj.bias.copy_(src_layer[2].input_proj.bias)
        
        # FFN
        dst_layer.ffn.linear1.weight.copy_(src_layer[3].linear1.weight)
        dst_layer.ffn.linear1.bias.copy_(src_layer[3].linear1.bias)
        dst_layer.ffn.linear2.weight.copy_(src_layer[3].linear2.weight)
        dst_layer.ffn.linear2.bias.copy_(src_layer[3].linear2.bias)
        
        # FFN normalization
        dst_layer.ffn.norm.weight.copy_(src_layer[3].norm.weight)
    
    print(f"✓ Transferred {model.n_layers} transformer layers")
    
    # 3. Transfer output layer norm
    model.norm_out.weight.copy_(vampnet.transformer.norm.weight)
    print("✓ Transferred output norm")
    
    # 4. Transfer classifier weights
    for cb in range(model.n_codebooks):
        # Get the Linear layer weights (transposed from Conv1d)
        conv_weight = vampnet.classifier.layers[cb].weight  # [out_channels, in_channels, kernel_size]
        linear_weight = conv_weight.squeeze(-1).t()  # [in_channels, out_channels]
        
        model.output_projections[cb].weight.copy_(linear_weight)
        model.output_projections[cb].bias.copy_(vampnet.classifier.layers[cb].bias)
    
    print(f"✓ Transferred {model.n_codebooks} output projections")
    
    # 5. Transfer codec embeddings
    codec_embeddings = []
    for cb in range(model.n_codebooks):
        if cb < len(codec_ckpt['state_dict']):
            key = f'quantizer.quantizers.{cb}.codebook.initialized'
            if key in codec_ckpt['state_dict']:
                embed_key = f'quantizer.quantizers.{cb}.codebook.embedding'
                if embed_key in codec_ckpt['state_dict']:
                    embeddings = codec_ckpt['state_dict'][embed_key]
                    codec_embeddings.append(embeddings)
    
    if codec_embeddings:
        # Stack embeddings: [n_codebooks, vocab_size, latent_dim]
        codec_embeddings = torch.stack(codec_embeddings[:model.n_codebooks])
        model.codec_embeddings.copy_(codec_embeddings)
        print(f"✓ Loaded codec embeddings: {codec_embeddings.shape}")
    
    print("✓ Weight transfer complete!")


def main():
    print("="*80)
    print("EXPORTING C2F TRANSFORMER V11")
    print("="*80)
    
    # Create model
    model = VampNetC2FTransformerV11()
    print(f"\nModel configuration:")
    print(f"  n_codebooks: {model.n_codebooks}")
    print(f"  n_conditioning_codebooks: {model.n_conditioning_codebooks}")
    print(f"  n_layers: {model.n_layers}")
    print(f"  d_model: {model.d_model}")
    print(f"  n_heads: {model.n_heads}")
    
    # Transfer weights
    transfer_weights_c2f("models/vampnet/c2f.pth", model, "models/vampnet/codec.pth")
    
    # Test model
    print("\nTesting model...")
    model.eval()
    with torch.no_grad():
        # Test input: [batch, n_codebooks * latent_dim, seq_len]
        test_latents = torch.randn(1, 14 * 8, 50)  # 14 codebooks for C2F
        output = model(test_latents)
        print(f"Input shape: {test_latents.shape}")
        print(f"Output shape: {output.shape}")  # Should be [1, 14, 50, 1025]
    
    # Export to ONNX
    print("\nExporting to ONNX...")
    dummy_input = torch.randn(1, 14 * 8, 100)
    
    torch.onnx.export(
        model,
        dummy_input,
        "vampnet_c2f_transformer_v11.onnx",
        input_names=['latents'],
        output_names=['logits'],
        dynamic_axes={
            'latents': {0: 'batch', 2: 'sequence'},
            'logits': {0: 'batch', 2: 'sequence'}
        },
        opset_version=14,
        verbose=False
    )
    
    print("✓ Exported to vampnet_c2f_transformer_v11.onnx")
    
    # Save PyTorch checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'n_codebooks': model.n_codebooks,
            'n_conditioning_codebooks': model.n_conditioning_codebooks,
            'vocab_size': model.vocab_size,
            'latent_dim': model.latent_dim,
            'd_model': model.d_model,
            'n_heads': model.n_heads,
            'n_layers': model.n_layers
        }
    }, 'vampnet_c2f_transformer_v11.pth')
    
    print("✓ Saved model to vampnet_c2f_transformer_v11.pth")
    
    # Test ONNX model
    print("\nTesting ONNX model...")
    import onnxruntime as ort
    ort_session = ort.InferenceSession("vampnet_c2f_transformer_v11.onnx")
    
    onnx_output = ort_session.run(None, {'latents': test_latents.numpy()})[0]
    print(f"ONNX output shape: {onnx_output.shape}")
    
    # Compare outputs
    pytorch_output = output.numpy()
    diff = np.abs(pytorch_output - onnx_output).max()
    print(f"Max difference PyTorch vs ONNX: {diff:.6f}")
    
    if diff < 1e-4:
        print("\n✅ C2F export successful!")
    else:
        print("\n⚠️  Warning: Large difference between PyTorch and ONNX")
    
    print("="*80)


if __name__ == "__main__":
    main()