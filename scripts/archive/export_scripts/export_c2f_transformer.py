"""
Export VampNet C2F (Coarse-to-Fine) transformer to ONNX.
This handles the fine codebooks (4-13) for high-quality audio generation.
"""

import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort
import vampnet
from pathlib import Path
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.export_vampnet_transformer import VampNetTransformerONNX
from scripts.transfer_weights_improved import transfer_attention_weights
from scripts.fix_ffn_weight_transfer import transfer_ffn_weights
from scripts.complete_weight_transfer import transfer_embedding_weights, transfer_classifier_weights


def export_c2f_transformer():
    """Export the C2F transformer model to ONNX format."""
    
    print("=== VampNet C2F Transformer ONNX Export ===")
    
    # Load C2F checkpoint
    c2f_checkpoint_path = Path("models/vampnet/c2f.pth")
    if not c2f_checkpoint_path.exists():
        raise FileNotFoundError(f"C2F checkpoint not found at {c2f_checkpoint_path}")
    
    print(f"Loading C2F checkpoint from {c2f_checkpoint_path}")
    checkpoint = torch.load(c2f_checkpoint_path, map_location='cpu')
    
    # Extract model configuration
    if 'cfg' in checkpoint:
        cfg = checkpoint['cfg']
        print("Configuration found in checkpoint")
    else:
        print("No configuration in checkpoint, analyzing state dict...")
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        
        # Analyze the model structure
        embedding_keys = [k for k in state_dict.keys() if 'embedding' in k]
        layer_keys = [k for k in state_dict.keys() if 'layers.' in k]
        
        print(f"Found {len(embedding_keys)} embedding keys")
        print(f"Found {len(layer_keys)} layer keys")
        
        # Determine number of layers
        layer_indices = set()
        for k in layer_keys:
            if 'layers.' in k:
                parts = k.split('.')
                if len(parts) > 1 and parts[1].isdigit():
                    layer_indices.add(int(parts[1]))
        
        n_layers = len(layer_indices)
        print(f"Detected {n_layers} transformer layers")
    
    # C2F model configuration
    # C2F handles codebooks 4-13 (10 fine codebooks)
    n_codebooks = 10  # Fine codebooks only
    n_conditioning_codebooks = 4  # Conditioned on coarse codes
    vocab_size = 1024
    d_model = 1280
    n_heads = 20
    n_layers = n_layers if 'n_layers' in locals() else 16  # Default to 16 if not detected
    
    print(f"\nC2F Model Configuration:")
    print(f"  Codebooks: {n_codebooks} (fine: 4-13)")
    print(f"  Conditioning codebooks: {n_conditioning_codebooks} (coarse: 0-3)")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Model dimension: {d_model}")
    print(f"  Attention heads: {n_heads}")
    print(f"  Transformer layers: {n_layers}")
    
    # Create ONNX-compatible model
    model = VampNetTransformerONNX(
        n_codebooks=n_codebooks,
        n_conditioning_codebooks=n_conditioning_codebooks,
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=0.0,  # Disable dropout for inference
        mask_token=1024
    )
    
    model.eval()
    
    # Transfer weights from checkpoint
    print("\nTransferring weights from C2F checkpoint...")
    
    # Load state dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Transfer weights layer by layer
    total_transferred = 0
    
    # 1. Transfer transformer layer weights
    print("\n1. Transferring transformer layers...")
    for i in range(n_layers):
        # Attention weights
        attn_transferred = transfer_attention_weights(
            model.layers[i], 
            state_dict, 
            layer_idx=i, 
            prefix='net.'
        )
        total_transferred += attn_transferred
        
        # FFN weights
        ffn_transferred = transfer_ffn_weights(
            model.layers[i].ffn,
            state_dict,
            layer_idx=i,
            prefix='net.'
        )
        total_transferred += ffn_transferred
        
        print(f"  Layer {i}: {attn_transferred + ffn_transferred} weights transferred")
    
    # 2. Transfer embeddings
    print("\n2. Transferring embeddings...")
    embedding_transferred = transfer_embedding_weights(model, state_dict)
    total_transferred += embedding_transferred
    
    # 3. Transfer classifier weights
    print("\n3. Transferring classifier weights...")
    classifier_transferred = transfer_classifier_weights(model, state_dict)
    total_transferred += classifier_transferred
    
    print(f"\nTotal weights transferred: {total_transferred}")
    
    # Prepare dummy inputs for export
    batch_size = 1
    seq_length = 256
    
    # C2F takes both coarse codes (as conditioning) and generates fine codes
    coarse_codes = torch.randint(0, vocab_size, (batch_size, n_conditioning_codebooks, seq_length))
    fine_codes = torch.randint(0, vocab_size, (batch_size, n_codebooks, seq_length))
    
    # Combine for input (coarse as conditioning, fine as target)
    codes = torch.cat([coarse_codes, fine_codes], dim=1)
    
    # Create sample mask (for fine codes only)
    mask = torch.ones(batch_size, n_codebooks, seq_length).bool()
    mask[:, :, seq_length//4:] = False  # Mask last 3/4 for generation
    
    # Padding mask
    padding_mask = torch.ones(batch_size, seq_length).bool()
    
    print("\nExporting to ONNX...")
    output_path = Path("onnx_models/vampnet_c2f_transformer.onnx")
    output_path.parent.mkdir(exist_ok=True)
    
    # Export with dynamic axes for flexibility
    dynamic_axes = {
        'codes': {0: 'batch', 2: 'sequence'},
        'mask': {0: 'batch', 2: 'sequence'},
        'padding_mask': {0: 'batch', 1: 'sequence'},
        'logits': {0: 'batch', 2: 'sequence'}
    }
    
    torch.onnx.export(
        model,
        (codes, mask, padding_mask),
        str(output_path),
        input_names=['codes', 'mask', 'padding_mask'],
        output_names=['logits'],
        dynamic_axes=dynamic_axes,
        opset_version=14,
        do_constant_folding=True,
        export_params=True
    )
    
    print(f"C2F model exported to {output_path}")
    
    # Verify the exported model
    print("\nVerifying exported model...")
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print("ONNX model validation passed!")
    
    # Test inference
    print("\nTesting ONNX inference...")
    ort_session = ort.InferenceSession(str(output_path))
    
    # Run inference
    ort_inputs = {
        'codes': codes.numpy(),
        'mask': mask.numpy(),
        'padding_mask': padding_mask.numpy()
    }
    
    ort_outputs = ort_session.run(None, ort_inputs)
    logits = ort_outputs[0]
    
    print(f"Output shape: {logits.shape}")
    print(f"Expected shape: ({batch_size}, {n_codebooks}, {seq_length}, {vocab_size + 1})")
    
    # Save model info
    model_info = {
        'model_type': 'c2f_transformer',
        'n_codebooks': n_codebooks,
        'n_conditioning_codebooks': n_conditioning_codebooks,
        'vocab_size': vocab_size,
        'd_model': d_model,
        'n_heads': n_heads,
        'n_layers': n_layers,
        'input_codebooks': 'coarse (0-3) + fine (4-13)',
        'output_codebooks': 'fine (4-13)',
        'checkpoint_path': str(c2f_checkpoint_path)
    }
    
    import json
    info_path = output_path.with_suffix('.json')
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"\nModel info saved to {info_path}")
    print("\nC2F transformer export completed successfully!")
    
    return output_path


if __name__ == "__main__":
    export_c2f_transformer()