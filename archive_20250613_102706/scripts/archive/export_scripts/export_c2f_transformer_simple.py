"""
Export VampNet C2F (Coarse-to-Fine) transformer to ONNX - Simple version.
This creates the model structure first, weights can be transferred separately.
"""

import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort
from pathlib import Path
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.export_vampnet_transformer import VampNetTransformerONNX


def export_c2f_transformer_simple():
    """Export the C2F transformer model structure to ONNX format."""
    
    print("=== VampNet C2F Transformer ONNX Export (Simple) ===")
    
    # Check if C2F checkpoint exists
    c2f_checkpoint_path = Path("models/vampnet/c2f.pth")
    if c2f_checkpoint_path.exists():
        print(f"✓ Found C2F checkpoint at {c2f_checkpoint_path}")
        
        # Load checkpoint to analyze structure
        checkpoint = torch.load(c2f_checkpoint_path, map_location='cpu')
        
        # Try to determine number of layers
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
            
        # Count transformer layers
        layer_indices = set()
        for k in state_dict.keys():
            if 'layers.' in k or 'layer.' in k:
                parts = k.split('.')
                for i, part in enumerate(parts):
                    if part.isdigit() and i > 0 and 'layer' in parts[i-1]:
                        layer_indices.add(int(part))
        
        n_layers = len(layer_indices) if layer_indices else 16
        print(f"  Detected {n_layers} transformer layers")
    else:
        print(f"⚠️  C2F checkpoint not found at {c2f_checkpoint_path}")
        n_layers = 16  # Default
    
    # C2F model configuration
    # C2F handles codebooks 4-13 (10 fine codebooks)
    # It takes coarse codes (0-3) as conditioning
    n_codebooks = 10  # Fine codebooks (4-13)
    n_conditioning_codebooks = 4  # Coarse codebooks (0-3) as conditioning
    vocab_size = 1024
    d_model = 1280
    n_heads = 20
    
    print(f"\nC2F Model Configuration:")
    print(f"  Output codebooks: {n_codebooks} (indices 4-13)")
    print(f"  Conditioning codebooks: {n_conditioning_codebooks} (indices 0-3)")
    print(f"  Total input codebooks: {n_codebooks + n_conditioning_codebooks}")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Model dimension: {d_model}")
    print(f"  Attention heads: {n_heads}")
    print(f"  Transformer layers: {n_layers}")
    
    # Create ONNX-compatible model
    # Note: The model expects total codebooks in n_codebooks parameter
    total_codebooks = n_codebooks + n_conditioning_codebooks
    model = VampNetTransformerONNX(
        n_codebooks=total_codebooks,  # Total: conditioning + output
        n_conditioning_codebooks=0,  # We'll handle conditioning differently
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=0.0,  # Disable dropout for inference
        mask_token=1024
    )
    
    model.eval()
    
    print("\nModel created successfully")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Prepare dummy inputs for export
    batch_size = 1
    seq_length = 256
    
    # C2F takes both coarse codes (as conditioning) and fine codes
    # During generation, fine codes would be masked
    codes = torch.randint(0, vocab_size, (batch_size, total_codebooks, seq_length))
    
    # Create mask - for C2F, we don't mask the conditioning (coarse) codes
    # but we do mask the fine codes we want to generate
    mask = torch.zeros(batch_size, total_codebooks, seq_length).bool()
    # Only mask the fine codebooks (indices 4-13) at some positions
    mask[:, n_conditioning_codebooks:, seq_length//2:] = True
    
    # Temperature for generation (fixed for ONNX)
    temperature = torch.tensor(1.0)
    
    print("\nExporting to ONNX...")
    output_path = Path("onnx_models/vampnet_c2f_transformer.onnx")
    output_path.parent.mkdir(exist_ok=True)
    
    # Export with dynamic axes for flexibility
    dynamic_axes = {
        'codes': {0: 'batch', 2: 'sequence'},
        'mask': {0: 'batch', 2: 'sequence'},
        'temperature': {},  # Scalar
        'output': {0: 'batch', 2: 'sequence'}
    }
    
    # Test the forward pass first
    with torch.no_grad():
        test_output = model(codes, mask, temperature)
        print(f"Test forward pass successful, output shape: {test_output.shape}")
    
    torch.onnx.export(
        model,
        (codes, mask, temperature),
        str(output_path),
        input_names=['codes', 'mask', 'temperature'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        opset_version=14,
        do_constant_folding=True,
        export_params=True,
        verbose=False
    )
    
    print(f"✓ C2F model exported to {output_path}")
    
    # Verify the exported model
    print("\nVerifying exported model...")
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model validation passed!")
    
    # Test inference
    print("\nTesting ONNX inference...")
    ort_session = ort.InferenceSession(str(output_path))
    
    # Run inference
    ort_inputs = {
        'codes': codes.numpy(),
        'mask': mask.numpy(),
        'temperature': temperature.numpy()
    }
    
    ort_outputs = ort_session.run(None, ort_inputs)
    output = ort_outputs[0]
    
    print(f"✓ Inference successful!")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected: ({batch_size}, {total_codebooks}, {seq_length})")
    
    # Save model info
    model_info = {
        'model_type': 'c2f_transformer',
        'description': 'Coarse-to-Fine transformer for VampNet',
        'n_codebooks': n_codebooks,
        'n_conditioning_codebooks': n_conditioning_codebooks,
        'codebook_indices': {
            'conditioning': list(range(0, 4)),  # Coarse codes
            'output': list(range(4, 14))  # Fine codes
        },
        'vocab_size': vocab_size,
        'd_model': d_model,
        'n_heads': n_heads,
        'n_layers': n_layers,
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'checkpoint_available': c2f_checkpoint_path.exists(),
        'weights_transferred': False,
        'export_info': {
            'opset_version': 14,
            'dynamic_axes': True
        }
    }
    
    info_path = output_path.with_suffix('.json')
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"\n✓ Model info saved to {info_path}")
    
    print("\n" + "="*60)
    print("C2F transformer structure exported successfully!")
    print("Note: This export contains random weights.")
    print("To transfer pretrained weights, run the weight transfer script.")
    print("="*60)
    
    return output_path


if __name__ == "__main__":
    export_c2f_transformer_simple()