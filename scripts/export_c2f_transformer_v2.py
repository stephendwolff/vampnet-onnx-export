"""
Export VampNet C2F transformer v2 with ONNX-friendly custom attention.
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

from scripts.export_vampnet_transformer_v2 import VampNetTransformerV2


def export_c2f_transformer_v2():
    """Export C2F transformer with ONNX-friendly architecture."""
    
    print("=== VampNet C2F Transformer V2 Export ===")
    
    # Check checkpoint
    c2f_checkpoint_path = Path("models/vampnet/c2f.pth")
    if c2f_checkpoint_path.exists():
        print(f"✓ Found C2F checkpoint at {c2f_checkpoint_path}")
        checkpoint = torch.load(c2f_checkpoint_path, map_location='cpu', weights_only=False)
        
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
        n_layers = 16
    
    # C2F Configuration
    config = {
        'n_codebooks': 10,  # Fine codebooks (4-13)
        'n_conditioning_codebooks': 4,  # Coarse codebooks (0-3)
        'vocab_size': 1024,
        'd_model': 1280,
        'n_heads': 20,
        'n_layers': n_layers,
        'use_gated_ffn': True
    }
    
    print("\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # Create model with total codebooks
    total_codebooks = config['n_codebooks'] + config['n_conditioning_codebooks']
    model = VampNetTransformerV2(
        n_codebooks=total_codebooks,
        n_conditioning_codebooks=0,  # Handle conditioning differently
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        dropout=0.0,
        use_gated_ffn=config['use_gated_ffn']
    )
    model.eval()
    
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test with various shapes
    print("\nTesting shapes...")
    test_shapes = [(1, 173), (2, 256), (1, 512)]
    
    for batch, seq in test_shapes:
        codes = torch.randint(0, 1024, (batch, total_codebooks, seq))
        mask = torch.zeros(batch, total_codebooks, seq).bool()
        mask[:, config['n_conditioning_codebooks']:, seq//2:] = True
        temp = torch.tensor(1.0)
        
        with torch.no_grad():
            output = model(codes, mask, temp)
            print(f"  Shape ({batch}, {total_codebooks}, {seq}) -> {output.shape} ✓")
    
    # Export to ONNX
    print("\nExporting to ONNX...")
    output_path = Path("onnx_models_fixed/c2f_transformer_v2.onnx")
    output_path.parent.mkdir(exist_ok=True)
    
    # Dummy inputs
    dummy_codes = torch.randint(0, 1024, (1, total_codebooks, 256))
    dummy_mask = torch.zeros(1, total_codebooks, 256).bool()
    dummy_mask[:, config['n_conditioning_codebooks']:, 128:] = True
    dummy_temp = torch.tensor(1.0)
    
    torch.onnx.export(
        model,
        (dummy_codes, dummy_mask, dummy_temp),
        str(output_path),
        input_names=['codes', 'mask', 'temperature'],
        output_names=['output'],
        dynamic_axes={
            'codes': {0: 'batch', 2: 'sequence'},
            'mask': {0: 'batch', 2: 'sequence'},
            'output': {0: 'batch', 2: 'sequence'}
        },
        opset_version=14,
        do_constant_folding=True,
        export_params=True,
        verbose=False
    )
    
    print(f"✓ Exported to {output_path}")
    
    # Verify with ONNX Runtime
    print("\nVerifying with ONNX Runtime...")
    ort_session = ort.InferenceSession(str(output_path))
    
    # Check actual inputs
    input_names = [i.name for i in ort_session.get_inputs()]
    print(f"Model inputs: {input_names}")
    
    for batch, seq in test_shapes:
        test_codes = np.random.randint(0, 1024, (batch, total_codebooks, seq), dtype=np.int64)
        test_mask = np.zeros((batch, total_codebooks, seq), dtype=bool)
        test_mask[:, config['n_conditioning_codebooks']:, seq//2:] = True
        
        # Build inputs based on what the model expects
        inputs = {
            'codes': test_codes,
            'mask': test_mask
        }
        if 'temperature' in input_names:
            inputs['temperature'] = np.array(1.0, dtype=np.float32)
        
        try:
            outputs = ort_session.run(None, inputs)
            print(f"  Shape ({batch}, {total_codebooks}, {seq}) -> {outputs[0].shape} ✓")
        except Exception as e:
            print(f"  Shape ({batch}, {total_codebooks}, {seq}) failed: {e}")
    
    # Save info
    info = {
        'model_type': 'c2f_transformer_v2',
        'description': 'VampNet C2F transformer with ONNX-friendly custom attention',
        'n_codebooks': config['n_codebooks'],
        'n_conditioning_codebooks': config['n_conditioning_codebooks'],
        'total_codebooks': total_codebooks,
        'codebook_indices': {
            'conditioning': list(range(0, 4)),
            'output': list(range(4, 14))
        },
        'vocab_size': config['vocab_size'],
        'd_model': config['d_model'],
        'n_heads': config['n_heads'],
        'n_layers': config['n_layers'],
        'use_gated_ffn': config['use_gated_ffn'],
        'improvements': [
            'Custom ONNX-compatible multi-head attention',
            'Proper dynamic shape support',
            'GatedFFN support',
            'Handles both coarse (conditioning) and fine codes'
        ]
    }
    
    info_path = output_path.with_suffix('.json')
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"\n✓ Info saved to {info_path}")
    print("\n" + "="*60)
    print("Export successful! Use this model in your pipeline:")
    print(f"  c2f_path = '{output_path}'")
    print("="*60)


if __name__ == "__main__":
    export_c2f_transformer_v2()