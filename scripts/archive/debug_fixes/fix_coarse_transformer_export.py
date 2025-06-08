"""
Fix and re-export the coarse transformer with correct architecture.
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


def create_fixed_coarse_transformer():
    """Create coarse transformer with correct VampNet architecture."""
    
    print("=== Creating Fixed Coarse Transformer ===")
    
    # Correct VampNet architecture parameters
    n_codebooks = 4  # Coarse only
    n_conditioning_codebooks = 0  # No conditioning for coarse
    vocab_size = 1024
    d_model = 1280  # Correct dimension
    n_heads = 20    # Correct number of heads (NOT 8!)
    n_layers = 20   # Correct number of layers
    
    print(f"\nModel Configuration:")
    print(f"  Codebooks: {n_codebooks}")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Model dimension: {d_model}")
    print(f"  Attention heads: {n_heads}")
    print(f"  Head dimension: {d_model // n_heads} (= {d_model}/{n_heads})")
    print(f"  Transformer layers: {n_layers}")
    
    # Create model
    model = VampNetTransformerONNX(
        n_codebooks=n_codebooks,
        n_conditioning_codebooks=n_conditioning_codebooks,
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=0.0,  # No dropout for inference
        mask_token=1024
    )
    
    model.eval()
    
    # Check if we have existing weights to load
    existing_weights = [
        "vampnet_onnx_weights_complete.pth",
        "vampnet_onnx_weights.pth",
        "scripts/vampnet_onnx_weights.pth"
    ]
    
    for weight_file in existing_weights:
        if Path(weight_file).exists():
            print(f"\nLoading existing weights from {weight_file}")
            try:
                state_dict = torch.load(weight_file, map_location='cpu', weights_only=True)
                model.load_state_dict(state_dict, strict=False)
                print("✓ Loaded weights (partial match okay)")
                break
            except Exception as e:
                print(f"  Could not load weights: {e}")
    
    return model, {
        'n_codebooks': n_codebooks,
        'n_conditioning_codebooks': n_conditioning_codebooks,
        'vocab_size': vocab_size,
        'd_model': d_model,
        'n_heads': n_heads,
        'n_layers': n_layers
    }


def test_model_shapes(model):
    """Test the model with various input shapes."""
    
    print("\n\n=== Testing Model Shapes ===")
    
    test_cases = [
        (1, 4, 100, "Original fixed length"),
        (1, 4, 173, "Pipeline typical length"),
        (1, 4, 256, "Longer sequence"),
        (2, 4, 173, "Batch size 2"),
        (1, 4, 512, "Very long sequence")
    ]
    
    for batch, codebooks, seq_len, desc in test_cases:
        print(f"\nTest: {desc} - batch={batch}, codebooks={codebooks}, seq={seq_len}")
        
        try:
            with torch.no_grad():
                codes = torch.randint(0, 1024, (batch, codebooks, seq_len))
                mask = torch.zeros(batch, codebooks, seq_len).bool()
                mask[:, :, seq_len//2:] = True  # Mask second half
                temperature = torch.tensor(1.0)
                
                output = model(codes, mask, temperature)
                print(f"  ✓ Success! Output shape: {output.shape}")
                
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            return False
    
    return True


def export_fixed_transformer(model, config):
    """Export the fixed transformer to ONNX with proper dynamic axes."""
    
    print("\n\n=== Exporting Fixed Transformer ===")
    
    # Prepare dummy inputs
    batch_size = 1
    seq_length = 256  # Use a reasonable default
    n_codebooks = config['n_codebooks']
    vocab_size = config['vocab_size']
    
    codes = torch.randint(0, vocab_size, (batch_size, n_codebooks, seq_length))
    mask = torch.zeros(batch_size, n_codebooks, seq_length).bool()
    mask[:, :, seq_length//2:] = True
    temperature = torch.tensor(1.0)
    
    # Output path
    output_dir = Path("onnx_models_fixed")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "coarse_transformer_fixed.onnx"
    
    print(f"\nExporting to: {output_path}")
    
    # Export with PROPER dynamic axes
    dynamic_axes = {
        'codes': {0: 'batch', 2: 'sequence'},  # batch and sequence are dynamic
        'mask': {0: 'batch', 2: 'sequence'},   # batch and sequence are dynamic
        'temperature': {},  # scalar, no dynamic axes
        'output': {0: 'batch', 2: 'sequence'}  # batch and sequence are dynamic
    }
    
    print("\nDynamic axes configuration:")
    for name, axes in dynamic_axes.items():
        print(f"  {name}: {axes}")
    
    try:
        torch.onnx.export(
            model,
            (codes, mask, temperature),
            str(output_path),
            input_names=['codes', 'mask', 'temperature'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            opset_version=14,  # Use newer opset
            do_constant_folding=True,
            export_params=True,
            verbose=False
        )
        
        print("✓ Export successful!")
        
    except Exception as e:
        print(f"✗ Export failed: {e}")
        return None
    
    # Verify the exported model
    print("\nVerifying exported model...")
    try:
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model validation passed!")
    except Exception as e:
        print(f"✗ Validation failed: {e}")
        return None
    
    # Test with ONNX Runtime
    print("\nTesting with ONNX Runtime...")
    try:
        ort_session = ort.InferenceSession(str(output_path))
        
        # Test different shapes
        test_shapes = [(1, 4, 173), (2, 4, 256), (1, 4, 512)]
        
        for batch, cb, seq in test_shapes:
            print(f"\n  Testing shape: batch={batch}, codebooks={cb}, seq={seq}")
            
            test_codes = np.random.randint(0, vocab_size, (batch, cb, seq), dtype=np.int64)
            test_mask = np.zeros((batch, cb, seq), dtype=bool)
            test_mask[:, :, seq//2:] = True
            test_temp = np.array(1.0, dtype=np.float32)
            
            ort_outputs = ort_session.run(
                None,
                {
                    'codes': test_codes,
                    'mask': test_mask,
                    'temperature': test_temp
                }
            )
            
            print(f"    ✓ Success! Output shape: {ort_outputs[0].shape}")
            
    except Exception as e:
        print(f"✗ Runtime test failed: {e}")
        return None
    
    # Save model info
    info_path = output_path.with_suffix('.json')
    model_info = {
        'model_type': 'coarse_transformer_fixed',
        'description': 'Fixed VampNet coarse transformer with correct architecture',
        **config,
        'export_info': {
            'opset_version': 14,
            'dynamic_axes': ['batch', 'sequence'],
            'fixed_issues': [
                'Corrected n_heads from 8 to 20',
                'Enabled truly dynamic sequence length',
                'Fixed attention reshape dimensions'
            ]
        }
    }
    
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"\n✓ Model info saved to {info_path}")
    
    return output_path


def main():
    """Main function to fix and re-export the coarse transformer."""
    
    print("=== Fixing Coarse Transformer Export ===\n")
    
    # Create fixed model
    model, config = create_fixed_coarse_transformer()
    
    # Test shapes
    if not test_model_shapes(model):
        print("\n✗ Model shape tests failed!")
        return
    
    # Export fixed model
    output_path = export_fixed_transformer(model, config)
    
    if output_path:
        print("\n" + "="*60)
        print("SUCCESS! Fixed transformer exported to:")
        print(f"  {output_path}")
        print("\nTo use this model, update your pipeline to use:")
        print(f"  coarse_path = '{output_path}'")
        print("="*60)
    else:
        print("\n✗ Export failed!")


if __name__ == "__main__":
    main()