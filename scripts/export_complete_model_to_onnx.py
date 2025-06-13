"""
Export the complete VampNet model with transferred weights to ONNX format.

This script loads the PyTorch model with all weights properly transferred
(including codec embeddings) and exports it to ONNX format.
"""

import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort
from pathlib import Path
import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.export_vampnet_transformer_v2 import VampNetTransformerV2


def export_complete_model_to_onnx(
    weights_path: str = "models/coarse_complete_v3.pth",
    output_path: str = "onnx_models_fixed/coarse_complete_v3.onnx",
    model_type: str = "coarse",
):
    """Export complete model with transferred weights to ONNX."""
    
    print(f"\n{'='*60}")
    print(f"Exporting Complete {model_type.upper()} Model to ONNX")
    print(f"{'='*60}")
    
    # Create output directory
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Model configuration
    if model_type == "coarse":
        config = {
            'n_codebooks': 4,
            'n_conditioning_codebooks': 0,
            'vocab_size': 1024,
            'd_model': 1280,
            'n_heads': 20,
            'n_layers': 20,
            'use_gated_ffn': True
        }
    else:  # c2f
        config = {
            'n_codebooks': 14,  # 4 conditioning + 10 prediction
            'n_conditioning_codebooks': 4,
            'vocab_size': 1024,
            'd_model': 1280,
            'n_heads': 20,
            'n_layers': 16,
            'use_gated_ffn': True
        }
    
    print("\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # Create model
    print(f"\nCreating model...")
    model = VampNetTransformerV2(**config)
    
    # Load weights
    print(f"Loading weights from {weights_path}")
    state_dict = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    
    print("✓ Weights loaded successfully")
    
    # Verify embeddings are non-zero
    print("\nVerifying embeddings:")
    for i in range(config['n_codebooks']):
        emb_weight = model.embedding.embeddings[i].weight
        non_zero = (emb_weight != 0).sum().item()
        total = emb_weight.numel()
        print(f"  Codebook {i}: {non_zero}/{total} non-zero ({100*non_zero/total:.1f}%)")
    
    # Create dummy inputs for export
    batch_size = 1
    seq_len = 100  # Fixed for now
    
    dummy_codes = torch.randint(0, 1024, (batch_size, config['n_codebooks'], seq_len))
    dummy_mask = torch.zeros(batch_size, config['n_codebooks'], seq_len).bool()
    dummy_mask[:, :, 40:60] = True  # Mask middle portion
    dummy_temperature = torch.tensor(1.0)
    
    print(f"\nDummy input shapes:")
    print(f"  codes: {dummy_codes.shape}")
    print(f"  mask: {dummy_mask.shape}")
    print(f"  temperature: {dummy_temperature.shape}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    with torch.no_grad():
        output = model(dummy_codes, dummy_mask, dummy_temperature)
    print(f"✓ Output shape: {output.shape}")
    
    # Export to ONNX
    print(f"\nExporting to ONNX...")
    
    # Dynamic axes for variable batch size (seq_len is fixed for now)
    dynamic_axes = {
        'codes': {0: 'batch'},
        'mask': {0: 'batch'},
        'output': {0: 'batch'}
    }
    
    torch.onnx.export(
        model,
        (dummy_codes, dummy_mask, dummy_temperature),
        output_path,
        input_names=['codes', 'mask', 'temperature'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        opset_version=13,
        do_constant_folding=True,
        export_params=True,
    )
    
    print(f"✓ Exported to {output_path}")
    
    # Verify ONNX model
    print("\nVerifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model is valid")
    
    # Test with ONNX Runtime
    print("\nTesting with ONNX Runtime...")
    ort_session = ort.InferenceSession(output_path)
    
    # Check actual input names
    input_names = [inp.name for inp in ort_session.get_inputs()]
    print(f"Actual ONNX input names: {input_names}")
    
    # Run inference
    ort_inputs = {
        'codes': dummy_codes.numpy(),
        'mask': dummy_mask.numpy(),
    }
    
    # Only add temperature if it's an input
    if 'temperature' in input_names:
        ort_inputs['temperature'] = dummy_temperature.numpy()
    
    ort_outputs = ort_session.run(None, ort_inputs)
    ort_output = ort_outputs[0]
    
    print(f"✓ ONNX output shape: {ort_output.shape}")
    
    # Compare outputs
    torch_output = output.numpy()
    max_diff = np.abs(torch_output - ort_output).max()
    mean_diff = np.abs(torch_output - ort_output).mean()
    
    print(f"\nOutput comparison:")
    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")
    
    if max_diff < 1e-3:
        print("✓ Outputs match closely!")
    else:
        print("⚠️ Outputs have significant differences")
    
    # Print model statistics
    model_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nModel statistics:")
    print(f"  File size: {model_size:.2f} MB")
    print(f"  Input nodes: {len(ort_session.get_inputs())}")
    print(f"  Output nodes: {len(ort_session.get_outputs())}")
    
    # Print token distribution in output
    unique_tokens = len(np.unique(ort_output))
    print(f"\nOutput token distribution:")
    print(f"  Unique tokens: {unique_tokens}")
    print(f"  Min token: {ort_output.min()}")
    print(f"  Max token: {ort_output.max()}")
    
    return ort_session


def test_generation_quality(ort_session, n_codebooks=4):
    """Test the quality of token generation."""
    
    print("\n=== Testing Generation Quality ===")
    
    # Create more realistic input
    batch_size = 2
    seq_len = 100
    
    # Create partially filled sequence (like in real generation)
    codes = torch.randint(0, 1024, (batch_size, n_codebooks, seq_len))
    
    # Mask the second half for generation
    mask = torch.zeros(batch_size, n_codebooks, seq_len).bool()
    mask[:, :, 50:] = True
    codes[mask] = 1024  # Set masked positions to mask token
    
    # Test with different temperatures
    temperatures = [0.5, 0.8, 1.0, 1.2]
    
    # Check if temperature is an input
    input_names = [inp.name for inp in ort_session.get_inputs()]
    has_temperature = 'temperature' in input_names
    
    if not has_temperature:
        print("Note: Temperature was folded into constants during ONNX export")
        print("Testing with fixed temperature from export...")
    
    for temp in temperatures:
        if has_temperature:
            print(f"\nTemperature {temp}:")
        
        ort_inputs = {
            'codes': codes.numpy(),
            'mask': mask.numpy(),
        }
        
        if has_temperature:
            ort_inputs['temperature'] = np.array(temp, dtype=np.float32)
        
        output = ort_session.run(None, ort_inputs)[0]
        
        # Analyze generated tokens (masked positions only)
        generated = output[mask.numpy()]
        unique = len(np.unique(generated))
        
        print(f"  Unique tokens generated: {unique}")
        print(f"  Token range: [{generated.min()}, {generated.max()}]")
        print(f"  Most common tokens: {np.bincount(generated).argsort()[-5:][::-1].tolist()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export complete model to ONNX")
    parser.add_argument(
        "--weights",
        type=str,
        default="models/coarse_complete_v3.pth",
        help="Path to PyTorch weights file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="onnx_models_fixed/coarse_complete_v3.onnx",
        help="Output ONNX file path",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["coarse", "c2f"],
        default="coarse",
        help="Model type",
    )
    parser.add_argument(
        "--test-quality",
        action="store_true",
        help="Run generation quality tests",
    )
    
    args = parser.parse_args()
    
    # Export model
    ort_session = export_complete_model_to_onnx(
        weights_path=args.weights,
        output_path=args.output,
        model_type=args.model_type,
    )
    
    # Test generation quality if requested
    if args.test_quality:
        n_codebooks = 4 if args.model_type == "coarse" else 14
        test_generation_quality(ort_session, n_codebooks)