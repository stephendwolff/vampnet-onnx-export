"""
Test the weighted ONNX models to debug the mask token issue.
"""

import numpy as np
import onnxruntime as ort
from pathlib import Path


def test_model_outputs(model_path, model_name):
    """Test what tokens a model generates."""
    print(f"\n=== Testing {model_name} ===")
    print(f"Model: {model_path}")
    
    if not Path(model_path).exists():
        print("Model not found!")
        return
        
    # Load model
    session = ort.InferenceSession(model_path)
    
    # Get input info
    inputs = session.get_inputs()
    print("\nInputs:")
    for inp in inputs:
        print(f"  {inp.name}: {inp.shape}")
    
    # Prepare test inputs
    batch = 1
    n_codebooks = 4 if 'coarse' in model_name else 14
    seq_len = 100
    
    # Create random codes
    codes = np.random.randint(0, 1024, (batch, n_codebooks, seq_len), dtype=np.int64)
    
    # Create mask - mask some positions
    mask = np.zeros((batch, n_codebooks, seq_len), dtype=bool)
    mask[:, :, seq_len//2:] = True  # Mask second half
    
    # Build inputs
    feed_dict = {
        'codes': codes,
        'mask': mask
    }
    
    # Add temperature if needed
    if any(inp.name == 'temperature' for inp in inputs):
        feed_dict['temperature'] = np.array(1.0, dtype=np.float32)
    
    print(f"\nRunning inference...")
    print(f"  Input codes shape: {codes.shape}")
    print(f"  Masked positions: {mask.sum()}")
    
    # Run inference
    try:
        outputs = session.run(None, feed_dict)
        output = outputs[0]
        
        print(f"\nOutput shape: {output.shape}")
        print(f"Output range: [{output.min()}, {output.max()}]")
        
        # Check for mask tokens (1024)
        mask_tokens = (output == 1024).sum()
        print(f"Mask tokens (1024) in output: {mask_tokens}")
        
        # Show unique values
        unique_vals = np.unique(output)
        print(f"Unique values: {len(unique_vals)}")
        if len(unique_vals) < 20:
            print(f"  Values: {unique_vals}")
        else:
            print(f"  Min: {unique_vals[0]}, Max: {unique_vals[-1]}")
            
        # Check masked vs unmasked regions
        masked_output = output[mask]
        unmasked_output = output[~mask]
        
        print(f"\nMasked region stats:")
        print(f"  Unique values: {len(np.unique(masked_output))}")
        print(f"  Contains 1024: {1024 in masked_output}")
        
        print(f"\nUnmasked region stats:")
        print(f"  Unique values: {len(np.unique(unmasked_output))}")
        print(f"  Contains 1024: {1024 in unmasked_output}")
        
        # Sample some outputs
        print(f"\nSample outputs (first 10 from masked region):")
        print(f"  {masked_output[:10]}")
        
    except Exception as e:
        print(f"Error during inference: {e}")


def test_mask_token_handling():
    """Test how to handle mask tokens in the pipeline."""
    print("\n=== Testing Mask Token Handling ===")
    
    # Simulate model output with mask tokens
    output = np.array([[[500, 1024, 300, 1024, 100]]], dtype=np.int64)
    print(f"Model output: {output}")
    
    # Option 1: Clip to valid range
    clipped = np.clip(output, 0, 1023)
    print(f"Clipped: {clipped}")
    
    # Option 2: Replace mask tokens with a valid token (e.g., 0)
    replaced = np.where(output == 1024, 0, output)
    print(f"Replaced with 0: {replaced}")
    
    # Option 3: Use modulo to wrap
    wrapped = output % 1024
    print(f"Wrapped (modulo): {wrapped}")


def main():
    """Test all weighted models."""
    
    # Test coarse model
    test_model_outputs(
        "onnx_models_fixed/coarse_transformer_v2_weighted.onnx",
        "coarse"
    )
    
    # Test C2F model
    test_model_outputs(
        "onnx_models_fixed/c2f_transformer_v2_weighted.onnx",
        "c2f"
    )
    
    # Test mask token handling strategies
    test_mask_token_handling()
    
    print("\n" + "="*60)
    print("RECOMMENDATION:")
    print("The models are outputting token 1024 (mask token) in some positions.")
    print("The decoder only accepts tokens 0-1023.")
    print("Solution: Clip or replace mask tokens before decoding.")
    print("="*60)


if __name__ == "__main__":
    main()