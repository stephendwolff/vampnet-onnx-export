"""
Simple test of ONNX model generation using existing components.
"""

import torch
import numpy as np
import onnxruntime as ort
from pathlib import Path
import scipy.io.wavfile as wavfile
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_onnx_generation_simple(
    onnx_model_path: str = "onnx_models_fixed/coarse_complete_v3.onnx",
    n_codebooks: int = 4,
    seq_len: int = 100,
    mask_ratio: float = 0.5,
):
    """Test ONNX model with synthetic tokens."""
    
    print("=== Simple ONNX Generation Test ===")
    
    # Load ONNX model
    print(f"\nLoading ONNX model from {onnx_model_path}")
    ort_session = ort.InferenceSession(onnx_model_path)
    
    # Print model info
    input_names = [inp.name for inp in ort_session.get_inputs()]
    output_names = [out.name for out in ort_session.get_outputs()]
    print(f"Model inputs: {input_names}")
    print(f"Model outputs: {output_names}")
    
    # Create synthetic input tokens
    print(f"\nCreating synthetic input tokens...")
    # Use a pattern that resembles real audio tokens
    tokens = np.zeros((1, n_codebooks, seq_len), dtype=np.int64)
    
    # Create a simple pattern
    for i in range(n_codebooks):
        # Create a wave-like pattern
        base = 100 + i * 50
        for j in range(seq_len):
            tokens[0, i, j] = base + int(50 * np.sin(j * 0.1))
    
    print(f"Token shape: {tokens.shape}")
    print(f"Token range: [{tokens.min()}, {tokens.max()}]")
    
    # Create mask
    print(f"\nCreating mask with ratio {mask_ratio}")
    mask = np.random.random((1, n_codebooks, seq_len)) < mask_ratio
    print(f"Masked positions: {mask.sum()}/{mask.size} ({100*mask.sum()/mask.size:.1f}%)")
    
    # Apply mask
    masked_tokens = tokens.copy()
    masked_tokens[mask] = 1024  # mask token
    
    # Run inference
    print("\nRunning ONNX inference...")
    ort_inputs = {
        'codes': masked_tokens,
        'mask': mask
    }
    
    ort_outputs = ort_session.run(None, ort_inputs)
    generated = ort_outputs[0]
    
    print(f"\nGenerated shape: {generated.shape}")
    print(f"Generated range: [{generated.min()}, {generated.max()}]")
    print(f"Unique tokens in output: {len(np.unique(generated))}")
    
    # Analyze generation quality
    print("\n=== Generation Analysis ===")
    
    # Check if masked positions were filled
    masked_positions = mask[0]
    original_masked = tokens[0][masked_positions]
    generated_masked = generated[0][masked_positions]
    
    print(f"Original tokens at masked positions - unique values: {len(np.unique(original_masked))}")
    print(f"Generated tokens at masked positions - unique values: {len(np.unique(generated_masked))}")
    
    # Check token distribution
    print("\nToken distribution in generated output:")
    unique, counts = np.unique(generated, return_counts=True)
    top_10 = sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)[:10]
    for token, count in top_10:
        print(f"  Token {token}: {count} occurrences ({100*count/generated.size:.1f}%)")
    
    # Check if generation is diverse
    diversity_score = len(np.unique(generated)) / 1024  # normalized by vocab size
    print(f"\nDiversity score: {diversity_score:.3f} (higher is better)")
    
    # Save a visualization
    print("\nSaving visualization...")
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(n_codebooks, 1, figsize=(12, 8))
    for i in range(n_codebooks):
        ax = axes[i] if n_codebooks > 1 else axes
        ax.plot(tokens[0, i, :], label='Original', alpha=0.7)
        ax.plot(generated[0, i, :], label='Generated', alpha=0.7)
        ax.scatter(np.where(mask[0, i])[0], generated[0, i][mask[0, i]], 
                  c='red', s=10, label='Generated (masked)')
        ax.set_ylabel(f'Codebook {i}')
        ax.legend()
    
    plt.xlabel('Time step')
    plt.tight_layout()
    plt.savefig('outputs/onnx_generation_test.png')
    print("✓ Saved to outputs/onnx_generation_test.png")
    
    return generated


def test_with_real_tokens():
    """Test with tokens from a real audio file using codec."""
    
    print("\n\n=== Testing with Real Audio Tokens ===")
    
    try:
        # Load codec ONNX models
        print("Loading codec models...")
        encoder_session = ort.InferenceSession("scripts/models/vampnet_codec_encoder.onnx")
        
        # Load test audio
        print("Loading test audio...")
        sample_rate, audio = wavfile.read("test_audio/test_sine.wav")
        
        # Convert to float32 and normalize
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        
        # Ensure correct shape [batch, channels, time]
        if audio.ndim == 1:
            audio = audio[np.newaxis, np.newaxis, :]
        elif audio.ndim == 2:
            audio = audio[np.newaxis, :]
        
        print(f"Audio shape: {audio.shape}")
        
        # Encode to tokens
        print("Encoding to tokens...")
        encoder_output = encoder_session.run(None, {'audio': audio})[0]
        
        # The encoder output is tokens
        tokens = encoder_output
        print(f"Encoded token shape: {tokens.shape}")
        
        # Use only first 4 codebooks for coarse model
        coarse_tokens = tokens[:, :4, :]
        
        # Test generation with these real tokens
        print("\nTesting generation with real tokens...")
        test_onnx_generation_simple(
            n_codebooks=4,
            seq_len=coarse_tokens.shape[2],
            mask_ratio=0.7  # Higher mask ratio for more generation
        )
        
    except Exception as e:
        print(f"Error testing with real tokens: {e}")
        print("Codec models may not be available or compatible")


if __name__ == "__main__":
    # Create output directory
    Path("outputs").mkdir(exist_ok=True)
    
    # Run simple test
    generated = test_onnx_generation_simple()
    
    # Try with real tokens if possible
    test_with_real_tokens()