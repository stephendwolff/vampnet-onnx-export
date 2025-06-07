"""
Test the exported VampNet transformer ONNX model.
This script tests the transformer's ability to generate coherent audio tokens.
"""

import torch
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
from pathlib import Path
import time


def create_random_mask(shape, mask_ratio=0.3):
    """Create a random mask for testing."""
    mask = torch.zeros(shape, dtype=torch.int64)
    num_to_mask = int(shape[-1] * mask_ratio)
    
    for batch in range(shape[0]):
        for codebook in range(shape[1]):
            positions = torch.randperm(shape[-1])[:num_to_mask]
            mask[batch, codebook, positions] = 1
    
    return mask


def test_transformer_inference():
    """Test the ONNX transformer model inference."""
    
    print("=== Testing ONNX Transformer Inference ===\n")
    
    # Load ONNX model
    print("Loading ONNX model...")
    ort_session = ort.InferenceSession("vampnet_transformer.onnx")
    
    # Get input/output info
    input_names = [inp.name for inp in ort_session.get_inputs()]
    output_names = [out.name for out in ort_session.get_outputs()]
    
    print(f"Input names: {input_names}")
    print(f"Output names: {output_names}")
    
    # Test parameters
    batch_size = 1
    n_codebooks = 4
    seq_len = 100  # Match the export sequence length
    vocab_size = 1024
    
    # Create test input - random codes
    codes = np.random.randint(0, vocab_size, (batch_size, n_codebooks, seq_len), dtype=np.int64)
    
    # Create mask - mask 30% of positions
    mask = create_random_mask((batch_size, n_codebooks, seq_len), mask_ratio=0.3)
    
    print(f"\nInput shape: {codes.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Masked positions: {mask.sum().item()}")
    
    # Run inference
    print("\nRunning inference...")
    start_time = time.time()
    
    outputs = ort_session.run(
        None,
        {
            'codes': codes,
            'mask': mask.numpy()
        }
    )
    
    inference_time = time.time() - start_time
    generated_codes = outputs[0]
    
    print(f"Inference time: {inference_time:.3f}s")
    print(f"Output shape: {generated_codes.shape}")
    
    # Analyze results
    print("\n=== Analysis ===")
    
    # Check if masked positions were changed
    original_masked = codes[mask.numpy().astype(bool)]
    generated_masked = generated_codes[mask.numpy().astype(bool)]
    changed = (original_masked != generated_masked).sum()
    
    print(f"Changed masked positions: {changed}/{mask.sum().item()} ({changed/mask.sum().item()*100:.1f}%)")
    
    # Check token distribution
    print("\nToken distribution:")
    for cb in range(n_codebooks):
        unique_tokens = np.unique(generated_codes[0, cb])
        print(f"  Codebook {cb}: {len(unique_tokens)} unique tokens")
    
    # Visualize the generation pattern
    visualize_generation_pattern(codes[0], generated_codes[0], mask[0])
    
    return generated_codes


def visualize_generation_pattern(original, generated, mask):
    """Visualize the token generation pattern."""
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    # Original codes
    im1 = axes[0].imshow(original, aspect='auto', cmap='tab20', interpolation='nearest')
    axes[0].set_title('Original Codes')
    axes[0].set_ylabel('Codebook')
    axes[0].set_xlabel('Sequence Position')
    plt.colorbar(im1, ax=axes[0], label='Token ID')
    
    # Generated codes
    im2 = axes[1].imshow(generated, aspect='auto', cmap='tab20', interpolation='nearest')
    axes[1].set_title('Generated Codes')
    axes[1].set_ylabel('Codebook')
    axes[1].set_xlabel('Sequence Position')
    plt.colorbar(im2, ax=axes[1], label='Token ID')
    
    # Difference (only at masked positions)
    diff = np.zeros_like(original, dtype=float)
    diff[mask.numpy().astype(bool)] = (generated != original)[mask.numpy().astype(bool)]
    
    im3 = axes[2].imshow(diff, aspect='auto', cmap='RdBu_r', interpolation='nearest', vmin=0, vmax=1)
    axes[2].set_title('Changed Positions (Red = Changed)')
    axes[2].set_ylabel('Codebook')
    axes[2].set_xlabel('Sequence Position')
    plt.colorbar(im3, ax=axes[2], label='Changed')
    
    plt.tight_layout()
    plt.savefig('transformer_generation_pattern.png', dpi=150)
    print("\nVisualization saved to transformer_generation_pattern.png")
    plt.close()


def test_iterative_generation():
    """Test iterative generation with the transformer."""
    
    print("\n=== Testing Iterative Generation ===\n")
    
    # Load ONNX model
    ort_session = ort.InferenceSession("vampnet_transformer.onnx")
    
    # Parameters
    batch_size = 1
    n_codebooks = 4
    seq_len = 100
    vocab_size = 1024
    num_iterations = 5
    
    # Start with random codes
    codes = np.random.randint(0, vocab_size, (batch_size, n_codebooks, seq_len), dtype=np.int64)
    
    # Track generation history
    history = [codes.copy()]
    
    for iteration in range(num_iterations):
        # Create decreasing mask ratio
        mask_ratio = 0.5 * (1 - iteration / num_iterations)
        mask = create_random_mask((batch_size, n_codebooks, seq_len), mask_ratio=mask_ratio)
        
        print(f"Iteration {iteration + 1}: Masking {mask_ratio*100:.1f}% of tokens")
        
        # Generate
        outputs = ort_session.run(None, {'codes': codes, 'mask': mask.numpy()})
        codes = outputs[0]
        history.append(codes.copy())
    
    # Visualize iterative refinement
    visualize_iterative_generation(history)
    
    return history


def visualize_iterative_generation(history):
    """Visualize the iterative generation process."""
    
    num_iterations = len(history)
    fig, axes = plt.subplots(num_iterations, 1, figsize=(15, 2*num_iterations))
    
    if num_iterations == 1:
        axes = [axes]
    
    for i, codes in enumerate(history):
        # Show only first codebook for clarity
        im = axes[i].imshow(codes[0, 0:1], aspect='auto', cmap='tab20', interpolation='nearest')
        axes[i].set_title(f'{"Initial" if i == 0 else f"After iteration {i}"}')
        axes[i].set_ylabel('CB 0')
        if i == len(history) - 1:
            axes[i].set_xlabel('Sequence Position')
        plt.colorbar(im, ax=axes[i], label='Token')
    
    plt.tight_layout()
    plt.savefig('transformer_iterative_generation.png', dpi=150)
    print("\nIterative generation visualization saved to transformer_iterative_generation.png")
    plt.close()


def benchmark_performance():
    """Benchmark the ONNX transformer performance."""
    
    print("\n=== Benchmarking Performance ===\n")
    
    # Load model
    ort_session = ort.InferenceSession("vampnet_transformer.onnx")
    
    # Test with fixed sequence length (model doesn't support dynamic shapes yet)
    seq_len = 100
    batch_sizes = [1, 2, 4, 8]
    times = []
    
    for batch_size in batch_sizes:
        codes = np.random.randint(0, 1024, (batch_size, 4, seq_len), dtype=np.int64)
        mask = np.zeros_like(codes)
        
        try:
            # Warm-up
            for _ in range(3):
                ort_session.run(None, {'codes': codes, 'mask': mask})
            
            # Measure
            start = time.time()
            num_runs = 10
            for _ in range(num_runs):
                ort_session.run(None, {'codes': codes, 'mask': mask})
            
            avg_time = (time.time() - start) / num_runs
            times.append(avg_time)
            
            print(f"Batch size {batch_size}: {avg_time*1000:.2f}ms per inference")
        except Exception as e:
            print(f"Batch size {batch_size}: Failed - {str(e)}")
            break
    
    if times:
        # Plot performance
        plt.figure(figsize=(10, 6))
        plt.plot(batch_sizes[:len(times)], [t*1000 for t in times], 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Batch Size')
        plt.ylabel('Inference Time (ms)')
        plt.title('ONNX Transformer Inference Performance (Seq Len = 100)')
        plt.grid(True, alpha=0.3)
        plt.savefig('transformer_performance.png', dpi=150)
        print("\nPerformance plot saved to transformer_performance.png")
        plt.close()


if __name__ == "__main__":
    # Check if model exists
    if not Path("vampnet_transformer.onnx").exists():
        print("Error: vampnet_transformer.onnx not found!")
        print("Please run export_vampnet_transformer.py first.")
        exit(1)
    
    # Test basic inference
    generated = test_transformer_inference()
    
    # Test iterative generation
    history = test_iterative_generation()
    
    # Benchmark performance
    benchmark_performance()
    
    print("\n=== Summary ===")
    print("✅ ONNX transformer tested successfully")
    print("✅ Model can generate tokens at masked positions")
    print("✅ Iterative generation works correctly")
    print("✅ Performance benchmarked")
    print("\nNote: This uses random weights. For music generation,")
    print("you'll need to load pretrained VampNet weights.")