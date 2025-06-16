#!/usr/bin/env python3
"""
Example of using the proper VampNet mask generator.
Shows both numpy and ONNX usage.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from vampnet_onnx import VampNetMaskGenerator, Interface

def visualize_mask(mask, title="Mask Visualization"):
    """Visualize a mask as a heatmap."""
    plt.figure(figsize=(12, 4))
    plt.imshow(mask[0], aspect='auto', cmap='RdBu', interpolation='nearest')
    plt.colorbar(label='Masked (1) / Preserved (0)')
    plt.xlabel('Time Step')
    plt.ylabel('Codebook')
    plt.title(title)
    plt.tight_layout()
    return plt.gcf()

def main():
    print("=" * 80)
    print("PROPER VAMPNET MASK GENERATOR EXAMPLES")
    print("=" * 80)
    
    # Create test tokens
    np.random.seed(42)
    batch_size = 1
    n_codebooks = 14
    seq_len = 100
    tokens = np.random.randint(0, 1024, size=(batch_size, n_codebooks, seq_len), dtype=np.int64)
    
    print(f"\nTest tokens shape: {tokens.shape}")
    
    # Initialize mask generator
    mask_gen = VampNetMaskGenerator()
    
    # Example 1: Basic random masking
    print("\n" + "-" * 60)
    print("Example 1: Basic Random Masking")
    print("-" * 60)
    
    mask = mask_gen.build_mask(
        tokens,
        rand_mask_intensity=0.8,
        periodic_prompt=0,  # Disable periodic
        upper_codebook_mask=0,  # No codebook masking
        seed=42
    )
    
    print(f"Mask shape: {mask.shape}")
    print(f"Mask ratio: {np.mean(mask):.2%}")
    print(f"Masked positions: {np.sum(mask)}")
    
    fig = visualize_mask(mask, "Example 1: Random Masking (80% intensity)")
    plt.savefig("mask_example_1_random.png")
    plt.close()
    
    # Example 2: Periodic prompting
    print("\n" + "-" * 60)
    print("Example 2: Periodic Prompting")
    print("-" * 60)
    
    mask = mask_gen.build_mask(
        tokens,
        rand_mask_intensity=0.0,  # No random masking
        periodic_prompt=7,
        periodic_prompt_width=2,
        upper_codebook_mask=0,
        seed=42
    )
    
    print(f"Mask ratio: {np.mean(mask):.2%}")
    print(f"Preserved columns: {np.sum(np.all(mask == 0, axis=(0, 1)))}")
    
    fig = visualize_mask(mask, "Example 2: Periodic Prompting (every 7 steps, width 2)")
    plt.savefig("mask_example_2_periodic.png")
    plt.close()
    
    # Example 3: Combined masking with codebook hierarchy
    print("\n" + "-" * 60)
    print("Example 3: Combined Masking with Codebook Hierarchy")
    print("-" * 60)
    
    mask = mask_gen.build_mask(
        tokens,
        rand_mask_intensity=0.7,
        periodic_prompt=10,
        periodic_prompt_width=1,
        upper_codebook_mask=4,  # Mask codebooks 4 and above
        seed=42
    )
    
    print(f"Overall mask ratio: {np.mean(mask):.2%}")
    print(f"Lower codebooks (0-3) mask ratio: {np.mean(mask[:, :4, :]):.2%}")
    print(f"Upper codebooks (4-13) mask ratio: {np.mean(mask[:, 4:, :]):.2%}")
    
    fig = visualize_mask(mask, "Example 3: Combined (random + periodic + codebook hierarchy)")
    plt.savefig("mask_example_3_combined.png")
    plt.close()
    
    # Example 4: Inpainting (preserve prefix/suffix)
    print("\n" + "-" * 60)
    print("Example 4: Inpainting (Preserve Prefix/Suffix)")
    print("-" * 60)
    
    mask = mask_gen.build_mask(
        tokens,
        rand_mask_intensity=1.0,
        prefix_s=0.5,  # Preserve first 0.5 seconds
        suffix_s=0.5,  # Preserve last 0.5 seconds
        periodic_prompt=0,
        upper_codebook_mask=3,
        seed=42
    )
    
    # Calculate preserved tokens
    sample_rate = 44100
    hop_length = 768
    prefix_tokens = int(np.ceil(0.5 * sample_rate / hop_length))
    suffix_tokens = int(np.ceil(0.5 * sample_rate / hop_length))
    
    print(f"Prefix tokens preserved: {prefix_tokens}")
    print(f"Suffix tokens preserved: {suffix_tokens}")
    print(f"Overall mask ratio: {np.mean(mask):.2%}")
    print(f"Middle region mask ratio: {np.mean(mask[:, :, prefix_tokens:-suffix_tokens]):.2%}")
    
    fig = visualize_mask(mask, "Example 4: Inpainting (preserve 0.5s prefix/suffix)")
    plt.savefig("mask_example_4_inpainting.png")
    plt.close()
    
    # Example 5: Using with Interface
    print("\n" + "-" * 60)
    print("Example 5: Using with ONNX Interface")
    print("-" * 60)
    
    # Initialize interface (without models for this example)
    interface = Interface(device='cpu')
    
    # Build mask using interface
    mask = interface.build_mask(
        tokens,
        rand_mask_intensity=0.8,
        periodic_prompt=7,
        upper_codebook_mask=3
    )
    
    print(f"Interface mask shape: {mask.shape}")
    print(f"Interface mask ratio: {np.mean(mask):.2%}")
    
    # Example 6: Apply mask to tokens
    print("\n" + "-" * 60)
    print("Example 6: Applying Mask to Tokens")
    print("-" * 60)
    
    mask = mask_gen.build_mask(
        tokens,
        rand_mask_intensity=0.5,
        periodic_prompt=5,
        upper_codebook_mask=2,
        seed=42
    )
    
    masked_tokens, _ = mask_gen.apply_mask(tokens, mask)
    
    n_original = tokens.size
    n_masked = np.sum(masked_tokens == 1024)
    
    print(f"Original tokens: {n_original}")
    print(f"Masked tokens: {n_masked} ({n_masked/n_original:.1%})")
    print(f"Preserved tokens: {n_original - n_masked} ({(n_original - n_masked)/n_original:.1%})")
    
    # Visualize original vs masked
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Original tokens (show first 3 codebooks)
    im1 = ax1.imshow(tokens[0, :3, :], aspect='auto', cmap='viridis', interpolation='nearest')
    ax1.set_title("Original Tokens (first 3 codebooks)")
    ax1.set_ylabel("Codebook")
    plt.colorbar(im1, ax=ax1, label="Token ID")
    
    # Masked tokens
    masked_display = masked_tokens[0, :3, :].astype(float)
    masked_display[masked_display == 1024] = np.nan  # Show mask tokens as NaN
    im2 = ax2.imshow(masked_display, aspect='auto', cmap='viridis', interpolation='nearest')
    ax2.set_title("Masked Tokens (mask token = white)")
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Codebook")
    plt.colorbar(im2, ax=ax2, label="Token ID")
    
    plt.tight_layout()
    plt.savefig("mask_example_6_applied.png")
    plt.close()
    
    print("\n" + "=" * 80)
    print("EXAMPLES COMPLETE")
    print("=" * 80)
    print("\nGenerated visualizations:")
    print("- mask_example_1_random.png")
    print("- mask_example_2_periodic.png")
    print("- mask_example_3_combined.png")
    print("- mask_example_4_inpainting.png")
    print("- mask_example_6_applied.png")


if __name__ == "__main__":
    main()