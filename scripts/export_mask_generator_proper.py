#!/usr/bin/env python3
"""
Export the proper VampNet mask generator to ONNX format.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.onnx
import numpy as np
import onnxruntime as ort
from vampnet_onnx.mask_generator_proper import VampNetMaskGeneratorTorch

def export_mask_generator(output_path: str = "models/vampnet_mask_generator_proper.onnx"):
    """Export the mask generator to ONNX format."""
    print("=" * 80)
    print("EXPORTING PROPER VAMPNET MASK GENERATOR TO ONNX")
    print("=" * 80)
    
    # Create model
    model = VampNetMaskGeneratorTorch()
    model.eval()
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Example inputs
    batch_size = 1
    n_codebooks = 14
    seq_len = 100
    
    dummy_z = torch.randint(0, 1024, (batch_size, n_codebooks, seq_len), dtype=torch.long)
    
    print(f"\nExporting model with input shape: {dummy_z.shape}")
    
    # Export with dynamic axes
    torch.onnx.export(
        model,
        (dummy_z, 1.0, 7, 1, 3, 0, 0, 42),
        output_path,
        input_names=[
            "z", "rand_mask_intensity", "periodic_prompt", 
            "periodic_prompt_width", "upper_codebook_mask",
            "prefix_tokens", "suffix_tokens", "random_seed"
        ],
        output_names=["mask", "masked_z"],
        dynamic_axes={
            "z": {0: "batch", 2: "seq_len"},
            "mask": {0: "batch", 2: "seq_len"},
            "masked_z": {0: "batch", 2: "seq_len"}
        },
        opset_version=11,
        do_constant_folding=True,
        verbose=False
    )
    
    print(f"✓ Exported mask generator to {output_path}")
    
    # Verify the exported model
    print("\n" + "-" * 60)
    print("VERIFYING EXPORTED MODEL")
    print("-" * 60)
    
    # Create ONNX session
    session = ort.InferenceSession(output_path)
    
    # Test with various configurations
    test_configs = [
        {"rand_mask_intensity": 1.0, "periodic_prompt": 0, "upper_codebook_mask": 0},
        {"rand_mask_intensity": 0.8, "periodic_prompt": 7, "upper_codebook_mask": 3},
        {"rand_mask_intensity": 0.5, "periodic_prompt": 10, "upper_codebook_mask": 4},
    ]
    
    for i, config in enumerate(test_configs):
        print(f"\nTest {i+1}: {config}")
        
        # Prepare inputs
        inputs = {
            "z": dummy_z.numpy(),
            "rand_mask_intensity": np.float32(config["rand_mask_intensity"]),
            "periodic_prompt": np.int64(config["periodic_prompt"]),
            "periodic_prompt_width": np.int64(1),
            "upper_codebook_mask": np.int64(config["upper_codebook_mask"]),
            "prefix_tokens": np.int64(0),
            "suffix_tokens": np.int64(0),
            "random_seed": np.int64(42)
        }
        
        # Run inference
        mask, masked_z = session.run(None, inputs)
        
        # Check results
        mask_ratio = np.mean(mask)
        n_masked = np.sum(masked_z == 1024)
        
        print(f"  Mask shape: {mask.shape}")
        print(f"  Mask ratio: {mask_ratio:.2%}")
        print(f"  Masked tokens: {n_masked}/{mask.size}")
        
        # Verify mask properties
        if config["upper_codebook_mask"] > 0:
            upper_mask_ratio = np.mean(mask[:, config["upper_codebook_mask"]:, :])
            print(f"  Upper codebook mask ratio: {upper_mask_ratio:.2%}")
    
    # Test with different sequence lengths
    print("\n" + "-" * 60)
    print("TESTING DYNAMIC SEQUENCE LENGTHS")
    print("-" * 60)
    
    for seq_len in [50, 100, 200]:
        test_z = torch.randint(0, 1024, (1, 14, seq_len), dtype=torch.long)
        
        inputs = {
            "z": test_z.numpy(),
            "rand_mask_intensity": np.float32(0.8),
            "periodic_prompt": np.int64(7),
            "periodic_prompt_width": np.int64(1),
            "upper_codebook_mask": np.int64(3),
            "prefix_tokens": np.int64(0),
            "suffix_tokens": np.int64(0),
            "random_seed": np.int64(42)
        }
        
        try:
            mask, masked_z = session.run(None, inputs)
            print(f"✓ Sequence length {seq_len}: Success (output shape: {mask.shape})")
        except Exception as e:
            print(f"✗ Sequence length {seq_len}: Failed - {e}")
    
    print("\n" + "=" * 80)
    print("EXPORT COMPLETE")
    print("=" * 80)
    
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export VampNet mask generator to ONNX")
    parser.add_argument("--output", "-o", default="models/vampnet_mask_generator_proper.onnx",
                        help="Output ONNX file path")
    args = parser.parse_args()
    
    export_mask_generator(args.output)