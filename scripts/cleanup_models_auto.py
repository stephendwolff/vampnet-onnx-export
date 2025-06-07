#!/usr/bin/env python3
"""
Cleanup script to remove intermediate ONNX models and other large files.
Auto mode - no interaction required.
"""

import os
from pathlib import Path
import shutil


def get_file_size_mb(file_path):
    """Get file size in MB."""
    return os.path.getsize(file_path) / (1024 * 1024)


def cleanup_intermediate_models():
    """Remove intermediate transformer models, keeping only the final version."""
    
    print("=== Cleaning Up Intermediate Models ===\n")
    
    # Models to remove (intermediate versions)
    transformer_models_to_remove = [
        "scripts/vampnet_transformer.onnx",  # Random weights
        "scripts/vampnet_transformer_improved.onnx",  # Partial weights
        "scripts/vampnet_transformer_with_weights.onnx",  # Early attempt
        "scripts/vampnet_transformer_complete.onnx",  # Before final fixes
    ]
    
    # Model to keep
    final_model = "scripts/vampnet_transformer_final.onnx"
    
    total_freed = 0
    
    print("Transformer models:")
    for model in transformer_models_to_remove:
        if Path(model).exists():
            size_mb = get_file_size_mb(model)
            print(f"  Removing {model} ({size_mb:.1f} MB)")
            os.remove(model)
            total_freed += size_mb
    
    if Path(final_model).exists():
        size_mb = get_file_size_mb(final_model)
        print(f"  Keeping {final_model} ({size_mb:.1f} MB)")
    
    return total_freed


def cleanup_test_models():
    """Remove test and experimental models."""
    
    print("\n\nTest/Experimental models:")
    
    test_models = [
        "scripts/mini_vampnet.onnx",
        "scripts/model_with_rmsnorm.onnx",
        "scripts/combined_custom_ops.onnx",
        # Custom operator test models
        "scripts/rmsnorm_*.onnx",
        "scripts/film_*.onnx", 
        "scripts/codebook_embedding_*.onnx",
    ]
    
    total_freed = 0
    
    for pattern in test_models:
        for model in Path(".").glob(pattern):
            if model.exists():
                size_mb = get_file_size_mb(model)
                print(f"  Removing {model} ({size_mb:.1f} MB)")
                os.remove(model)
                total_freed += size_mb
    
    return total_freed


def cleanup_duplicate_codecs():
    """Remove duplicate codec models, keeping only the optimized versions."""
    
    print("\n\nCodec models:")
    
    # Keep these codec models
    keep_codecs = [
        "models/codec_encoder.onnx",
        "models/codec_decoder.onnx",
    ]
    
    # Remove duplicates
    remove_codecs = [
        "onnx_models/codec_encoder.onnx",
        "onnx_models/codec_decoder.onnx",
        "onnx_models_test/codec_encoder.onnx",
        "onnx_models_test/codec_decoder.onnx",
        "onnx_models_optimized/codec_encoder.onnx",
        "onnx_models_optimized/codec_decoder.onnx",
        "onnx_models_optimized/codec_encoder_opt.onnx",
        "onnx_models_optimized/codec_decoder_opt.onnx",
        # Large vampire codec models
        "onnx_models*/vampnet_codec/*.onnx",
        "onnx_models*/encoder.onnx",
        "onnx_models*/decoder.onnx",
        "onnx_models*/encoder_opt.onnx",
        "onnx_models*/decoder_opt.onnx",
        # Test models
        "models/test_codec_*_simplified.onnx",
    ]
    
    total_freed = 0
    
    for pattern in remove_codecs:
        for model in Path(".").glob(pattern):
            if model.exists():
                size_mb = get_file_size_mb(model)
                print(f"  Removing {model} ({size_mb:.1f} MB)")
                os.remove(model)
                total_freed += size_mb
    
    print("\n  Keeping:")
    for model in keep_codecs:
        if Path(model).exists():
            size_mb = get_file_size_mb(model)
            print(f"    {model} ({size_mb:.1f} MB)")
    
    return total_freed


def cleanup_test_outputs():
    """Remove test audio files and plots."""
    
    print("\n\nTest outputs:")
    
    test_outputs = [
        "scripts/test_*.wav",
        "scripts/*.png",
        "outputs/**/*.wav",
        "notebooks/onnx_test_outputs/*.wav",
    ]
    
    total_freed = 0
    
    for pattern in test_outputs:
        for file in Path(".").glob(pattern):
            if file.exists():
                size_mb = get_file_size_mb(file)
                print(f"  Removing {file} ({size_mb:.1f} MB)")
                os.remove(file)
                total_freed += size_mb
    
    return total_freed


def cleanup_directories():
    """Remove empty directories."""
    
    print("\n\nEmpty directories:")
    
    dirs_to_check = [
        "onnx_models_test/vampnet_codec",
        "onnx_models/vampnet_codec",
        "onnx_models_test",
        "onnx_models",
        "onnx_models_optimized",
        "onnx_models_quantized",
        "notebooks/onnx_test_outputs",
        "outputs/preprocessing_comparison",
        "outputs/codec_comparison",
        "outputs/onnx_inference_test",
    ]
    
    for dir_path in dirs_to_check:
        if Path(dir_path).exists():
            try:
                # Check if directory is empty or contains only test files
                if Path(dir_path).is_dir():
                    files = list(Path(dir_path).iterdir())
                    if not files:
                        print(f"  Removing empty directory: {dir_path}")
                        os.rmdir(dir_path)
                    elif all(f.name.startswith('.') for f in files):
                        # Only hidden files
                        print(f"  Removing directory with only hidden files: {dir_path}")
                        shutil.rmtree(dir_path)
            except Exception as e:
                print(f"  Could not remove {dir_path}: {e}")


def main():
    """Run cleanup process."""
    
    # Change to project root
    os.chdir(Path(__file__).parent.parent)
    
    print("=== Starting Cleanup ===\n")
    
    # Perform cleanup
    total_freed = 0
    
    total_freed += cleanup_intermediate_models()
    total_freed += cleanup_test_models()
    total_freed += cleanup_duplicate_codecs()
    total_freed += cleanup_test_outputs()
    cleanup_directories()
    
    print(f"\n\n✅ Cleanup complete!")
    print(f"Freed {total_freed:.1f} MB of disk space.")
    
    # Show remaining ONNX models
    print("\n\nRemaining ONNX models:")
    for model in sorted(Path(".").glob("**/*.onnx")):
        size_mb = get_file_size_mb(model)
        print(f"  {model} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()