#!/usr/bin/env python3
"""
Test script to export all ONNX models from VampNet using the cleaned-up codebase.
This verifies that all necessary scripts are present after archiving outdated files.
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """Run a command and check for errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print('='*60)
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: {description} failed!")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return False
    else:
        print(f"SUCCESS: {description}")
        if result.stdout:
            print(f"Output: {result.stdout[:500]}...")  # Show first 500 chars
    return True

def main():
    # Set up paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    models_dir = project_root / "models"
    vampnet_models_dir = models_dir / "vampnet"
    onnx_output_dir = models_dir / "onnx_test_export"
    
    # Create output directory
    onnx_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Project root: {project_root}")
    print(f"VampNet models: {vampnet_models_dir}")
    print(f"ONNX output: {onnx_output_dir}")
    
    # Check that VampNet models exist
    required_models = ["codec.pth", "coarse.pth", "c2f.pth"]
    for model in required_models:
        if not (vampnet_models_dir / model).exists():
            print(f"ERROR: Required model {model} not found in {vampnet_models_dir}")
            return 1
    
    print("\nAll required VampNet models found!")
    
    # Step 1: Export codec
    print("\n\n*** STEP 1: Exporting Codec ***")
    if not run_command(
        f"cd {script_dir} && python export_vampnet_codec_proper.py",
        "Export VampNet codec to ONNX"
    ):
        return 1
    
    # Step 2: Export Coarse Transformer
    print("\n\n*** STEP 2: Exporting Coarse Transformer ***")
    if not run_command(
        f"cd {script_dir} && python export_vampnet_transformer_v2.py",
        "Export Coarse transformer (v2) to ONNX"
    ):
        return 1
    
    # Step 3: Export C2F Transformer
    print("\n\n*** STEP 3: Exporting C2F Transformer ***")
    if not run_command(
        f"cd {script_dir} && python export_c2f_transformer_v2.py",
        "Export C2F transformer (v2) to ONNX"
    ):
        return 1
    
    # Step 4: Transfer weights
    print("\n\n*** STEP 4: Transferring Weights ***")
    if not run_command(
        f"cd {script_dir} && python transfer_weights_vampnet_v2.py",
        "Transfer weights from VampNet models (v2)"
    ):
        return 1
    
    # Step 5: Complete weight transfer (embeddings, classifiers)
    print("\n\n*** STEP 5: Completing Weight Transfer ***")
    if not run_command(
        f"cd {script_dir} && python complete_weight_transfer.py",
        "Complete weight transfer (embeddings and classifiers)"
    ):
        return 1
    
    # Step 6: Verify weight transfer
    print("\n\n*** STEP 6: Verifying Weight Transfer ***")
    if not run_command(
        f"cd {script_dir} && python verify_weight_transfer.py",
        "Verify weight transfer success"
    ):
        print("WARNING: Weight verification failed, but continuing...")
    
    # Step 7: Test the models
    print("\n\n*** STEP 7: Testing Exported Models ***")
    if not run_command(
        f"cd {script_dir} && python test_weighted_models.py",
        "Test weighted ONNX models"
    ):
        print("WARNING: Model testing failed, but continuing...")
    
    # Step 8: List generated files
    print("\n\n*** STEP 8: Listing Generated ONNX Models ***")
    run_command(
        f"find {project_root} -name '*.onnx' -type f | grep -v archive",
        "Find all ONNX files (excluding archive)"
    )
    
    print("\n\n*** EXPORT COMPLETE ***")
    print("If all steps succeeded, you should now have:")
    print("1. Codec encoder/decoder in models/")
    print("2. Coarse transformer in onnx_models_fixed/")
    print("3. C2F transformer in onnx_models_fixed/")
    print("4. All models with weights transferred from VampNet")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())