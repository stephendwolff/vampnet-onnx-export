#!/usr/bin/env python3
"""Identify essential scripts to keep from the scripts folder."""

import os
from pathlib import Path

# Essential scripts based on our work
ESSENTIAL_SCRIPTS = [
    # Weight transfer
    "transfer_weights_complete_v3.py",  # Complete weight transfer script
    "export_complete_model_to_onnx.py", # Export script for ONNX
    
    # Working encoders/decoders
    "export_working_encoder.py",         # Pre-padded encoder export
    "vampnet_encoder_prepadded.onnx",   # If in scripts/models/
    
    # Pipeline
    "vampnet_full_pipeline_fixed.py",   # Working pipeline
    
    # Model definitions
    "export_vampnet_transformer_v2.py", # Transformer definition
]

# Essential model files
ESSENTIAL_MODELS = [
    "models/vampnet_encoder_prepadded.onnx",
    "models/vampnet_codec_decoder.onnx",
    "models/vampnet_codec_encoder.onnx",
]

print("Essential scripts to keep:\n")
scripts_dir = Path("scripts")

if scripts_dir.exists():
    for script in ESSENTIAL_SCRIPTS:
        script_path = scripts_dir / script
        if script_path.exists():
            print(f"✓ {script}")
        else:
            print(f"✗ {script} (not found)")
    
    print("\nEssential model files:")
    for model in ESSENTIAL_MODELS:
        model_path = scripts_dir / model
        if model_path.exists():
            print(f"✓ {model}")
        else:
            print(f"✗ {model} (not found)")

print("\n\nSuggested minimal scripts to create:")
print("1. export_to_onnx.py - Simple script to export VampNet models to ONNX")
print("2. generate_audio.py - Simple audio generation using ONNX models")
print("3. compare_models.py - Compare VampNet vs ONNX output")