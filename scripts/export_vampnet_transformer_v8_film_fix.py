#!/usr/bin/env python3
"""
Export VampNet transformer V8 - V7 + correct FiLM handling for input_dim=0.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.export_vampnet_transformer_v6_proper import export_model_v6

if __name__ == "__main__":
    # Export with correct codec embeddings and FiLM fix
    codec_path = "models/vampnet/codec.pth"
    
    print("\n" + "="*80)
    print("EXPORTING V8 MODELS WITH FILM FIX")
    print("="*80)
    
    # Export coarse
    export_model_v6(
        "coarse", 
        output_path="onnx_models_fixed/coarse_v8_film_fix.onnx",
        codec_path=codec_path
    )
    
    # Export c2f
    export_model_v6(
        "c2f",
        output_path="onnx_models_fixed/c2f_v8_film_fix.onnx", 
        codec_path=codec_path
    )
    
    print("\n" + "="*80)
    print("V8 MODELS EXPORTED WITH:")
    print("- NO positional encoding (uses relative position bias)")
    print("- Relative attention in first layer only")
    print("- Correct codec embeddings from DAC model")
    print("- GatedGELU activation with custom GELU")
    print("- FiLM fix for input_dim=0 (identity transform)")
    print("="*80)