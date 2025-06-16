#!/usr/bin/env python3
"""
Export VampNet transformer V7 - same as V6 but with correct codec embeddings.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.export_vampnet_transformer_v6_proper import export_model_v6

if __name__ == "__main__":
    # Export with correct codec embeddings
    codec_path = "../models/vampnet/codec.pth"
    
    print("\n" + "="*80)
    print("EXPORTING V7 MODELS WITH CORRECT CODEC EMBEDDINGS")
    print("="*80)
    
    # Export coarse
    export_model_v6(
        "coarse", 
        output_path="../onnx_models_fixed/coarse_v7_codec_fix.onnx",
        codec_path=codec_path
    )
    
    # Export c2f
    export_model_v6(
        "c2f",
        output_path="../onnx_models_fixed/c2f_v7_codec_fix.onnx", 
        codec_path=codec_path
    )
    
    print("\n" + "="*80)
    print("V7 MODELS EXPORTED WITH:")
    print("- NO positional encoding (uses relative position bias)")
    print("- Relative attention in first layer only")
    print("- Correct codec embeddings from DAC model")
    print("- GatedGELU activation")
    print("="*80)