"""
Test ONNX export of the VampNet codec.
"""

import torch
import numpy as np
from pathlib import Path
import os

# Import vampnet components
import vampnet
from vampnet_onnx.exporters import export_codec_encoder, export_codec_decoder


def test_onnx_export():
    """Test exporting VampNet codec to ONNX."""
    print("=== Testing ONNX Export of VampNet Codec ===")
    
    # Create output directory
    output_dir = Path("onnx_models/vampnet_codec")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the codec
    interface = vampnet.interface.Interface(
        codec_ckpt="models/vampnet/codec.pth",
        coarse_ckpt="models/vampnet/coarse.pth"
    )
    codec = interface.codec
    
    print(f"Codec info:")
    print(f"  Sample rate: {codec.sample_rate}")
    print(f"  Hop length: {codec.hop_length}")
    print(f"  Num codebooks: {codec.n_codebooks}")
    
    # Try to export encoder
    print("\n--- Exporting Encoder ---")
    try:
        export_codec_encoder(
            output_path=str(output_dir / "encoder.onnx"),
            n_codebooks=codec.n_codebooks,
            vocab_size=1024,  # Standard vocab size for LAC
            sample_rate=codec.sample_rate,
            hop_length=codec.hop_length,
            example_batch_size=1,
            example_audio_length=codec.sample_rate * 2,  # 2 seconds
            use_simplified=False,
            use_vampnet=True,
            codec_model=codec,
            device='cpu',
            opset_version=14
        )
        print("✅ Encoder export successful!")
    except Exception as e:
        print(f"❌ Encoder export failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Try to export decoder
    print("\n--- Exporting Decoder ---")
    try:
        export_codec_decoder(
            output_path=str(output_dir / "decoder.onnx"),
            n_codebooks=codec.n_codebooks,
            vocab_size=1024,  # Standard vocab size for LAC
            sample_rate=codec.sample_rate,
            hop_length=codec.hop_length,
            example_batch_size=1,
            example_sequence_length=100,
            use_simplified=False,
            use_vampnet=True,
            codec_model=codec,
            device='cpu',
            opset_version=14
        )
        print("✅ Decoder export successful!")
    except Exception as e:
        print(f"❌ Decoder export failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\nONNX models saved to {output_dir}")
    
    # Check what operators are not supported if export failed
    if os.path.exists(output_dir / "encoder.onnx"):
        print("\n--- Encoder ONNX Model Info ---")
        import onnx
        model = onnx.load(str(output_dir / "encoder.onnx"))
        print(f"Input names: {[inp.name for inp in model.graph.input]}")
        print(f"Output names: {[out.name for out in model.graph.output]}")
        print(f"Number of nodes: {len(model.graph.node)}")
        
        # Check for unsupported ops
        op_types = set(node.op_type for node in model.graph.node)
        print(f"Unique operators: {sorted(op_types)}")


if __name__ == "__main__":
    test_onnx_export()