import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import os

def test_quantization(input_model_path, output_model_path):
    """Test dynamic quantization on the fixed model"""
    
    print(f"=== Testing Quantization ===")
    print(f"Input model: {input_model_path}")
    print(f"Output model: {output_model_path}")
    
    # Get original model size
    original_size = os.path.getsize(input_model_path) / (1024 * 1024)  # MB
    print(f"Original model size: {original_size:.2f} MB")
    
    try:
        # Perform dynamic quantization
        print("\nPerforming dynamic quantization...")
        quantize_dynamic(
            model_input=input_model_path,
            model_output=output_model_path,
            per_channel=True,
            reduce_range=True,
            weight_type=QuantType.QInt8
        )
        
        # Get quantized model size
        quantized_size = os.path.getsize(output_model_path) / (1024 * 1024)  # MB
        print(f"\nQuantized model size: {quantized_size:.2f} MB")
        print(f"Size reduction: {(1 - quantized_size/original_size) * 100:.1f}%")
        
        # Verify the quantized model
        print("\n=== Verifying Quantized Model ===")
        quantized_model = onnx.load(output_model_path)
        
        # Count quantized operations
        op_counts = {}
        for node in quantized_model.graph.node:
            op_type = node.op_type
            if op_type not in op_counts:
                op_counts[op_type] = 0
            op_counts[op_type] += 1
        
        # Show quantization-related operations
        quant_ops = ['QuantizeLinear', 'DequantizeLinear', 'QLinearConv', 'QLinearMatMul']
        print("\nQuantization operations found:")
        for op in quant_ops:
            if op in op_counts:
                print(f"  {op}: {op_counts[op]}")
        
        return True
        
    except Exception as e:
        print(f"\nQuantization failed: {e}")
        return False

if __name__ == "__main__":
    # Test the original model (should fail)
    print("=== Testing Original Model ===")
    success = test_quantization(
        "onnx_models/codec_encoder.onnx",
        "onnx_models/codec_encoder_quantized_test.onnx"
    )
    
    print("\n" + "="*50 + "\n")
    
    # Test the fixed model (should succeed)
    print("=== Testing Fixed Model ===")
    success = test_quantization(
        "onnx_models/codec_encoder_fixed.onnx",
        "onnx_models/codec_encoder_quantized_fixed.onnx"
    )
    
    if success:
        print("\n✓ Quantization successful with the fixed model!")
    else:
        print("\n✗ Quantization still failing, further investigation needed")