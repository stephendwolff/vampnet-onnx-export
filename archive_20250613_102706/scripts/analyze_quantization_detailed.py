import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import os
from collections import defaultdict

def analyze_model_operations(model_path, title):
    """Analyze operations in the model"""
    print(f"\n=== {title} ===")
    
    model = onnx.load(model_path)
    graph = model.graph
    
    # Count all operations
    op_counts = defaultdict(int)
    for node in graph.node:
        op_counts[node.op_type] += 1
    
    # Show all operations
    print("\nAll operations:")
    for op_type, count in sorted(op_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {op_type}: {count}")
    
    # Check data types in initializers
    print("\nInitializer data types:")
    type_counts = defaultdict(int)
    total_params = 0
    for init in graph.initializer:
        type_counts[onnx.TensorProto.DataType.Name(init.data_type)] += 1
        # Count parameters
        size = 1
        for dim in init.dims:
            size *= dim
        total_params += size
    
    for dtype, count in type_counts.items():
        print(f"  {dtype}: {count} initializers")
    
    print(f"\nTotal parameters: {total_params:,}")
    
    # Check which operations might be quantizable
    quantizable_ops = ['Conv', 'MatMul', 'Gemm', 'Linear']
    print("\nPotentially quantizable operations:")
    for op in quantizable_ops:
        if op in op_counts:
            print(f"  {op}: {op_counts[op]}")
    
    return op_counts

def test_different_quantization_modes(input_path, base_output_path):
    """Test different quantization configurations"""
    
    print(f"\n=== Testing Different Quantization Modes ===")
    original_size = os.path.getsize(input_path) / (1024 * 1024)
    print(f"Original model size: {original_size:.2f} MB")
    
    configs = [
        {
            "name": "dynamic_default",
            "params": {
                "weight_type": QuantType.QInt8
            }
        },
        {
            "name": "dynamic_per_channel",
            "params": {
                "weight_type": QuantType.QInt8,
                "per_channel": True
            }
        },
        {
            "name": "dynamic_reduce_range",
            "params": {
                "weight_type": QuantType.QInt8,
                "reduce_range": True
            }
        },
        {
            "name": "dynamic_all_options",
            "params": {
                "weight_type": QuantType.QInt8,
                "per_channel": True,
                "reduce_range": True,
                "optimize_model": True
            }
        }
    ]
    
    for config in configs:
        output_path = base_output_path.replace(".onnx", f"_{config['name']}.onnx")
        print(f"\n--- Testing {config['name']} ---")
        
        try:
            quantize_dynamic(
                model_input=input_path,
                model_output=output_path,
                **config['params']
            )
            
            # Check size
            quantized_size = os.path.getsize(output_path) / (1024 * 1024)
            reduction = (1 - quantized_size/original_size) * 100
            print(f"Size: {quantized_size:.2f} MB (reduction: {reduction:.1f}%)")
            
            # Analyze the quantized model
            quantized_model = onnx.load(output_path)
            
            # Check if weights were actually quantized
            weight_types = defaultdict(int)
            for init in quantized_model.graph.initializer:
                weight_types[onnx.TensorProto.DataType.Name(init.data_type)] += 1
            
            print("Weight types in quantized model:")
            for dtype, count in weight_types.items():
                print(f"  {dtype}: {count}")
                
        except Exception as e:
            print(f"Failed: {e}")

def check_model_structure(model_path):
    """Check model structure for quantization compatibility"""
    
    print(f"\n=== Checking Model Structure for Quantization ===")
    
    model = onnx.load(model_path)
    graph = model.graph
    
    # Check Conv nodes in detail
    conv_count = 0
    for node in graph.node:
        if node.op_type == 'Conv':
            conv_count += 1
            print(f"\nConv node {conv_count}: {node.name}")
            
            # Check if it has weight initializer
            if len(node.input) >= 2:
                weight_name = node.input[1]
                weight_found = False
                
                for init in graph.initializer:
                    if init.name == weight_name:
                        weight_found = True
                        print(f"  Weight: {weight_name}")
                        print(f"  Weight type: {onnx.TensorProto.DataType.Name(init.data_type)}")
                        print(f"  Weight shape: {list(init.dims)}")
                        break
                
                if not weight_found:
                    print(f"  Warning: Weight '{weight_name}' not found in initializers!")
            
            # Check attributes
            for attr in node.attribute:
                if attr.name in ['kernel_shape', 'strides', 'pads']:
                    print(f"  {attr.name}: {attr.ints}")

if __name__ == "__main__":
    # Analyze original model
    analyze_model_operations("onnx_models/codec_encoder.onnx", "Original Model Analysis")
    
    # Analyze fixed model
    analyze_model_operations("onnx_models/codec_encoder_fixed.onnx", "Fixed Model Analysis")
    
    # Analyze quantized model
    if os.path.exists("onnx_models/codec_encoder_quantized_fixed.onnx"):
        analyze_model_operations("onnx_models/codec_encoder_quantized_fixed.onnx", 
                               "Quantized Model Analysis")
    
    # Test different quantization modes
    test_different_quantization_modes(
        "onnx_models/codec_encoder_fixed.onnx",
        "onnx_models/codec_encoder_quantized.onnx"
    )
    
    # Check model structure
    check_model_structure("onnx_models/codec_encoder_fixed.onnx")