import onnx
import onnx.helper
import onnx.shape_inference
import numpy as np
from collections import defaultdict

def infer_and_fix_types(model_path, output_path):
    """Fix type inference issues in ONNX model"""
    
    # Load the model
    model = onnx.load(model_path)
    graph = model.graph
    
    print("=== Original Model Info ===")
    print(f"Number of value_info entries: {len(graph.value_info)}")
    
    # First, try shape inference
    print("\n=== Running Shape Inference ===")
    try:
        inferred_model = onnx.shape_inference.infer_shapes(model)
        print("Shape inference completed successfully")
        
        # Check if the problematic tensors now have type info
        new_value_info = {vi.name: vi for vi in inferred_model.graph.value_info}
        
        problem_tensors = ['/encoder/encoder.3/Relu_output_0', '/Reshape_output_0']
        for tensor_name in problem_tensors:
            if tensor_name in new_value_info:
                vi = new_value_info[tensor_name]
                elem_type = vi.type.tensor_type.elem_type
                print(f"Found type for {tensor_name}: {onnx.TensorProto.DataType.Name(elem_type)}")
            else:
                print(f"Still missing type for {tensor_name}")
        
        print(f"New number of value_info entries: {len(inferred_model.graph.value_info)}")
        
        # Save the inferred model
        onnx.save(inferred_model, output_path)
        print(f"\nSaved inferred model to: {output_path}")
        
        return inferred_model
        
    except Exception as e:
        print(f"Shape inference failed: {e}")
        print("Attempting manual type propagation...")
    
    # If shape inference fails, manually propagate types
    print("\n=== Manual Type Propagation ===")
    
    # Build a map of tensor producers
    tensor_producers = {}
    for node in graph.node:
        for output in node.output:
            tensor_producers[output] = node
    
    # Get input types
    input_types = {}
    for input_tensor in graph.input:
        input_types[input_tensor.name] = input_tensor.type.tensor_type.elem_type
    
    # Get initializer types
    for init in graph.initializer:
        input_types[init.name] = init.data_type
    
    # Get existing value_info types
    for vi in graph.value_info:
        input_types[vi.name] = vi.type.tensor_type.elem_type
    
    # Function to propagate type through operations
    def get_output_type(node, input_types_dict):
        """Determine output type based on operation and input types"""
        
        # Most operations preserve the input type
        if node.input and node.input[0] in input_types_dict:
            return input_types_dict[node.input[0]]
        
        # For operations with multiple inputs, use the first non-shape input
        for inp in node.input:
            if inp in input_types_dict:
                return input_types_dict[inp]
        
        return onnx.TensorProto.FLOAT  # Default to float
    
    # Find and fix the encoder.3 Relu output
    relu_node = None
    reshape_node = None
    
    for node in graph.node:
        if node.name == '/encoder/encoder.3/Relu':
            relu_node = node
        elif node.name == '/Reshape' and '/Reshape_output_0' in node.output:
            reshape_node = node
    
    new_value_infos = []
    
    if relu_node:
        print(f"\nProcessing Relu node: {relu_node.name}")
        input_tensor = relu_node.input[0]
        
        # Trace back to find the type
        if input_tensor in tensor_producers:
            conv_node = tensor_producers[input_tensor]
            print(f"  Input produced by: {conv_node.name} ({conv_node.op_type})")
            
            # Conv operations typically output the same type as their input
            if conv_node.input[0] in input_types:
                output_type = input_types[conv_node.input[0]]
            else:
                output_type = onnx.TensorProto.FLOAT
        else:
            output_type = onnx.TensorProto.FLOAT
        
        # Create value_info for Relu output
        relu_output_vi = onnx.helper.make_tensor_value_info(
            '/encoder/encoder.3/Relu_output_0',
            output_type,
            None  # Dynamic shape
        )
        new_value_infos.append(relu_output_vi)
        print(f"  Added type info for Relu output: {onnx.TensorProto.DataType.Name(output_type)}")
    
    if reshape_node:
        print(f"\nProcessing Reshape node: {reshape_node.name}")
        input_tensor = reshape_node.input[0]
        
        # Trace back to find the type
        if input_tensor in tensor_producers:
            prev_node = tensor_producers[input_tensor]
            print(f"  Input produced by: {prev_node.name} ({prev_node.op_type})")
            
            # Find the type of the reshape input
            if prev_node.input and prev_node.input[0] in input_types:
                output_type = input_types[prev_node.input[0]]
            else:
                output_type = onnx.TensorProto.FLOAT
        else:
            output_type = onnx.TensorProto.FLOAT
        
        # Create value_info for Reshape output
        reshape_output_vi = onnx.helper.make_tensor_value_info(
            '/Reshape_output_0',
            output_type,
            None  # Dynamic shape
        )
        new_value_infos.append(reshape_output_vi)
        print(f"  Added type info for Reshape output: {onnx.TensorProto.DataType.Name(output_type)}")
    
    # Add new value_infos to the graph
    graph.value_info.extend(new_value_infos)
    
    print(f"\n=== Updated Model ===")
    print(f"Added {len(new_value_infos)} new value_info entries")
    print(f"Total value_info entries: {len(graph.value_info)}")
    
    # Run shape inference on the updated model
    print("\n=== Running Shape Inference on Updated Model ===")
    try:
        final_model = onnx.shape_inference.infer_shapes(model)
        print("Shape inference completed successfully")
        
        # Save the final model
        onnx.save(final_model, output_path)
        print(f"\nSaved fixed model to: {output_path}")
        
        return final_model
        
    except Exception as e:
        print(f"Shape inference failed again: {e}")
        # Save the model with manual fixes anyway
        onnx.save(model, output_path)
        print(f"\nSaved manually fixed model to: {output_path}")
        return model

def verify_fixed_model(model_path):
    """Verify that the model has proper type information"""
    
    model = onnx.load(model_path)
    graph = model.graph
    
    print("\n=== Verifying Fixed Model ===")
    
    # Check for the problematic tensors
    value_info_dict = {vi.name: vi for vi in graph.value_info}
    
    problem_tensors = ['/encoder/encoder.3/Relu_output_0', '/Reshape_output_0']
    all_good = True
    
    for tensor_name in problem_tensors:
        if tensor_name in value_info_dict:
            vi = value_info_dict[tensor_name]
            elem_type = vi.type.tensor_type.elem_type
            shape = [dim.dim_value if dim.dim_value else 'dynamic' for dim in vi.type.tensor_type.shape.dim]
            print(f"✓ {tensor_name}: Type={onnx.TensorProto.DataType.Name(elem_type)}, Shape={shape}")
        else:
            print(f"✗ {tensor_name}: Still missing type information")
            all_good = False
    
    return all_good

if __name__ == "__main__":
    input_path = "onnx_models/codec_encoder.onnx"
    output_path = "onnx_models/codec_encoder_fixed.onnx"
    
    # Fix the model
    fixed_model = infer_and_fix_types(input_path, output_path)
    
    # Verify the fix
    verify_fixed_model(output_path)