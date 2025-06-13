import onnx
import onnx.helper
import onnx.numpy_helper
from collections import defaultdict
import numpy as np

def analyze_onnx_model(model_path):
    """Analyze ONNX model structure and data types"""
    
    # Load the model
    model = onnx.load(model_path)
    
    print("=== Model Analysis ===")
    print(f"Model IR Version: {model.ir_version}")
    print(f"Producer Name: {model.producer_name}")
    print(f"Producer Version: {model.producer_version}")
    print(f"Opset Version: {model.opset_import[0].version if model.opset_import else 'Unknown'}")
    
    # Analyze graph
    graph = model.graph
    print(f"\nGraph Name: {graph.name}")
    print(f"Number of Nodes: {len(graph.node)}")
    print(f"Number of Inputs: {len(graph.input)}")
    print(f"Number of Outputs: {len(graph.output)}")
    print(f"Number of Initializers: {len(graph.initializer)}")
    
    # Create a mapping of tensor names to their types
    tensor_types = {}
    
    # Add input types
    print("\n=== Model Inputs ===")
    for input_tensor in graph.input:
        tensor_name = input_tensor.name
        tensor_type = input_tensor.type.tensor_type
        elem_type = tensor_type.elem_type
        shape = [dim.dim_value if dim.dim_value else 'dynamic' for dim in tensor_type.shape.dim]
        tensor_types[tensor_name] = elem_type
        print(f"Input: {tensor_name}, Type: {onnx.TensorProto.DataType.Name(elem_type)}, Shape: {shape}")
    
    # Add initializer types
    print("\n=== Initializers (first 10) ===")
    for i, init in enumerate(graph.initializer[:10]):
        tensor_types[init.name] = init.data_type
        print(f"Initializer: {init.name}, Type: {onnx.TensorProto.DataType.Name(init.data_type)}, Shape: {list(init.dims)}")
    
    # Find specific nodes and tensors mentioned in the warnings
    print("\n=== Searching for Problem Tensors ===")
    
    # Track all tensor names and their producers
    tensor_producers = {}
    tensor_consumers = defaultdict(list)
    
    # Analyze nodes
    encoder3_nodes = []
    reshape_nodes = []
    
    for node in graph.node:
        # Track tensor relationships
        for output in node.output:
            tensor_producers[output] = node
        for input_name in node.input:
            tensor_consumers[input_name].append(node)
        
        # Look for encoder.3 related nodes
        if 'encoder.3' in node.name or any('encoder.3' in out for out in node.output):
            encoder3_nodes.append(node)
        
        # Look for Reshape nodes
        if node.op_type == 'Reshape':
            reshape_nodes.append(node)
        
        # Check for the specific problem tensors
        if '/encoder/encoder.3/Relu_output_0' in node.output:
            print(f"\nFound encoder.3 Relu node:")
            print(f"  Node Name: {node.name}")
            print(f"  Op Type: {node.op_type}")
            print(f"  Inputs: {list(node.input)}")
            print(f"  Outputs: {list(node.output)}")
            
        if '/Reshape_output_0' in node.output:
            print(f"\nFound Reshape node:")
            print(f"  Node Name: {node.name}")
            print(f"  Op Type: {node.op_type}")
            print(f"  Inputs: {list(node.input)}")
            print(f"  Outputs: {list(node.output)}")
    
    # Analyze encoder.3 nodes
    print(f"\n=== Encoder.3 Related Nodes ({len(encoder3_nodes)} found) ===")
    for node in encoder3_nodes[:5]:  # Show first 5
        print(f"Node: {node.name}, Op: {node.op_type}")
        print(f"  Inputs: {list(node.input)[:3]}...")  # Show first 3 inputs
        print(f"  Outputs: {list(node.output)}")
    
    # Analyze Reshape nodes
    print(f"\n=== Reshape Nodes ({len(reshape_nodes)} found) ===")
    for i, node in enumerate(reshape_nodes[:5]):  # Show first 5
        print(f"\nReshape Node {i+1}: {node.name}")
        print(f"  Inputs: {list(node.input)}")
        print(f"  Outputs: {list(node.output)}")
        
        # Try to find the data type of inputs
        for input_name in node.input:
            if input_name in tensor_types:
                print(f"  Input '{input_name}' type: {onnx.TensorProto.DataType.Name(tensor_types[input_name])}")
            else:
                print(f"  Input '{input_name}' type: Unknown")
    
    # Check value info for intermediate tensors
    print("\n=== Value Info (Intermediate Tensors) ===")
    value_info_dict = {}
    for value_info in graph.value_info:
        tensor_name = value_info.name
        tensor_type = value_info.type.tensor_type
        elem_type = tensor_type.elem_type
        shape = [dim.dim_value if dim.dim_value else 'dynamic' for dim in tensor_type.shape.dim]
        value_info_dict[tensor_name] = (elem_type, shape)
        
        # Check if it's one of our problem tensors
        if 'encoder.3' in tensor_name or 'Reshape_output' in tensor_name:
            print(f"Tensor: {tensor_name}, Type: {onnx.TensorProto.DataType.Name(elem_type)}, Shape: {shape}")
    
    # Look for missing type information
    print("\n=== Analyzing Type Inference Issues ===")
    
    # Check if problem tensors are in value_info
    problem_tensors = ['/encoder/encoder.3/Relu_output_0', '/Reshape_output_0']
    for tensor_name in problem_tensors:
        if tensor_name in value_info_dict:
            elem_type, shape = value_info_dict[tensor_name]
            print(f"Found {tensor_name} in value_info: Type={onnx.TensorProto.DataType.Name(elem_type)}, Shape={shape}")
        else:
            print(f"WARNING: {tensor_name} NOT found in value_info - this is likely the issue!")
            
            # Try to trace back to find the type
            if tensor_name in tensor_producers:
                producer_node = tensor_producers[tensor_name]
                print(f"  Produced by node: {producer_node.name} (op: {producer_node.op_type})")
                
                # Check input types
                for input_name in producer_node.input:
                    if input_name in tensor_types:
                        print(f"    Input '{input_name}' has type: {onnx.TensorProto.DataType.Name(tensor_types[input_name])}")
                    elif input_name in value_info_dict:
                        elem_type, _ = value_info_dict[input_name]
                        print(f"    Input '{input_name}' has type: {onnx.TensorProto.DataType.Name(elem_type)}")
    
    # Count nodes by operation type
    print("\n=== Node Operation Types ===")
    op_counts = defaultdict(int)
    for node in graph.node:
        op_counts[node.op_type] += 1
    
    for op_type, count in sorted(op_counts.items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f"{op_type}: {count}")
    
    return model, graph, tensor_producers, value_info_dict

if __name__ == "__main__":
    model_path = "onnx_models/codec_encoder.onnx"
    analyze_onnx_model(model_path)