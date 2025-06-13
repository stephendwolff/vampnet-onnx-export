"""
Diagnose shape mismatches in the exported transformer models.
"""

import torch
import onnxruntime as ort
import numpy as np
import onnx
from pathlib import Path


def analyze_transformer_model(model_path):
    """Analyze the transformer ONNX model to understand shape requirements."""
    
    print(f"\n=== Analyzing {model_path} ===")
    
    # Load ONNX model
    onnx_model = onnx.load(model_path)
    
    # Check model inputs/outputs
    print("\nModel Inputs:")
    for input in onnx_model.graph.input:
        print(f"  {input.name}: {[d.dim_value if d.HasField('dim_value') else d.dim_param for d in input.type.tensor_type.shape.dim]}")
    
    print("\nModel Outputs:")
    for output in onnx_model.graph.output:
        print(f"  {output.name}: {[d.dim_value if d.HasField('dim_value') else d.dim_param for d in output.type.tensor_type.shape.dim]}")
    
    # Create session
    session = ort.InferenceSession(model_path)
    
    # Get input details
    print("\nONNX Runtime Input Details:")
    for input in session.get_inputs():
        print(f"  {input.name}:")
        print(f"    Shape: {input.shape}")
        print(f"    Type: {input.type}")
    
    # Test with different input shapes
    print("\n\nTesting different input shapes...")
    
    # Test 1: Expected shape from our pipeline
    print("\nTest 1: Pipeline shape (batch=1, codebooks=4, seq=173)")
    try:
        codes = np.random.randint(0, 1024, (1, 4, 173), dtype=np.int64)
        mask = np.zeros((1, 4, 173), dtype=np.int64)
        
        outputs = session.run(None, {'codes': codes, 'mask': mask})
        print(f"  ✓ Success! Output shape: {outputs[0].shape}")
    except Exception as e:
        print(f"  ✗ Failed: {str(e)}")
        
        # Try to understand the error
        if "reshape" in str(e).lower():
            print("\n  Analyzing reshape error...")
            # Calculate what the model might be expecting
            if "692,1,512" in str(e):
                print(f"    Model reshaped to: 692 = {1 * 4 * 173} (batch * codebooks * seq)")
                print(f"    But attention expects different batch size")
    
    # Test 2: Try with different sequence length
    print("\nTest 2: Different sequence length (batch=1, codebooks=4, seq=100)")
    try:
        codes = np.random.randint(0, 1024, (1, 4, 100), dtype=np.int64)
        mask = np.zeros((1, 4, 100), dtype=np.int64)
        
        outputs = session.run(None, {'codes': codes, 'mask': mask})
        print(f"  ✓ Success! Output shape: {outputs[0].shape}")
    except Exception as e:
        print(f"  ✗ Failed: {str(e)}")
    
    # Test 3: What if we reshape input differently?
    print("\nTest 3: Single codebook (batch=1, codebooks=1, seq=173)")
    try:
        codes = np.random.randint(0, 1024, (1, 1, 173), dtype=np.int64)
        mask = np.zeros((1, 1, 173), dtype=np.int64)
        
        outputs = session.run(None, {'codes': codes, 'mask': mask})
        print(f"  ✓ Success! Output shape: {outputs[0].shape}")
    except Exception as e:
        print(f"  ✗ Failed: {str(e)}")
    
    # Analyze model structure
    print("\n\nAnalyzing model structure...")
    
    # Look for reshape operations
    reshape_nodes = [node for node in onnx_model.graph.node if node.op_type == 'Reshape']
    print(f"\nFound {len(reshape_nodes)} Reshape operations")
    
    for i, node in enumerate(reshape_nodes[:5]):  # Show first 5
        print(f"\nReshape {i}: {node.name}")
        # Try to find the shape input
        if len(node.input) > 1:
            shape_input = node.input[1]
            # Look for the constant that defines the shape
            for init in onnx_model.graph.initializer:
                if init.name == shape_input:
                    shape = onnx.numpy_helper.to_array(init)
                    print(f"  Target shape: {shape}")
                    break
    
    # Look for attention-related nodes
    attention_nodes = [node for node in onnx_model.graph.node if 'attn' in node.name.lower()]
    print(f"\nFound {len(attention_nodes)} attention-related operations")


def check_original_export_script():
    """Check how the model was originally exported."""
    export_script = Path("scripts/export_vampnet_transformer.py")
    
    if export_script.exists():
        print("\n\n=== Checking Original Export Configuration ===")
        with open(export_script, 'r') as f:
            lines = f.readlines()
        
        # Look for the export call
        for i, line in enumerate(lines):
            if 'torch.onnx.export' in line:
                print(f"\nFound export at line {i+1}:")
                # Print surrounding lines
                start = max(0, i-10)
                end = min(len(lines), i+20)
                for j in range(start, end):
                    if j == i:
                        print(f">>> {lines[j].rstrip()}")
                    else:
                        print(f"    {lines[j].rstrip()}")
                break


def main():
    """Run diagnostics on both transformer models."""
    
    print("=== Transformer Shape Diagnostics ===")
    
    # Check coarse transformer
    coarse_path = "models/transformer.onnx"
    if Path(coarse_path).exists():
        analyze_transformer_model(coarse_path)
    else:
        print(f"\nCoarse transformer not found at {coarse_path}")
    
    # Check C2F transformer
    c2f_path = "onnx_models/vampnet_c2f_transformer.onnx"
    if Path(c2f_path).exists():
        analyze_transformer_model(c2f_path)
    else:
        print(f"\nC2F transformer not found at {c2f_path}")
    
    # Check original export configuration
    check_original_export_script()
    
    print("\n\n=== Diagnosis Summary ===")
    print("The shape mismatch suggests the model was exported with fixed dimensions.")
    print("The attention mechanism expects a specific batch*sequence size.")
    print("We need to re-export with dynamic axes properly configured.")


if __name__ == "__main__":
    main()