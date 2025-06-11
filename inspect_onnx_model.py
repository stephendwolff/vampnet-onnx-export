#!/usr/bin/env python3
"""Inspect ONNX model to understand the fixed output size issue."""

import onnx
import numpy as np

# Load the model
model_path = "scripts/models/vampnet_codec_encoder.onnx"
model = onnx.load(model_path)

print(f"Inspecting: {model_path}\n")

# Check inputs
print("Model inputs:")
for inp in model.graph.input:
    dims = []
    for d in inp.type.tensor_type.shape.dim:
        if hasattr(d, 'dim_value') and d.dim_value > 0:
            dims.append(str(d.dim_value))
        elif hasattr(d, 'dim_param') and d.dim_param:
            dims.append(d.dim_param)
        else:
            dims.append('?')
    print(f"  {inp.name}: [{', '.join(dims)}]")

# Check outputs  
print("\nModel outputs:")
for out in model.graph.output:
    dims = []
    for d in out.type.tensor_type.shape.dim:
        if hasattr(d, 'dim_value') and d.dim_value > 0:
            dims.append(str(d.dim_value))
        elif hasattr(d, 'dim_param') and d.dim_param:
            dims.append(d.dim_param)
        else:
            dims.append('?')
    print(f"  {out.name}: [{', '.join(dims)}]")

# Check for any constants that might be 173
print("\nChecking for constant value 173 in model...")
for init in model.graph.initializer:
    if init.data_type == onnx.TensorProto.INT64:
        data = np.frombuffer(init.raw_data, dtype=np.int64)
        if 173 in data:
            print(f"  Found 173 in initializer: {init.name}")

# Check node operations
print("\nChecking key operations:")
for node in model.graph.node:
    if node.op_type in ["Reshape", "Slice", "Pad", "Concat", "Gather"]:
        print(f"  {node.op_type}: {node.name}")
        for attr in node.attribute:
            if attr.name in ["shape", "axes", "starts", "ends"]:
                print(f"    {attr.name}: {attr}")
    elif "173" in str(node):
        print(f"  Node contains 173: {node.name} ({node.op_type})")