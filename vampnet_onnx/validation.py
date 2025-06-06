"""
ONNX model validation and testing utilities.
"""

import torch
import numpy as np
import onnx
import onnxruntime as ort
from typing import Dict, List, Tuple, Optional, Any
import time


def validate_onnx_model(onnx_path: str) -> bool:
    """
    Validate an ONNX model file.
    
    Args:
        onnx_path: Path to ONNX model
        
    Returns:
        True if model is valid
    """
    try:
        # Load the model
        onnx_model = onnx.load(onnx_path)
        
        # Check the model
        onnx.checker.check_model(onnx_model)
        
        print(f"✓ Model {onnx_path} is valid")
        
        # Print model info
        print(f"  - IR version: {onnx_model.ir_version}")
        print(f"  - Producer: {onnx_model.producer_name}")
        print(f"  - Opset version: {onnx_model.opset_import[0].version}")
        
        # Print input/output info
        print("  - Inputs:")
        for input in onnx_model.graph.input:
            print(f"    - {input.name}: {_get_tensor_shape(input)}")
            
        print("  - Outputs:")
        for output in onnx_model.graph.output:
            print(f"    - {output.name}: {_get_tensor_shape(output)}")
            
        return True
        
    except Exception as e:
        print(f"✗ Model validation failed: {e}")
        return False


def _get_tensor_shape(tensor_proto):
    """Extract shape from tensor proto."""
    shape = []
    for dim in tensor_proto.type.tensor_type.shape.dim:
        if dim.HasField('dim_value'):
            shape.append(dim.dim_value)
        elif dim.HasField('dim_param'):
            shape.append(dim.dim_param)
        else:
            shape.append('?')
    return shape


def create_onnx_session(onnx_path: str, 
                       providers: Optional[List[str]] = None) -> ort.InferenceSession:
    """
    Create ONNX Runtime inference session.
    
    Args:
        onnx_path: Path to ONNX model
        providers: Execution providers (default: auto-detect)
        
    Returns:
        ONNX Runtime session
    """
    if providers is None:
        # Auto-detect available providers
        available_providers = ort.get_available_providers()
        providers = []
        
        if 'CUDAExecutionProvider' in available_providers:
            providers.append('CUDAExecutionProvider')
        if 'CoreMLExecutionProvider' in available_providers:
            providers.append('CoreMLExecutionProvider')
        providers.append('CPUExecutionProvider')
    
    # Create session
    session = ort.InferenceSession(onnx_path, providers=providers)
    
    print(f"Created ONNX Runtime session with providers: {providers}")
    return session


def run_onnx_inference(session: ort.InferenceSession,
                      inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Run inference with ONNX Runtime.
    
    Args:
        session: ONNX Runtime session
        inputs: Dictionary of input names to numpy arrays
        
    Returns:
        Dictionary of output names to numpy arrays
    """
    # Get input/output names
    input_names = [i.name for i in session.get_inputs()]
    output_names = [o.name for o in session.get_outputs()]
    
    # Validate inputs
    for name in input_names:
        if name not in inputs:
            raise ValueError(f"Missing input: {name}")
    
    # Run inference
    outputs = session.run(output_names, inputs)
    
    # Create output dictionary
    result = {}
    for name, value in zip(output_names, outputs):
        result[name] = value
        
    return result


def compare_outputs(pytorch_output: Any,
                   onnx_output: Any,
                   rtol: float = 1e-3,
                   atol: float = 1e-5,
                   name: str = "output") -> bool:
    """
    Compare PyTorch and ONNX outputs.
    
    Args:
        pytorch_output: Output from PyTorch model
        onnx_output: Output from ONNX model
        rtol: Relative tolerance
        atol: Absolute tolerance
        name: Name for error messages
        
    Returns:
        True if outputs match within tolerance
    """
    # Convert to numpy if needed
    if isinstance(pytorch_output, torch.Tensor):
        pytorch_output = pytorch_output.detach().cpu().numpy()
    if isinstance(onnx_output, torch.Tensor):
        onnx_output = onnx_output.detach().cpu().numpy()
        
    # Handle different types
    if isinstance(pytorch_output, (list, tuple)):
        if not isinstance(onnx_output, (list, tuple)):
            print(f"✗ Type mismatch for {name}: PyTorch returned {type(pytorch_output)}, ONNX returned {type(onnx_output)}")
            return False
            
        if len(pytorch_output) != len(onnx_output):
            print(f"✗ Length mismatch for {name}: PyTorch returned {len(pytorch_output)}, ONNX returned {len(onnx_output)}")
            return False
            
        # Compare each element
        all_match = True
        for i, (pt, onnx) in enumerate(zip(pytorch_output, onnx_output)):
            if not compare_outputs(pt, onnx, rtol, atol, f"{name}[{i}]"):
                all_match = False
                
        return all_match
        
    elif isinstance(pytorch_output, dict):
        if not isinstance(onnx_output, dict):
            print(f"✗ Type mismatch for {name}: PyTorch returned dict, ONNX returned {type(onnx_output)}")
            return False
            
        # Compare keys
        pt_keys = set(pytorch_output.keys())
        onnx_keys = set(onnx_output.keys())
        
        if pt_keys != onnx_keys:
            print(f"✗ Key mismatch for {name}: PyTorch keys {pt_keys}, ONNX keys {onnx_keys}")
            return False
            
        # Compare values
        all_match = True
        for key in pt_keys:
            if not compare_outputs(pytorch_output[key], onnx_output[key], rtol, atol, f"{name}['{key}']"):
                all_match = False
                
        return all_match
        
    else:
        # Compare numpy arrays
        try:
            if not np.allclose(pytorch_output, onnx_output, rtol=rtol, atol=atol):
                diff = np.abs(pytorch_output - onnx_output)
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)
                print(f"✗ Output mismatch for {name}:")
                print(f"  - Max difference: {max_diff}")
                print(f"  - Mean difference: {mean_diff}")
                print(f"  - PyTorch shape: {pytorch_output.shape}")
                print(f"  - ONNX shape: {onnx_output.shape}")
                return False
            else:
                print(f"✓ Output match for {name}")
                return True
                
        except Exception as e:
            print(f"✗ Comparison failed for {name}: {e}")
            return False


def benchmark_model(session: ort.InferenceSession,
                   inputs: Dict[str, np.ndarray],
                   n_runs: int = 100,
                   warmup_runs: int = 10) -> Dict[str, float]:
    """
    Benchmark ONNX model performance.
    
    Args:
        session: ONNX Runtime session
        inputs: Input dictionary
        n_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
        
    Returns:
        Performance statistics
    """
    print(f"Benchmarking model ({n_runs} runs after {warmup_runs} warmup)...")
    
    # Warmup
    for _ in range(warmup_runs):
        _ = run_onnx_inference(session, inputs)
    
    # Benchmark
    times = []
    for _ in range(n_runs):
        start = time.time()
        _ = run_onnx_inference(session, inputs)
        end = time.time()
        times.append(end - start)
    
    times = np.array(times)
    
    stats = {
        'mean_ms': np.mean(times) * 1000,
        'std_ms': np.std(times) * 1000,
        'min_ms': np.min(times) * 1000,
        'max_ms': np.max(times) * 1000,
        'median_ms': np.median(times) * 1000,
        'p95_ms': np.percentile(times, 95) * 1000,
        'p99_ms': np.percentile(times, 99) * 1000,
    }
    
    print("Benchmark results:")
    for key, value in stats.items():
        print(f"  - {key}: {value:.2f}")
        
    return stats


def validate_vampnet_component(
    pytorch_model: torch.nn.Module,
    onnx_path: str,
    example_inputs: Tuple[torch.Tensor, ...],
    input_names: List[str],
    rtol: float = 1e-3,
    atol: float = 1e-5
) -> bool:
    """
    Validate a VampNet component's ONNX export.
    
    Args:
        pytorch_model: Original PyTorch model
        onnx_path: Path to ONNX model
        example_inputs: Example inputs
        input_names: Input names for ONNX
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Returns:
        True if validation passes
    """
    print(f"\nValidating {onnx_path}...")
    
    # Validate ONNX model structure
    if not validate_onnx_model(onnx_path):
        return False
    
    # Create ONNX session
    try:
        session = create_onnx_session(onnx_path)
    except Exception as e:
        print(f"✗ Failed to create ONNX session: {e}")
        return False
    
    # Prepare inputs
    onnx_inputs = {}
    for name, tensor in zip(input_names, example_inputs):
        onnx_inputs[name] = tensor.detach().cpu().numpy()
    
    # Run PyTorch model
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(*example_inputs)
    
    # Run ONNX model
    try:
        onnx_output = run_onnx_inference(session, onnx_inputs)
        
        # Handle single output case
        if len(onnx_output) == 1:
            onnx_output = list(onnx_output.values())[0]
            
    except Exception as e:
        print(f"✗ ONNX inference failed: {e}")
        return False
    
    # Compare outputs
    if compare_outputs(pytorch_output, onnx_output, rtol, atol):
        print("✓ Validation passed!")
        
        # Run benchmark
        benchmark_model(session, onnx_inputs)
        
        return True
    else:
        print("✗ Validation failed!")
        return False


def analyze_model_size(onnx_path: str) -> Dict[str, Any]:
    """
    Analyze ONNX model size and complexity.
    
    Args:
        onnx_path: Path to ONNX model
        
    Returns:
        Model analysis results
    """
    import os
    
    # Get file size
    file_size = os.path.getsize(onnx_path)
    
    # Load model
    model = onnx.load(onnx_path)
    
    # Count operations
    op_counts = {}
    for node in model.graph.node:
        op_type = node.op_type
        op_counts[op_type] = op_counts.get(op_type, 0) + 1
    
    # Count parameters
    param_count = 0
    param_size = 0
    
    for init in model.graph.initializer:
        shape = [dim for dim in init.dims]
        size = np.prod(shape) if shape else 1
        param_count += 1
        param_size += size * 4  # Assume float32
    
    analysis = {
        'file_size_mb': file_size / (1024 * 1024),
        'param_count': param_count,
        'param_size_mb': param_size / (1024 * 1024),
        'op_counts': op_counts,
        'total_ops': sum(op_counts.values()),
        'unique_ops': len(op_counts),
    }
    
    print(f"\nModel analysis for {onnx_path}:")
    print(f"  - File size: {analysis['file_size_mb']:.2f} MB")
    print(f"  - Parameters: {analysis['param_count']:,}")
    print(f"  - Parameter size: {analysis['param_size_mb']:.2f} MB")
    print(f"  - Total operations: {analysis['total_ops']}")
    print(f"  - Unique operations: {analysis['unique_ops']}")
    print(f"  - Top operations:")
    
    sorted_ops = sorted(op_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    for op, count in sorted_ops:
        print(f"    - {op}: {count}")
    
    return analysis