"""
Test and demonstrate custom RMSNorm ONNX operator.
"""

import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from custom_ops.rmsnorm_onnx import RMSNorm, SimpleRMSNorm, create_rmsnorm_onnx_model


def compare_rmsnorm_with_vampnet():
    """Compare our RMSNorm with VampNet's implementation."""
    
    print("=== Comparing RMSNorm Implementations ===\n")
    
    # Import VampNet to check their RMSNorm
    try:
        import vampnet
        
        # Load model to inspect RMSNorm
        interface = vampnet.interface.Interface(
            codec_ckpt="../models/vampnet/codec.pth",
            coarse_ckpt="../models/vampnet/coarse.pth",
        )
        
        model = interface.coarse
        if hasattr(model, '_orig_mod'):
            model = model._orig_mod
        
        # Find an RMSNorm layer
        rmsnorm_layer = None
        for name, module in model.named_modules():
            if 'RMSNorm' in type(module).__name__:
                rmsnorm_layer = module
                print(f"Found RMSNorm layer: {name}")
                print(f"  Type: {type(module)}")
                print(f"  Has weight: {hasattr(module, 'weight')}")
                if hasattr(module, 'weight'):
                    print(f"  Weight shape: {module.weight.shape}")
                break
        
        if rmsnorm_layer:
            # Test with same input
            dim = rmsnorm_layer.weight.shape[0]
            test_input = torch.randn(1, 10, dim)
            
            # VampNet's output
            vampnet_output = rmsnorm_layer(test_input)
            
            # Our implementation
            our_rmsnorm = RMSNorm(dim)
            our_rmsnorm.weight.data = rmsnorm_layer.weight.data.clone()
            our_output = our_rmsnorm(test_input)
            
            # Compare
            diff = torch.abs(vampnet_output - our_output).max().item()
            print(f"\nMax difference: {diff}")
            
            if diff < 1e-5:
                print("✓ Our RMSNorm matches VampNet's implementation!")
            else:
                print("✗ Implementations differ")
                
    except Exception as e:
        print(f"Could not load VampNet for comparison: {e}")


def demonstrate_rmsnorm_export():
    """Demonstrate exporting RMSNorm to ONNX."""
    
    print("\n=== Demonstrating RMSNorm ONNX Export ===\n")
    
    # Create a model with RMSNorm
    class ModelWithRMSNorm(nn.Module):
        def __init__(self, dim=64):
            super().__init__()
            self.linear1 = nn.Linear(dim, dim)
            self.norm = SimpleRMSNorm(dim)
            self.linear2 = nn.Linear(dim, dim)
            
        def forward(self, x):
            x = self.linear1(x)
            x = self.norm(x)
            x = self.linear2(x)
            return x
    
    # Create model
    model = ModelWithRMSNorm(64)
    model.eval()
    
    # Test input
    x = torch.randn(2, 10, 64)
    
    # Get PyTorch output
    with torch.no_grad():
        pytorch_output = model(x)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        x,
        "model_with_rmsnorm.onnx",
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch', 1: 'sequence'}},
        opset_version=13,
        verbose=True
    )
    
    print("✓ Exported model with RMSNorm to ONNX")
    
    # Load and test ONNX model
    ort_session = ort.InferenceSession("model_with_rmsnorm.onnx")
    onnx_output = ort_session.run(None, {'input': x.numpy()})[0]
    
    # Compare outputs
    diff = np.abs(pytorch_output.numpy() - onnx_output).max()
    print(f"Max difference PyTorch vs ONNX: {diff:.8f}")
    
    # Inspect the ONNX model
    onnx_model = onnx.load("model_with_rmsnorm.onnx")
    
    print("\nONNX model operations:")
    for node in onnx_model.graph.node:
        print(f"  {node.op_type}: {node.name}")
        if node.op_type in ['Mul', 'Sqrt', 'ReduceMean', 'Div']:
            print(f"    (Part of RMSNorm)")


def create_optimized_rmsnorm():
    """Create an optimized version of RMSNorm for ONNX."""
    
    print("\n=== Creating Optimized RMSNorm ===\n")
    
    class OptimizedRMSNorm(nn.Module):
        """
        Optimized RMSNorm that's more efficient in ONNX.
        Uses fused operations where possible.
        """
        
        def __init__(self, dim, eps=1e-8):
            super().__init__()
            self.dim = dim
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))
            
        def forward(self, x):
            # Compute variance (which is mean of squares for RMS)
            variance = x.pow(2).mean(dim=-1, keepdim=True)
            
            # Add epsilon and take reciprocal square root
            # This is more efficient than sqrt followed by division
            inv_rms = torch.rsqrt(variance + self.eps)
            
            # Normalize and scale in one operation
            return x * inv_rms * self.weight
    
    # Test the optimized version
    dim = 64
    x = torch.randn(2, 10, dim)
    
    # Compare implementations
    standard = SimpleRMSNorm(dim)
    optimized = OptimizedRMSNorm(dim)
    
    with torch.no_grad():
        standard_out = standard(x)
        optimized_out = optimized(x)
    
    diff = torch.abs(standard_out - optimized_out).max().item()
    print(f"Difference between standard and optimized: {diff:.8f}")
    
    # Export both versions
    for name, model in [("standard", standard), ("optimized", optimized)]:
        torch.onnx.export(
            model,
            x,
            f"rmsnorm_{name}.onnx",
            input_names=['input'],
            output_names=['output'],
            opset_version=13
        )
        
        # Check model size
        model_size = os.path.getsize(f"rmsnorm_{name}.onnx")
        print(f"{name.capitalize()} RMSNorm ONNX size: {model_size} bytes")


if __name__ == "__main__":
    # Run all demonstrations
    compare_rmsnorm_with_vampnet()
    demonstrate_rmsnorm_export()
    create_optimized_rmsnorm()
    
    print("\n=== Conclusion ===")
    print("RMSNorm can be successfully exported to ONNX using basic operations.")
    print("This shows that VampNet's custom layers can be converted to ONNX")
    print("by breaking them down into fundamental operations.")
    print("\nNext: We can apply the same approach to FiLM and CodebookEmbedding layers.")