"""
Custom ONNX operator for RMSNorm (Root Mean Square Normalization).
This is the simplest of VampNet's custom layers.
"""

import torch
import torch.nn as nn
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
from torch.onnx import symbolic_helper


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    RMSNorm(x) = x / RMS(x) * gamma
    where RMS(x) = sqrt(mean(x^2))
    """
    
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x):
        # Calculate RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        x_normed = x / rms
        return self.weight * x_normed


# Method 1: Using torch.onnx.symbolic_opset
def rmsnorm_symbolic(g, x, weight, eps):
    """
    Symbolic function for RMSNorm that tells PyTorch how to export to ONNX.
    This breaks down RMSNorm into basic ONNX operations.
    """
    # x^2
    x_squared = g.op("Mul", x, x)
    
    # mean(x^2, dim=-1, keepdim=True)
    # ReduceMean with axes=[-1]
    axes = g.op("Constant", value_t=torch.tensor([-1], dtype=torch.long))
    mean_x_squared = g.op("ReduceMean", x_squared, axes, keepdims_i=1)
    
    # mean + eps
    eps_const = g.op("Constant", value_t=torch.tensor(eps, dtype=torch.float32))
    mean_plus_eps = g.op("Add", mean_x_squared, eps_const)
    
    # sqrt(mean + eps)
    rms = g.op("Sqrt", mean_plus_eps)
    
    # x / rms
    x_normed = g.op("Div", x, rms)
    
    # weight * x_normed
    output = g.op("Mul", weight, x_normed)
    
    return output


# Register the symbolic function
torch.onnx.register_custom_op_symbolic(
    "custom_ops::rmsnorm",
    rmsnorm_symbolic,
    opset_version=11
)


class RMSNormONNX(nn.Module):
    """
    RMSNorm module that uses custom ONNX export.
    """
    
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x):
        # During normal PyTorch execution
        if not torch.onnx.is_in_onnx_export():
            rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
            x_normed = x / rms
            return self.weight * x_normed
        else:
            # During ONNX export, use custom op
            return torch.ops.custom_ops.rmsnorm(x, self.weight, self.eps)


# Method 2: Create a custom ONNX operator from scratch
def create_rmsnorm_onnx_model(input_shape, weight_shape, eps=1e-8):
    """
    Create an ONNX model with RMSNorm implemented using basic ONNX ops.
    """
    # Input tensors
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, input_shape)
    weight = helper.make_tensor_value_info('weight', TensorProto.FLOAT, weight_shape)
    
    # Output tensor
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, input_shape)
    
    # Create the ops
    nodes = [
        # X^2
        helper.make_node('Mul', inputs=['X', 'X'], outputs=['X_squared']),
        
        # mean(X^2, axis=-1, keepdim=True)
        helper.make_node('ReduceMean', inputs=['X_squared'], outputs=['mean_X_squared'],
                        axes=[-1], keepdims=1),
        
        # mean + eps
        helper.make_node('Add', inputs=['mean_X_squared', 'eps'], outputs=['mean_plus_eps']),
        
        # sqrt(mean + eps)
        helper.make_node('Sqrt', inputs=['mean_plus_eps'], outputs=['rms']),
        
        # X / rms
        helper.make_node('Div', inputs=['X', 'rms'], outputs=['X_normed']),
        
        # weight * X_normed
        helper.make_node('Mul', inputs=['weight', 'X_normed'], outputs=['Y'])
    ]
    
    # Create constant for eps
    eps_tensor = helper.make_tensor('eps', TensorProto.FLOAT, [1], [eps])
    
    # Create the graph
    graph_def = helper.make_graph(
        nodes,
        'RMSNorm',
        [X, weight],
        [Y],
        initializer=[eps_tensor]
    )
    
    # Create the model
    model_def = helper.make_model(graph_def, producer_name='rmsnorm_onnx')
    model_def.opset_import[0].version = 13
    
    return model_def


class SimpleRMSNorm(nn.Module):
    """Simple RMSNorm using only ONNX-compatible operations."""
    
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x):
        # Use only ONNX-compatible operations
        x_squared = x * x
        mean_x_squared = x_squared.mean(dim=-1, keepdim=True)
        rms = torch.sqrt(mean_x_squared + self.eps)
        x_normed = x / rms
        return self.weight * x_normed


def test_rmsnorm_implementations():
    """Test different RMSNorm implementations."""
    
    print("=== Testing RMSNorm Implementations ===\n")
    
    # Test parameters
    batch_size = 2
    seq_len = 10
    dim = 64
    
    # Create test input
    x = torch.randn(batch_size, seq_len, dim)
    
    # 1. Test PyTorch implementation
    print("1. PyTorch RMSNorm")
    pytorch_norm = RMSNorm(dim)
    pytorch_output = pytorch_norm(x)
    print(f"   Output shape: {pytorch_output.shape}")
    print(f"   Output mean: {pytorch_output.mean().item():.6f}")
    print(f"   Output std: {pytorch_output.std().item():.6f}")
    
    # 2. Export PyTorch model to ONNX using basic ops
    print("\n2. Export to ONNX using basic ops")
    
    simple_norm = SimpleRMSNorm(dim)
    simple_norm.weight.data = pytorch_norm.weight.data.clone()
    
    # Export to ONNX
    torch.onnx.export(
        simple_norm,
        x,
        "rmsnorm_simple.onnx",
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch', 1: 'sequence'}},
        opset_version=13
    )
    print("   ✓ Exported to rmsnorm_simple.onnx")
    
    # Test ONNX model
    ort_session = ort.InferenceSession("rmsnorm_simple.onnx")
    onnx_output = ort_session.run(None, {'input': x.numpy()})[0]
    
    # Compare outputs
    diff = np.abs(pytorch_output.detach().numpy() - onnx_output).max()
    print(f"   Max difference PyTorch vs ONNX: {diff:.8f}")
    
    # 3. Create custom ONNX model from scratch
    print("\n3. Custom ONNX model from scratch")
    custom_model = create_rmsnorm_onnx_model(
        input_shape=[batch_size, seq_len, dim],
        weight_shape=[dim]
    )
    
    # Save model
    onnx.save(custom_model, "rmsnorm_custom.onnx")
    print("   ✓ Created custom ONNX model: rmsnorm_custom.onnx")
    
    # Verify model
    onnx.checker.check_model(custom_model)
    print("   ✓ Model verification passed")
    
    # Test custom model
    ort_session_custom = ort.InferenceSession("rmsnorm_custom.onnx")
    
    # Prepare inputs
    inputs = {
        'X': x.numpy(),
        'weight': pytorch_norm.weight.detach().numpy()
    }
    
    custom_output = ort_session_custom.run(None, inputs)[0]
    
    # Compare with PyTorch
    diff_custom = np.abs(pytorch_output.detach().numpy() - custom_output).max()
    print(f"   Max difference PyTorch vs Custom ONNX: {diff_custom:.8f}")
    
    return pytorch_output, onnx_output, custom_output


def create_vampnet_compatible_rmsnorm():
    """
    Create an RMSNorm that matches VampNet's implementation.
    """
    
    class VampNetRMSNorm(nn.Module):
        """RMSNorm matching VampNet's implementation."""
        
        def __init__(self, dim: int, eps: float = 1e-8):
            super().__init__()
            self.eps = eps
            self.scale = nn.Parameter(torch.ones(dim))
            
        def forward(self, x):
            # VampNet might use a specific axis or normalization approach
            # This matches the standard RMSNorm formula
            norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
            return x / norm * self.scale
    
    return VampNetRMSNorm


if __name__ == "__main__":
    # Test implementations
    pytorch_out, onnx_out, custom_out = test_rmsnorm_implementations()
    
    print("\n=== Summary ===")
    print("Successfully created RMSNorm as a custom ONNX operator!")
    print("This can be used as a building block for exporting VampNet to ONNX.")
    print("\nNext steps:")
    print("1. Implement FiLM layer (more complex, involves conditional scaling)")
    print("2. Implement CodebookEmbedding (handles discrete token embeddings)")
    print("3. Combine all custom ops to export full VampNet model")