"""
Custom ONNX operator for FiLM (Feature-wise Linear Modulation).
FiLM applies conditional affine transformations: output = gamma * input + beta
where gamma and beta are derived from a conditioning signal.
"""

import torch
import torch.nn as nn
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation layer.
    
    FiLM(x, condition) = gamma(condition) * x + beta(condition)
    """
    
    def __init__(self, feature_dim: int, condition_dim: int = None):
        super().__init__()
        self.feature_dim = feature_dim
        self.condition_dim = condition_dim or feature_dim
        
        # Linear projections to generate gamma and beta from condition
        self.gamma_linear = nn.Linear(self.condition_dim, self.feature_dim)
        self.beta_linear = nn.Linear(self.condition_dim, self.feature_dim)
        
        # Initialize to identity transform
        nn.init.ones_(self.gamma_linear.weight)
        nn.init.zeros_(self.gamma_linear.bias)
        nn.init.zeros_(self.beta_linear.weight)
        nn.init.zeros_(self.beta_linear.bias)
        
    def forward(self, x, condition=None):
        """
        Args:
            x: Features [batch, seq_len, feature_dim] or [batch, feature_dim]
            condition: Conditioning signal [batch, seq_len, condition_dim] or [batch, condition_dim]
                      If None, returns x unchanged
        """
        if condition is None:
            return x
            
        # Generate modulation parameters
        gamma = self.gamma_linear(condition)
        beta = self.beta_linear(condition)
        
        # Apply FiLM
        return gamma * x + beta


class SimpleFiLM(nn.Module):
    """
    Simplified FiLM using only ONNX-compatible operations.
    """
    
    def __init__(self, feature_dim: int, condition_dim: int = None):
        super().__init__()
        self.feature_dim = feature_dim
        self.condition_dim = condition_dim or feature_dim
        
        # Use separate weight and bias for clarity in ONNX
        self.gamma_weight = nn.Parameter(torch.ones(self.feature_dim, self.condition_dim))
        self.gamma_bias = nn.Parameter(torch.zeros(self.feature_dim))
        self.beta_weight = nn.Parameter(torch.zeros(self.feature_dim, self.condition_dim))
        self.beta_bias = nn.Parameter(torch.zeros(self.feature_dim))
        
    def forward(self, x, condition):
        """
        ONNX-friendly forward pass.
        """
        # For 3D inputs (batch, seq, dim), we need to handle the matmul properly
        if x.dim() == 3 and condition.dim() == 3:
            # gamma = condition @ gamma_weight.T + gamma_bias
            gamma = torch.matmul(condition, self.gamma_weight.t()) + self.gamma_bias
            beta = torch.matmul(condition, self.beta_weight.t()) + self.beta_bias
        else:
            # For 2D inputs
            gamma = torch.matmul(condition, self.gamma_weight.t()) + self.gamma_bias
            beta = torch.matmul(condition, self.beta_weight.t()) + self.beta_bias
        
        # Apply FiLM transformation
        return gamma * x + beta


def create_film_onnx_model(input_shape, condition_shape, feature_dim, condition_dim=None):
    """
    Create an ONNX model with FiLM implemented using basic ONNX ops.
    
    Args:
        input_shape: Shape of input tensor (e.g., [batch, seq, feature_dim])
        condition_shape: Shape of condition tensor
        feature_dim: Feature dimension
        condition_dim: Condition dimension (defaults to feature_dim)
    """
    if condition_dim is None:
        condition_dim = feature_dim
    
    # Input tensors
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, input_shape)
    condition = helper.make_tensor_value_info('condition', TensorProto.FLOAT, condition_shape)
    
    # Weight tensors
    gamma_weight = helper.make_tensor_value_info('gamma_weight', TensorProto.FLOAT, [feature_dim, condition_dim])
    gamma_bias = helper.make_tensor_value_info('gamma_bias', TensorProto.FLOAT, [feature_dim])
    beta_weight = helper.make_tensor_value_info('beta_weight', TensorProto.FLOAT, [feature_dim, condition_dim])
    beta_bias = helper.make_tensor_value_info('beta_bias', TensorProto.FLOAT, [feature_dim])
    
    # Output tensor
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, input_shape)
    
    # Create nodes
    nodes = [
        # Transpose weights for matmul
        helper.make_node('Transpose', inputs=['gamma_weight'], outputs=['gamma_weight_t'], perm=[1, 0]),
        helper.make_node('Transpose', inputs=['beta_weight'], outputs=['beta_weight_t'], perm=[1, 0]),
        
        # Compute gamma = condition @ gamma_weight.T + gamma_bias
        helper.make_node('MatMul', inputs=['condition', 'gamma_weight_t'], outputs=['gamma_linear']),
        helper.make_node('Add', inputs=['gamma_linear', 'gamma_bias'], outputs=['gamma']),
        
        # Compute beta = condition @ beta_weight.T + beta_bias
        helper.make_node('MatMul', inputs=['condition', 'beta_weight_t'], outputs=['beta_linear']),
        helper.make_node('Add', inputs=['beta_linear', 'beta_bias'], outputs=['beta']),
        
        # Apply FiLM: Y = gamma * X + beta
        helper.make_node('Mul', inputs=['gamma', 'X'], outputs=['gamma_x']),
        helper.make_node('Add', inputs=['gamma_x', 'beta'], outputs=['Y'])
    ]
    
    # Create the graph
    graph_def = helper.make_graph(
        nodes,
        'FiLM',
        [X, condition, gamma_weight, gamma_bias, beta_weight, beta_bias],
        [Y]
    )
    
    # Create the model
    model_def = helper.make_model(graph_def, producer_name='film_onnx')
    model_def.opset_import[0].version = 13
    
    # Set IR version to match ONNX Runtime requirements
    model_def.ir_version = 8  # ONNX Runtime compatible version
    
    return model_def


def test_film_implementations():
    """Test different FiLM implementations."""
    
    print("=== Testing FiLM Implementations ===\n")
    
    # Test parameters
    batch_size = 2
    seq_len = 10
    feature_dim = 64
    condition_dim = 32
    
    # Create test inputs
    x = torch.randn(batch_size, seq_len, feature_dim)
    condition = torch.randn(batch_size, seq_len, condition_dim)
    
    # 1. Test PyTorch implementation
    print("1. PyTorch FiLM")
    pytorch_film = FiLM(feature_dim, condition_dim)
    pytorch_output = pytorch_film(x, condition)
    print(f"   Output shape: {pytorch_output.shape}")
    print(f"   Output mean: {pytorch_output.mean().item():.6f}")
    print(f"   Output std: {pytorch_output.std().item():.6f}")
    
    # 2. Test SimpleFiLM
    print("\n2. SimpleFiLM (ONNX-compatible)")
    simple_film = SimpleFiLM(feature_dim, condition_dim)
    
    # Copy weights from PyTorch version
    simple_film.gamma_weight.data = pytorch_film.gamma_linear.weight.data.clone()
    simple_film.gamma_bias.data = pytorch_film.gamma_linear.bias.data.clone()
    simple_film.beta_weight.data = pytorch_film.beta_linear.weight.data.clone()
    simple_film.beta_bias.data = pytorch_film.beta_linear.bias.data.clone()
    
    simple_output = simple_film(x, condition)
    
    # Compare
    diff = torch.abs(pytorch_output - simple_output).max().item()
    print(f"   Max difference from PyTorch: {diff:.8f}")
    
    # 3. Export SimpleFiLM to ONNX
    print("\n3. Export to ONNX")
    torch.onnx.export(
        simple_film,
        (x, condition),
        "film_simple.onnx",
        input_names=['x', 'condition'],
        output_names=['output'],
        dynamic_axes={
            'x': {0: 'batch', 1: 'sequence'},
            'condition': {0: 'batch', 1: 'sequence'},
            'output': {0: 'batch', 1: 'sequence'}
        },
        opset_version=13
    )
    print("   ✓ Exported to film_simple.onnx")
    
    # Test ONNX model
    ort_session = ort.InferenceSession("film_simple.onnx")
    onnx_output = ort_session.run(None, {
        'x': x.numpy(),
        'condition': condition.numpy()
    })[0]
    
    # Compare outputs
    diff_onnx = np.abs(pytorch_output.detach().numpy() - onnx_output).max()
    print(f"   Max difference PyTorch vs ONNX: {diff_onnx:.8f}")
    
    # 4. Create custom ONNX model from scratch
    print("\n4. Custom ONNX model from scratch")
    custom_model = create_film_onnx_model(
        input_shape=[batch_size, seq_len, feature_dim],
        condition_shape=[batch_size, seq_len, condition_dim],
        feature_dim=feature_dim,
        condition_dim=condition_dim
    )
    
    # Save model
    onnx.save(custom_model, "film_custom.onnx")
    print("   ✓ Created custom ONNX model: film_custom.onnx")
    
    # Verify model
    onnx.checker.check_model(custom_model)
    print("   ✓ Model verification passed")
    
    # Test custom model
    ort_session_custom = ort.InferenceSession("film_custom.onnx")
    
    # Prepare inputs
    inputs = {
        'X': x.numpy(),
        'condition': condition.numpy(),
        'gamma_weight': simple_film.gamma_weight.detach().numpy(),
        'gamma_bias': simple_film.gamma_bias.detach().numpy(),
        'beta_weight': simple_film.beta_weight.detach().numpy(),
        'beta_bias': simple_film.beta_bias.detach().numpy()
    }
    
    custom_output = ort_session_custom.run(None, inputs)[0]
    
    # Compare with PyTorch
    diff_custom = np.abs(pytorch_output.detach().numpy() - custom_output).max()
    print(f"   Max difference PyTorch vs Custom ONNX: {diff_custom:.8f}")
    
    return pytorch_output, onnx_output, custom_output


class VampNetFiLM(nn.Module):
    """
    FiLM layer that matches VampNet's implementation style.
    VampNet might use FiLM without explicit conditioning (self-modulation).
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # In VampNet, FiLM might be implemented as simple learned scaling
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        
    def forward(self, x):
        # Simple element-wise affine transform
        return self.gamma * x + self.beta


def test_film_special_cases():
    """Test special cases of FiLM (e.g., no conditioning, self-modulation)."""
    
    print("\n=== Testing FiLM Special Cases ===\n")
    
    # Test self-modulation (using input as its own condition)
    print("1. Self-modulation FiLM")
    
    batch_size = 2
    seq_len = 10
    dim = 64
    
    x = torch.randn(batch_size, seq_len, dim)
    
    # FiLM with self-modulation
    self_film = FiLM(dim, dim)
    output = self_film(x, x)  # Use x as its own condition
    
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    
    # Export self-modulation version
    torch.onnx.export(
        self_film,
        (x, x),
        "film_self_modulation.onnx",
        input_names=['x', 'condition'],
        output_names=['output'],
        opset_version=13
    )
    print("   ✓ Exported self-modulation FiLM")
    
    # Test VampNet-style FiLM
    print("\n2. VampNet-style FiLM (no conditioning)")
    vampnet_film = VampNetFiLM(dim)
    vamp_output = vampnet_film(x)
    
    print(f"   Output shape: {vamp_output.shape}")
    
    # Export VampNet-style
    torch.onnx.export(
        vampnet_film,
        x,
        "film_vampnet_style.onnx",
        input_names=['x'],
        output_names=['output'],
        dynamic_axes={'x': {0: 'batch', 1: 'sequence'}},
        opset_version=13
    )
    print("   ✓ Exported VampNet-style FiLM")


if __name__ == "__main__":
    # Test implementations
    pytorch_out, onnx_out, custom_out = test_film_implementations()
    
    # Test special cases
    test_film_special_cases()
    
    print("\n=== Summary ===")
    print("Successfully created FiLM as a custom ONNX operator!")
    print("FiLM decomposes into simple operations: MatMul, Add, Mul")
    print("This can handle conditional feature modulation in VampNet.")
    print("\nKey insights:")
    print("- FiLM is just learned affine transforms based on conditioning")
    print("- Can be self-modulating or use external conditioning")
    print("- VampNet might use simpler variants without explicit conditioning")