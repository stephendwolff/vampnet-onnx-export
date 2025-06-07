"""
Visualize custom ONNX operators to understand their structure.
"""

import onnx
import onnx.helper as helper
from onnx import numpy_helper
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import os
from custom_ops.rmsnorm_onnx import SimpleRMSNorm
from custom_ops.film_onnx import SimpleFiLM


def visualize_onnx_graph(model_path, title):
    """Visualize ONNX model operations."""
    model = onnx.load(model_path)
    
    print(f"\n=== {title} ===")
    print(f"IR version: {model.ir_version}")
    print(f"Producer: {model.producer_name}")
    print(f"Opset version: {model.opset_import[0].version}")
    
    print("\nInputs:")
    for inp in model.graph.input:
        print(f"  {inp.name}: {[d.dim_value if d.dim_value else d.dim_param for d in inp.type.tensor_type.shape.dim]}")
    
    print("\nOutputs:")
    for out in model.graph.output:
        print(f"  {out.name}: {[d.dim_value if d.dim_value else d.dim_param for d in out.type.tensor_type.shape.dim]}")
    
    print("\nOperations:")
    op_counts = {}
    for i, node in enumerate(model.graph.node):
        op_type = node.op_type
        op_counts[op_type] = op_counts.get(op_type, 0) + 1
        print(f"  {i}: {op_type} ({node.name})")
        if node.input:
            print(f"     Inputs: {list(node.input)}")
        if node.output:
            print(f"     Outputs: {list(node.output)}")
    
    print("\nOperation summary:")
    for op, count in sorted(op_counts.items()):
        print(f"  {op}: {count}")


def create_comparison_plot():
    """Create visual comparison of RMSNorm and FiLM operations."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Generate test data
    x = torch.randn(1, 10, 64)
    condition = torch.randn(1, 10, 32)
    
    # RMSNorm visualization
    rmsnorm = SimpleRMSNorm(64)
    
    # Before RMSNorm
    axes[0, 0].imshow(x[0].numpy(), aspect='auto', cmap='coolwarm')
    axes[0, 0].set_title('Input to RMSNorm')
    axes[0, 0].set_xlabel('Feature dimension')
    axes[0, 0].set_ylabel('Sequence position')
    
    # After RMSNorm
    rmsnorm_out = rmsnorm(x)
    axes[0, 1].imshow(rmsnorm_out[0].detach().numpy(), aspect='auto', cmap='coolwarm')
    axes[0, 1].set_title('Output from RMSNorm (normalized)')
    axes[0, 1].set_xlabel('Feature dimension')
    axes[0, 1].set_ylabel('Sequence position')
    
    # FiLM visualization
    film = SimpleFiLM(64, 32)
    
    # Before FiLM
    axes[1, 0].imshow(x[0].numpy(), aspect='auto', cmap='coolwarm')
    axes[1, 0].set_title('Input to FiLM')
    axes[1, 0].set_xlabel('Feature dimension')
    axes[1, 0].set_ylabel('Sequence position')
    
    # After FiLM
    film_out = film(x, condition)
    axes[1, 1].imshow(film_out[0].detach().numpy(), aspect='auto', cmap='coolwarm')
    axes[1, 1].set_title('Output from FiLM (modulated)')
    axes[1, 1].set_xlabel('Feature dimension')
    axes[1, 1].set_ylabel('Sequence position')
    
    plt.tight_layout()
    plt.savefig('custom_ops_visualization.png', dpi=150)
    plt.show()


def analyze_ops_breakdown():
    """Analyze how custom ops break down into basic operations."""
    
    print("\n=== Custom Operators Breakdown ===\n")
    
    print("1. RMSNorm breakdown:")
    print("   RMSNorm(x) = x / sqrt(mean(x²) + eps) * weight")
    print("   ONNX operations:")
    print("   - Mul (x²)")
    print("   - ReduceMean (mean of x²)")
    print("   - Add (+ epsilon)")
    print("   - Sqrt (square root)")
    print("   - Div (normalize)")
    print("   - Mul (scale by weight)")
    
    print("\n2. FiLM breakdown:")
    print("   FiLM(x, condition) = gamma(condition) * x + beta(condition)")
    print("   ONNX operations:")
    print("   - MatMul (condition @ gamma_weight)")
    print("   - Add (+ gamma_bias)")
    print("   - MatMul (condition @ beta_weight)")
    print("   - Add (+ beta_bias)")
    print("   - Mul (gamma * x)")
    print("   - Add (+ beta)")
    
    print("\n3. Efficiency comparison:")
    print("   RMSNorm: 6 operations")
    print("   FiLM: 6 operations")
    print("   LayerNorm (standard): 8-10 operations")
    print("   BatchNorm: 7-9 operations")


def test_combined_layers():
    """Test combining RMSNorm and FiLM in a single model."""
    
    print("\n=== Testing Combined Custom Operators ===\n")
    
    class CombinedModel(nn.Module):
        def __init__(self, dim=64, condition_dim=32):
            super().__init__()
            self.norm1 = SimpleRMSNorm(dim)
            self.film = SimpleFiLM(dim, condition_dim)
            self.norm2 = SimpleRMSNorm(dim)
            
        def forward(self, x, condition):
            # Typical transformer block pattern
            x = self.norm1(x)
            x = self.film(x, condition)
            x = self.norm2(x)
            return x
    
    # Create and test model
    model = CombinedModel()
    x = torch.randn(2, 10, 64)
    condition = torch.randn(2, 10, 32)
    
    # Test forward pass
    output = model(x, condition)
    print(f"Combined model output shape: {output.shape}")
    
    # Export to ONNX
    torch.onnx.export(
        model,
        (x, condition),
        "combined_custom_ops.onnx",
        input_names=['x', 'condition'],
        output_names=['output'],
        dynamic_axes={
            'x': {0: 'batch', 1: 'sequence'},
            'condition': {0: 'batch', 1: 'sequence'}
        },
        opset_version=13
    )
    print("✓ Exported combined model to ONNX")
    
    # Analyze the combined model
    visualize_onnx_graph("combined_custom_ops.onnx", "Combined RMSNorm + FiLM Model")


if __name__ == "__main__":
    # Visualize individual operators
    print("Analyzing custom ONNX operators...")
    
    # Create simple test models if they don't exist
    if not os.path.exists("rmsnorm_simple.onnx"):
        print("Creating RMSNorm test model...")
        rmsnorm = SimpleRMSNorm(64)
        x = torch.randn(1, 10, 64)
        torch.onnx.export(rmsnorm, x, "rmsnorm_simple.onnx", opset_version=13)
    
    if not os.path.exists("film_simple.onnx"):
        print("Creating FiLM test model...")
        film = SimpleFiLM(64, 32)
        x = torch.randn(1, 10, 64)
        c = torch.randn(1, 10, 32)
        torch.onnx.export(film, (x, c), "film_simple.onnx", opset_version=13)
    
    # Visualize graphs
    visualize_onnx_graph("rmsnorm_simple.onnx", "RMSNorm ONNX Graph")
    visualize_onnx_graph("film_simple.onnx", "FiLM ONNX Graph")
    
    # Analyze breakdown
    analyze_ops_breakdown()
    
    # Test combined layers
    test_combined_layers()
    
    # Create visualization
    create_comparison_plot()
    
    print("\n✓ Analysis complete! Check custom_ops_visualization.png")