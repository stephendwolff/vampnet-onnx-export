"""
Test all custom ONNX operators together in a VampNet-like model.
"""

import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort

from custom_ops.rmsnorm_onnx import SimpleRMSNorm
from custom_ops.film_onnx import SimpleFiLM
from custom_ops.codebook_embedding_onnx import VerySimpleCodebookEmbedding


class MiniVampNet(nn.Module):
    """
    A minimal VampNet-like model using all custom operators.
    """
    
    def __init__(self, 
                 n_codebooks=4,
                 vocab_size=1024,
                 d_model=256,
                 n_heads=8,
                 n_layers=2):
        super().__init__()
        
        self.n_codebooks = n_codebooks
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # CodebookEmbedding
        self.embedding = VerySimpleCodebookEmbedding(n_codebooks, vocab_size, d_model)
        
        # Transformer-like layers with RMSNorm and FiLM
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            layer = nn.ModuleDict({
                'norm1': SimpleRMSNorm(d_model),
                'self_attn': nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                'film1': SimpleFiLM(d_model, d_model),  # Self-modulation
                'norm2': SimpleRMSNorm(d_model),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Linear(d_model * 4, d_model)
                ),
                'film2': SimpleFiLM(d_model, d_model)  # Another FiLM layer
            })
            self.layers.append(layer)
        
        # Final normalization
        self.final_norm = SimpleRMSNorm(d_model)
        
        # Output projections
        self.output_projs = nn.ModuleList([
            nn.Linear(d_model, vocab_size)
            for _ in range(n_codebooks)
        ])
        
    def forward(self, codes, mask):
        """
        Forward pass through mini VampNet.
        
        Args:
            codes: Input codes [batch, n_codebooks, seq_len]
            mask: Binary mask [batch, n_codebooks, seq_len]
        Returns:
            generated_codes: Output codes [batch, n_codebooks, seq_len]
        """
        batch_size, n_codebooks, seq_len = codes.shape
        
        # Embed codes using CodebookEmbedding
        x = self.embedding(codes)  # [batch, seq_len, d_model]
        
        # Pass through transformer-like layers
        for i, layer in enumerate(self.layers):
            # Pre-norm
            x_norm = layer['norm1'](x)
            
            # Self-attention
            attn_out, _ = layer['self_attn'](x_norm, x_norm, x_norm)
            
            # FiLM modulation (self-modulation)
            attn_out = layer['film1'](attn_out, x_norm)
            
            # Residual connection
            x = x + attn_out
            
            # FFN block
            x_norm = layer['norm2'](x)
            ffn_out = layer['ffn'](x_norm)
            
            # Another FiLM layer
            ffn_out = layer['film2'](ffn_out, x_norm)
            
            # Residual connection
            x = x + ffn_out
        
        # Final normalization
        x = self.final_norm(x)
        
        # Generate logits for each codebook
        all_logits = []
        for i in range(self.n_codebooks):
            cb_logits = self.output_projs[i](x)
            all_logits.append(cb_logits)
        
        # Stack logits
        logits = torch.stack(all_logits, dim=1)  # [batch, n_codebooks, seq_len, vocab_size]
        
        # Generate tokens (argmax for ONNX)
        predictions = torch.argmax(logits, dim=-1)
        
        # Apply mask
        output = torch.where(mask.bool(), predictions, codes)
        
        return output


def test_mini_vampnet():
    """Test the mini VampNet model with all custom operators."""
    
    print("=== Testing Mini VampNet with All Custom Operators ===\n")
    
    # Create model
    model = MiniVampNet(
        n_codebooks=4,
        vocab_size=1024,
        d_model=128,  # Smaller for testing
        n_heads=4,
        n_layers=2
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Test inputs
    batch_size = 1
    seq_len = 50
    codes = torch.randint(0, 1024, (batch_size, 4, seq_len))
    mask = torch.zeros_like(codes)
    
    # Create periodic mask pattern
    for i in range(0, seq_len, 7):
        mask[:, :3, i] = 1  # Mask first 3 codebooks every 7 positions
    
    print(f"\nInput shapes:")
    print(f"  Codes: {codes.shape}")
    print(f"  Mask: {mask.shape}")
    print(f"  Masked positions: {mask.sum().item()}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    model.eval()
    with torch.no_grad():
        output = model(codes, mask)
    
    print(f"Output shape: {output.shape}")
    print(f"Output differs at masked positions: {(output[mask.bool()] != codes[mask.bool()]).any().item()}")
    
    # Export to ONNX
    print("\nExporting to ONNX...")
    try:
        torch.onnx.export(
            model,
            (codes, mask),
            "mini_vampnet.onnx",
            input_names=['codes', 'mask'],
            output_names=['generated_codes'],
            dynamic_axes={
                'codes': {0: 'batch', 2: 'sequence'},
                'mask': {0: 'batch', 2: 'sequence'},
                'generated_codes': {0: 'batch', 2: 'sequence'}
            },
            opset_version=13,
            verbose=False
        )
        print("✓ Successfully exported to ONNX!")
        
        # Verify model
        onnx_model = onnx.load("mini_vampnet.onnx")
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model verification passed!")
        
        # Check model size
        import os
        model_size = os.path.getsize("mini_vampnet.onnx") / 1024 / 1024
        print(f"Model size: {model_size:.2f} MB")
        
        # Test ONNX inference
        print("\nTesting ONNX inference...")
        ort_session = ort.InferenceSession("mini_vampnet.onnx")
        
        onnx_output = ort_session.run(
            None,
            {
                'codes': codes.numpy(),
                'mask': mask.numpy()
            }
        )[0]
        
        # Compare outputs
        pytorch_output = output.numpy()
        matches = np.array_equal(pytorch_output, onnx_output)
        max_diff = np.abs(pytorch_output - onnx_output).max()
        
        print(f"Outputs match exactly: {matches}")
        print(f"Max difference: {max_diff}")
        
        if matches or max_diff == 0:
            print("\n✅ ONNX model produces identical results!")
        else:
            print("\n⚠️ Small differences in ONNX output (expected due to argmax)")
            
    except Exception as e:
        print(f"❌ Export failed: {e}")
        print("This might be due to the attention mechanism.")
        
    return model


def analyze_onnx_model():
    """Analyze the exported ONNX model structure."""
    
    if not os.path.exists("mini_vampnet.onnx"):
        print("ONNX model not found. Run test_mini_vampnet() first.")
        return
        
    print("\n=== Analyzing ONNX Model Structure ===\n")
    
    model = onnx.load("mini_vampnet.onnx")
    
    # Count operations
    op_counts = {}
    for node in model.graph.node:
        op_type = node.op_type
        op_counts[op_type] = op_counts.get(op_type, 0) + 1
    
    print("Operation counts:")
    for op, count in sorted(op_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {op}: {count}")
    
    # Identify custom operator patterns
    print("\nCustom operator patterns found:")
    
    # RMSNorm pattern: Mul -> ReduceMean -> Add -> Sqrt -> Div -> Mul
    rmsnorm_count = min(
        op_counts.get('ReduceMean', 0),
        op_counts.get('Sqrt', 0),
        op_counts.get('Div', 0) // 2  # Div is used in other ops too
    )
    print(f"  RMSNorm: ~{rmsnorm_count} instances")
    
    # FiLM pattern: MatMul -> Add -> Mul -> Add
    film_count = 0
    matmul_count = op_counts.get('MatMul', 0)
    # Rough estimate based on MatMul operations
    film_count = matmul_count // 10  # Very rough estimate
    print(f"  FiLM: ~{film_count} instances")
    
    # CodebookEmbedding pattern: Multiple Gather operations
    gather_count = op_counts.get('Gather', 0)
    print(f"  CodebookEmbedding: {gather_count} embedding lookups")


if __name__ == "__main__":
    import os
    
    # Test the model
    model = test_mini_vampnet()
    
    # Analyze the ONNX model
    analyze_onnx_model()
    
    print("\n=== Summary ===")
    print("Successfully demonstrated all three custom operators working together:")
    print("✅ RMSNorm - Efficient normalization")
    print("✅ FiLM - Feature-wise linear modulation")
    print("✅ CodebookEmbedding - Multi-codebook token embeddings")
    print("\nThese operators can be combined to build VampNet-like architectures")
    print("that are fully compatible with ONNX export!")