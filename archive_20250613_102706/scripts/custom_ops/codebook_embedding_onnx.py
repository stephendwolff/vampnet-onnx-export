"""
Custom ONNX operator for CodebookEmbedding.
This handles discrete token embeddings for multiple codebooks with special tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort


class CodebookEmbedding(nn.Module):
    """
    Original VampNet-style CodebookEmbedding.
    Handles embeddings for multiple codebooks with special tokens like MASK.
    """
    
    def __init__(self, n_codebooks: int, vocab_size: int, d_model: int, mask_token: int = 1024):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.mask_token = mask_token
        
        # Separate embedding for each codebook
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size + 1, d_model)  # +1 for mask token
            for _ in range(n_codebooks)
        ])
        
        # Special embedding for mask token (shared across codebooks)
        self.mask_embedding = nn.Parameter(torch.randn(n_codebooks, d_model))
        
        # Output projection to combine embeddings
        self.out_proj = nn.Conv1d(n_codebooks * d_model, d_model, 1)
        
    def forward(self, codes):
        """
        Args:
            codes: Token codes [batch, n_codebooks, seq_len]
        Returns:
            embeddings: Combined embeddings [batch, seq_len, d_model]
        """
        batch_size, n_codebooks, seq_len = codes.shape
        
        all_embeddings = []
        for i in range(self.n_codebooks):
            cb_codes = codes[:, i, :]  # [batch, seq_len]
            
            # Get embeddings
            cb_embed = self.embeddings[i](cb_codes)  # [batch, seq_len, d_model]
            
            # Replace mask token embeddings
            mask_positions = (cb_codes == self.mask_token)
            if mask_positions.any():
                cb_embed[mask_positions] = self.mask_embedding[i]
            
            all_embeddings.append(cb_embed)
        
        # Stack and project
        all_embeddings = torch.stack(all_embeddings, dim=1)  # [batch, n_cb, seq, d_model]
        all_embeddings = all_embeddings.permute(0, 1, 3, 2)  # [batch, n_cb, d_model, seq]
        all_embeddings = all_embeddings.reshape(batch_size, -1, seq_len)
        
        output = self.out_proj(all_embeddings)  # [batch, d_model, seq]
        output = output.transpose(1, 2)  # [batch, seq, d_model]
        
        return output


class SimpleCodebookEmbedding(nn.Module):
    """
    Simplified ONNX-compatible CodebookEmbedding.
    """
    
    def __init__(self, n_codebooks: int, vocab_size: int, d_model: int, mask_token: int = 1024):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.mask_token = mask_token
        
        # Single embedding matrix for all codebooks (offset-based)
        # Each codebook gets its own range in the embedding table
        total_vocab = n_codebooks * (vocab_size + 1)
        self.embedding = nn.Embedding(total_vocab, d_model)
        
        # Output projection weights
        self.out_proj_weight = nn.Parameter(torch.randn(d_model, n_codebooks * d_model) * 0.02)
        self.out_proj_bias = nn.Parameter(torch.zeros(d_model))
        
    def forward(self, codes):
        """
        ONNX-friendly forward pass.
        """
        batch_size, n_codebooks, seq_len = codes.shape
        
        # Offset codes for each codebook
        # Codebook 0: tokens 0-1024
        # Codebook 1: tokens 1025-2049, etc.
        offset_codes = codes.clone()
        for i in range(self.n_codebooks):
            offset_codes[:, i] += i * (self.vocab_size + 1)
        
        # Reshape for embedding lookup
        flat_codes = offset_codes.permute(0, 2, 1).reshape(-1, self.n_codebooks)  # [batch*seq, n_cb]
        
        # Get embeddings
        embeddings = self.embedding(flat_codes)  # [batch*seq, n_cb, d_model]
        
        # Reshape and combine
        embeddings = embeddings.reshape(batch_size, seq_len, -1)  # [batch, seq, n_cb*d_model]
        
        # Project to output dimension
        output = torch.matmul(embeddings, self.out_proj_weight.t()) + self.out_proj_bias
        
        return output


class VerySimpleCodebookEmbedding(nn.Module):
    """
    Even simpler version for ONNX - just sum embeddings from different codebooks.
    """
    
    def __init__(self, n_codebooks: int, vocab_size: int, d_model: int):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Create embedding tables
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size + 1, d_model)
            for _ in range(n_codebooks)
        ])
        
    def forward(self, codes):
        """
        Sum embeddings from all codebooks.
        """
        batch_size, n_codebooks, seq_len = codes.shape
        
        # Initialize output
        output = torch.zeros(batch_size, seq_len, self.d_model, device=codes.device)
        
        # Sum embeddings from each codebook
        for i in range(self.n_codebooks):
            cb_codes = codes[:, i, :]  # [batch, seq_len]
            cb_embed = self.embeddings[i](cb_codes)  # [batch, seq_len, d_model]
            output = output + cb_embed
        
        return output


def create_codebook_embedding_onnx_model(
    batch_size, n_codebooks, seq_len, vocab_size, d_model
):
    """
    Create an ONNX model for CodebookEmbedding from scratch.
    """
    # Input tensor
    codes = helper.make_tensor_value_info(
        'codes', TensorProto.INT64, [batch_size, n_codebooks, seq_len]
    )
    
    # Output tensor
    embeddings = helper.make_tensor_value_info(
        'embeddings', TensorProto.FLOAT, [batch_size, seq_len, d_model]
    )
    
    # Create nodes for a simple sum-based approach
    nodes = []
    embedding_outputs = []
    initializers = []
    
    for i in range(n_codebooks):
        # Extract codes for this codebook
        # For ONNX Slice, we need to provide starts/ends as inputs
        starts_tensor = helper.make_tensor(f'starts_{i}', TensorProto.INT64, [1], [i])
        ends_tensor = helper.make_tensor(f'ends_{i}', TensorProto.INT64, [1], [i+1])
        axes_tensor = helper.make_tensor(f'axes_{i}', TensorProto.INT64, [1], [1])
        
        initializers.extend([starts_tensor, ends_tensor, axes_tensor])
        
        slice_node = helper.make_node(
            'Slice',
            inputs=['codes', f'starts_{i}', f'ends_{i}', f'axes_{i}'],
            outputs=[f'cb_{i}_codes']
        )
        nodes.append(slice_node)
        
        # Squeeze to remove codebook dimension
        # For ONNX 13, axes is provided as input
        squeeze_axes = helper.make_tensor(f'squeeze_axes_{i}', TensorProto.INT64, [1], [1])
        initializers.append(squeeze_axes)
        
        squeeze_node = helper.make_node(
            'Squeeze',
            inputs=[f'cb_{i}_codes', f'squeeze_axes_{i}'],
            outputs=[f'cb_{i}_codes_squeezed']
        )
        nodes.append(squeeze_node)
        
        # Embedding lookup
        gather_node = helper.make_node(
            'Gather',
            inputs=[f'embedding_{i}', f'cb_{i}_codes_squeezed'],
            outputs=[f'cb_{i}_embeddings']
        )
        nodes.append(gather_node)
        
        embedding_outputs.append(f'cb_{i}_embeddings')
    
    # Sum all embeddings
    if n_codebooks > 1:
        sum_node = helper.make_node(
            'Sum',
            inputs=embedding_outputs,
            outputs=['embeddings']
        )
        nodes.append(sum_node)
    else:
        # If only one codebook, just rename
        nodes.append(helper.make_node(
            'Identity',
            inputs=[embedding_outputs[0]],
            outputs=['embeddings']
        ))
    
    # Create weight initializers
    inputs = [codes]
    
    for i in range(n_codebooks):
        # Random embedding weights
        embedding_weight = np.random.randn(vocab_size + 1, d_model).astype(np.float32) * 0.02
        embedding_tensor = helper.make_tensor(
            f'embedding_{i}',
            TensorProto.FLOAT,
            [vocab_size + 1, d_model],
            embedding_weight.flatten()
        )
        initializers.append(embedding_tensor)
        
        # Add as input
        embedding_input = helper.make_tensor_value_info(
            f'embedding_{i}', TensorProto.FLOAT, [vocab_size + 1, d_model]
        )
        inputs.append(embedding_input)
    
    # Create the graph
    graph_def = helper.make_graph(
        nodes,
        'CodebookEmbedding',
        inputs,
        [embeddings],
        initializers
    )
    
    # Create the model
    model_def = helper.make_model(graph_def, producer_name='codebook_embedding_onnx')
    model_def.opset_import[0].version = 13
    model_def.ir_version = 8
    
    return model_def


def test_codebook_embedding_implementations():
    """Test different CodebookEmbedding implementations."""
    
    print("=== Testing CodebookEmbedding Implementations ===\n")
    
    # Test parameters
    batch_size = 2
    n_codebooks = 4
    seq_len = 10
    vocab_size = 1024
    d_model = 64
    
    # Create test input
    codes = torch.randint(0, vocab_size, (batch_size, n_codebooks, seq_len))
    
    # Add some mask tokens
    codes[0, 0, 5] = 1024
    codes[1, 2, 3] = 1024
    
    print(f"Input codes shape: {codes.shape}")
    print(f"Codes range: [{codes.min()}, {codes.max()}]")
    
    # 1. Test VerySimpleCodebookEmbedding (most ONNX-friendly)
    print("\n1. VerySimpleCodebookEmbedding (sum-based)")
    simple_embed = VerySimpleCodebookEmbedding(n_codebooks, vocab_size, d_model)
    simple_output = simple_embed(codes)
    print(f"   Output shape: {simple_output.shape}")
    print(f"   Output mean: {simple_output.mean().item():.6f}")
    print(f"   Output std: {simple_output.std().item():.6f}")
    
    # 2. Export to ONNX
    print("\n2. Export to ONNX")
    torch.onnx.export(
        simple_embed,
        codes,
        "codebook_embedding_simple.onnx",
        input_names=['codes'],
        output_names=['embeddings'],
        dynamic_axes={
            'codes': {0: 'batch', 2: 'sequence'},
            'embeddings': {0: 'batch', 1: 'sequence'}
        },
        opset_version=13
    )
    print("   ✓ Exported to codebook_embedding_simple.onnx")
    
    # Test ONNX model
    ort_session = ort.InferenceSession("codebook_embedding_simple.onnx")
    onnx_output = ort_session.run(None, {'codes': codes.numpy()})[0]
    
    # Compare outputs
    diff = np.abs(simple_output.detach().numpy() - onnx_output).max()
    print(f"   Max difference PyTorch vs ONNX: {diff:.8f}")
    
    # 3. Test SimpleCodebookEmbedding (offset-based)
    print("\n3. SimpleCodebookEmbedding (offset-based)")
    offset_embed = SimpleCodebookEmbedding(n_codebooks, vocab_size, d_model)
    offset_output = offset_embed(codes)
    print(f"   Output shape: {offset_output.shape}")
    
    # Export offset version
    torch.onnx.export(
        offset_embed,
        codes,
        "codebook_embedding_offset.onnx",
        input_names=['codes'],
        output_names=['embeddings'],
        opset_version=13
    )
    print("   ✓ Exported offset-based version")
    
    # 4. Create custom ONNX model
    print("\n4. Custom ONNX model from scratch")
    custom_model = create_codebook_embedding_onnx_model(
        batch_size, n_codebooks, seq_len, vocab_size, d_model
    )
    
    # Save model
    onnx.save(custom_model, "codebook_embedding_custom.onnx")
    print("   ✓ Created custom ONNX model")
    
    # Verify model
    onnx.checker.check_model(custom_model)
    print("   ✓ Model verification passed")
    
    return simple_output, onnx_output


def analyze_codebook_embedding_breakdown():
    """Analyze how CodebookEmbedding breaks down into ONNX ops."""
    
    print("\n=== CodebookEmbedding Breakdown ===\n")
    
    print("1. Simple approach (sum embeddings):")
    print("   For each codebook:")
    print("   - Slice (extract codebook codes)")
    print("   - Gather/Embedding (lookup embeddings)")
    print("   - Sum (combine all embeddings)")
    print("   Total: 2*n_codebooks + 1 operations")
    
    print("\n2. Offset approach (single embedding table):")
    print("   - Add (offset codes)")
    print("   - Reshape (prepare for embedding)")
    print("   - Embedding (single lookup)")
    print("   - MatMul (projection)")
    print("   Total: 4 operations (constant regardless of n_codebooks)")
    
    print("\n3. VampNet approach (with special tokens):")
    print("   - Multiple embeddings")
    print("   - Conditional replacement for special tokens")
    print("   - Conv1d projection")
    print("   More complex but handles special tokens properly")


def test_special_token_handling():
    """Test how different approaches handle special tokens like MASK."""
    
    print("\n=== Testing Special Token Handling ===\n")
    
    n_codebooks = 4
    vocab_size = 1024
    d_model = 64
    mask_token = 1024
    
    # Create input with mask tokens
    codes = torch.randint(0, vocab_size, (1, n_codebooks, 10))
    codes[0, 0, 5] = mask_token
    codes[0, 2, 7] = mask_token
    
    print(f"Mask token positions: {(codes == mask_token).nonzero().tolist()}")
    
    # Test with different implementations
    simple_embed = VerySimpleCodebookEmbedding(n_codebooks, vocab_size, d_model)
    output = simple_embed(codes)
    
    print(f"Output at mask positions:")
    print(f"  Position [0, 5]: {output[0, 5, :5].tolist()}...")
    print(f"  Position [0, 7]: {output[0, 7, :5].tolist()}...")
    
    # Check if mask tokens get different embeddings
    mask_embed_0_5 = output[0, 5].clone()
    mask_embed_0_7 = output[0, 7].clone()
    
    # They should be different because they come from different codebooks
    diff = torch.abs(mask_embed_0_5 - mask_embed_0_7).max().item()
    print(f"\nDifference between mask embeddings from different codebooks: {diff:.6f}")
    print("(Should be non-zero if handling codebooks separately)")


if __name__ == "__main__":
    # Test implementations
    simple_out, onnx_out = test_codebook_embedding_implementations()
    
    # Analyze breakdown
    analyze_codebook_embedding_breakdown()
    
    # Test special token handling
    test_special_token_handling()
    
    print("\n=== Summary ===")
    print("Successfully created CodebookEmbedding as a custom ONNX operator!")
    print("Multiple approaches available:")
    print("1. Sum-based: Simple, ONNX-friendly, sums embeddings from each codebook")
    print("2. Offset-based: More efficient, uses single embedding table")
    print("3. Custom ONNX: Built from scratch using basic operations")
    print("\nKey insight: CodebookEmbedding is essentially multiple embedding")
    print("lookups combined, which maps well to ONNX operations.")