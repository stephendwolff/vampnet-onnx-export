#!/usr/bin/env python3
"""
ONNX-compatible implementation of VampNet's CodebookEmbedding with correct architecture.
This version avoids conditional logic for better ONNX compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CodebookEmbeddingCorrectV2(nn.Module):
    """
    ONNX-friendly implementation that matches VampNet's architecture:
    - Each codebook has vocab_size x latent_dim embeddings (not d_model!)
    - Embeddings are concatenated to n_codebooks * latent_dim
    - Then projected to d_model with Conv1d
    - Mask tokens are handled by putting them in the embedding table
    """
    
    def __init__(self, n_codebooks: int, vocab_size: int, latent_dim: int, d_model: int):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.d_model = d_model
        
        # Create embedding tables with CORRECT dimensions
        # Include the mask token in the embedding table (vocab_size + 1)
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size + 1, latent_dim)
            for _ in range(n_codebooks)
        ])
        
        # Output projection: (n_codebooks * latent_dim) -> d_model
        # Using Conv1d with kernel_size=1 to match VampNet
        self.out_proj = nn.Conv1d(n_codebooks * latent_dim, d_model, kernel_size=1)
        
    def forward(self, codes):
        """
        Args:
            codes: [batch, n_codebooks, seq_len]
        Returns:
            embeddings: [batch, seq_len, d_model]
        """
        batch_size, n_codebooks, seq_len = codes.shape
        
        # Process each codebook
        embeddings = []
        for i in range(self.n_codebooks):
            # Get codes for this codebook
            cb_codes = codes[:, i, :]  # [batch, seq_len]
            
            # Get embeddings (mask token is at index vocab_size)
            cb_emb = self.embeddings[i](cb_codes)  # [batch, seq_len, latent_dim]
            embeddings.append(cb_emb)
        
        # Concatenate all embeddings
        # [batch, seq_len, n_codebooks * latent_dim]
        embeddings = torch.cat(embeddings, dim=-1)
        
        # Transpose for Conv1d: [batch, n_cb * latent_dim, seq_len]
        embeddings = embeddings.transpose(1, 2)
        
        # Project to d_model
        output = self.out_proj(embeddings)  # [batch, d_model, seq_len]
        
        # Transpose back: [batch, seq_len, d_model]
        output = output.transpose(1, 2)
        
        return output


def transfer_embedding_weights_v2(vampnet_checkpoint, latent_dim=8):
    """
    Transfer weights from VampNet checkpoint to the correct embedding structure.
    """
    print("Transferring embedding weights with CORRECT architecture (V2)...")
    
    # Load VampNet
    from vampnet.modules.transformer import VampNet
    vampnet = VampNet.load(vampnet_checkpoint, map_location='cpu')
    
    # Create correct embedding
    embedding = CodebookEmbeddingCorrectV2(
        n_codebooks=vampnet.n_codebooks,
        vocab_size=vampnet.vocab_size,
        latent_dim=latent_dim,
        d_model=vampnet.embedding_dim
    )
    
    # Transfer embedding weights
    print("\nTransferring embedding tables...")
    
    # Get the checkpoint data
    ckpt = torch.load(vampnet_checkpoint, map_location='cpu')
    
    # Look for codec embeddings in the checkpoint
    if 'codec' in ckpt:
        codec_state = ckpt['codec']
        for i in range(vampnet.n_codebooks):
            key = f'quantizer.quantizers.{i}.codebook.weight'
            if key in codec_state:
                codec_emb = codec_state[key]  # [vocab_size, latent_dim]
                print(f"  Codebook {i}: {codec_emb.shape}")
                # Transfer to our embedding (first vocab_size entries)
                embedding.embeddings[i].weight.data[:vampnet.vocab_size] = codec_emb
    
    # Transfer special token embeddings to the last position
    if hasattr(vampnet.embedding, 'special'):
        print("\nTransferring special token embeddings...")
        if 'MASK' in vampnet.embedding.special:
            mask_emb = vampnet.embedding.special['MASK']  # [n_codebooks, latent_dim]
            for i in range(vampnet.n_codebooks):
                # Put mask embedding at index vocab_size
                embedding.embeddings[i].weight.data[vampnet.vocab_size] = mask_emb[i]
            print(f"  MASK embeddings transferred to index {vampnet.vocab_size}")
    
    # Transfer Conv1d projection weights
    print("\nTransferring output projection...")
    embedding.out_proj.weight.data = vampnet.embedding.out_proj.weight.data
    embedding.out_proj.bias.data = vampnet.embedding.out_proj.bias.data
    print(f"  out_proj: {embedding.out_proj.weight.shape}")
    
    return embedding


if __name__ == "__main__":
    # Test the implementation
    print("Testing ONNX-friendly CodebookEmbedding implementation...")
    
    # Parameters
    n_codebooks = 4
    vocab_size = 1024
    latent_dim = 8
    d_model = 1280
    batch_size = 2
    seq_len = 10
    
    # Create model
    embedding = CodebookEmbeddingCorrectV2(n_codebooks, vocab_size, latent_dim, d_model)
    
    # Test input (mask token is at index 1024)
    codes = torch.randint(0, vocab_size, (batch_size, n_codebooks, seq_len))
    codes[0, 0, 5] = vocab_size  # Add mask token
    
    # Forward pass
    output = embedding(codes)
    
    print(f"\nInput shape: {codes.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: ({batch_size}, {seq_len}, {d_model})")
    
    # Check dimensions
    assert output.shape == (batch_size, seq_len, d_model)
    print("✓ Dimensions correct!")
    
    # Export to ONNX
    print("\nExporting to ONNX...")
    torch.onnx.export(
        embedding,
        codes,
        "codebook_embedding_correct_v2.onnx",
        input_names=['codes'],
        output_names=['embeddings'],
        dynamic_axes={
            'codes': {0: 'batch', 2: 'sequence'},
            'embeddings': {0: 'batch', 1: 'sequence'}
        },
        opset_version=14,
        do_constant_folding=True
    )
    print("✓ Exported successfully!")
    
    # Test ONNX
    import onnxruntime as ort
    session = ort.InferenceSession("codebook_embedding_correct_v2.onnx")
    onnx_output = session.run(None, {'codes': codes.numpy()})[0]
    
    print(f"\nONNX output shape: {onnx_output.shape}")
    print(f"Max difference: {(output.detach().numpy() - onnx_output).max():.8f}")
    print("✓ ONNX verification complete!")