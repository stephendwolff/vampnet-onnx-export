#!/usr/bin/env python3
"""
Export VampNet coarse transformer to ONNX with logits output (not sampled codes).
This allows proper sampling to be implemented post-inference.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from vampnet.modules.transformer import VampNet
from vampnet_onnx.models import CoarseTransformer


class CoarseTransformerLogits(nn.Module):
    """
    Wrapper that exports logits instead of sampled codes.
    This allows proper temperature and top-p sampling after ONNX inference.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.n_codebooks = model.n_codebooks
        self.vocab_size = model.vocab_size
        self.mask_token = model.mask_token
        
    def forward(self, codes, mask):
        # Get the model's internal representation
        # We need to extract logits before sampling
        
        # Prepare inputs
        x = codes
        mask = mask.bool()
        
        # Apply mask
        x_masked = x.clone()
        x_masked[mask] = self.mask_token
        
        # Get embeddings
        x_emb = self.model.embedding.from_codes(x_masked, self.model.codec)
        
        # Add positional encoding
        x_emb = self.model.embedding.add_positional_encoding(x_emb)
        
        # Run through transformer
        x_out = self.model.transformer(x_emb, None)
        
        # Get logits from output projection
        logits = self.model.classifier(x_out)
        
        # Reshape to (batch, n_codebooks, seq_len, vocab_size)
        batch, seq_len, _ = logits.shape
        logits = logits.view(batch, seq_len, self.n_codebooks, self.vocab_size)
        logits = logits.permute(0, 2, 1, 3)  # (batch, n_codebooks, seq_len, vocab_size)
        
        return logits


def export_coarse_with_logits(checkpoint_path, output_path, device='cpu'):
    """Export coarse model that returns logits for proper sampling."""
    
    print(f"Loading checkpoint from {checkpoint_path}")
    
    # Load the model
    model = VampNet.load(checkpoint_path, map_location=device)
    model.eval()
    model.to(device)
    
    # Wrap in logits extractor
    logits_model = CoarseTransformerLogits(model)
    
    # Test input
    batch_size = 1
    n_codebooks = 4
    seq_len = 100
    
    codes = torch.randint(0, 1024, (batch_size, n_codebooks, seq_len))
    mask = torch.zeros((batch_size, n_codebooks, seq_len), dtype=torch.bool)
    mask[:, :, 50:60] = True  # Mask some positions
    
    # Test forward pass
    print("Testing forward pass...")
    with torch.no_grad():
        logits = logits_model(codes, mask)
    
    print(f"Logits shape: {logits.shape}")
    print(f"Expected: ({batch_size}, {n_codebooks}, {seq_len}, {model.vocab_size})")
    
    # Export to ONNX
    print(f"\nExporting to {output_path}")
    
    torch.onnx.export(
        logits_model,
        (codes, mask),
        output_path,
        input_names=['codes', 'mask'],
        output_names=['logits'],
        dynamic_axes={
            'codes': {0: 'batch', 2: 'seq_len'},
            'mask': {0: 'batch', 2: 'seq_len'},
            'logits': {0: 'batch', 2: 'seq_len'}
        },
        opset_version=14,
        do_constant_folding=True,
        export_params=True,
    )
    
    print(f"✓ Exported to {output_path}")
    
    # Verify the export
    import onnxruntime as ort
    
    print("\nVerifying ONNX model...")
    session = ort.InferenceSession(str(output_path))
    
    # Test inference
    outputs = session.run(None, {
        'codes': codes.numpy().astype(np.int64),
        'mask': mask.numpy()
    })
    
    onnx_logits = outputs[0]
    print(f"ONNX output shape: {onnx_logits.shape}")
    print(f"Output range: [{onnx_logits.min():.2f}, {onnx_logits.max():.2f}]")
    
    return output_path


def sample_from_logits(logits, temperature=1.0, top_p=0.9, top_k=None):
    """
    Sample from logits with temperature and top-p sampling.
    This is what you'll use after ONNX inference.
    """
    # Apply temperature
    if temperature > 0:
        logits = logits / temperature
    
    # Convert to probabilities
    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    
    # Apply top-p (nucleus) sampling
    if top_p < 1.0:
        sorted_indices = np.argsort(probs, axis=-1)[..., ::-1]
        sorted_probs = np.take_along_axis(probs, sorted_indices, axis=-1)
        cumsum_probs = np.cumsum(sorted_probs, axis=-1)
        
        # Find cutoff
        cutoff_mask = cumsum_probs > top_p
        cutoff_mask[..., 0] = False  # Keep at least one token
        
        # Zero out probabilities beyond cutoff
        sorted_probs[cutoff_mask] = 0
        sorted_probs = sorted_probs / sorted_probs.sum(axis=-1, keepdims=True)
        
        # Restore original order
        inv_indices = np.argsort(sorted_indices, axis=-1)
        probs = np.take_along_axis(sorted_probs, inv_indices, axis=-1)
    
    # Sample
    batch, n_codebooks, seq_len, vocab_size = probs.shape
    codes = np.zeros((batch, n_codebooks, seq_len), dtype=np.int64)
    
    for b in range(batch):
        for c in range(n_codebooks):
            for t in range(seq_len):
                codes[b, c, t] = np.random.choice(vocab_size, p=probs[b, c, t])
    
    return codes


if __name__ == "__main__":
    # Export coarse model with logits
    checkpoint_path = Path("../models/vampnet/coarse.pth")
    output_path = Path("../onnx_models_fixed/coarse_logits.onnx")
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    export_coarse_with_logits(checkpoint_path, output_path)
    
    print("\n" + "="*60)
    print("IMPORTANT: Use this model with proper sampling:")
    print("="*60)
    print("""
# After ONNX inference:
logits = session.run(None, {'codes': codes, 'mask': mask})[0]

# Apply sampling (don't use ArgMax!)
from export_coarse_with_logits import sample_from_logits
codes = sample_from_logits(logits, temperature=0.8, top_p=0.9)
""")