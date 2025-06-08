"""
Export VampNet transformer v2 with ONNX-friendly custom attention.
This version uses custom operators that work properly with dynamic shapes.
"""

import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort
from pathlib import Path
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.custom_ops.rmsnorm_onnx import SimpleRMSNorm
from scripts.custom_ops.film_onnx import SimpleFiLM
from scripts.custom_ops.codebook_embedding_onnx import VerySimpleCodebookEmbedding
from scripts.custom_ops.multihead_attention_onnx import OnnxMultiheadAttention


class GatedFFN(nn.Module):
    """FFN with GatedGELU activation matching VampNet."""
    
    def __init__(self, d_model):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_model * 4)  # 1280 -> 5120
        self.w_2 = nn.Linear(d_model * 2, d_model)  # 2560 -> 1280
        self.activation = nn.GELU()
        
    def forward(self, x):
        # Project to 4*d_model
        hidden = self.w_1(x)
        
        # Split into two halves for gating
        hidden, gate = hidden.chunk(2, dim=-1)  # Each is 2560
        
        # Apply gated activation
        hidden = hidden * self.activation(gate)
        
        # Project back
        return self.w_2(hidden)


class VampNetTransformerV2(nn.Module):
    """
    VampNet transformer with ONNX-friendly implementations.
    Uses custom attention that works with dynamic shapes.
    """
    
    def __init__(self,
                 n_codebooks: int = 4,
                 n_conditioning_codebooks: int = 0,
                 vocab_size: int = 1024,
                 d_model: int = 1280,
                 n_heads: int = 20,
                 n_layers: int = 20,
                 dropout: float = 0.0,
                 mask_token: int = 1024,
                 use_gated_ffn: bool = True):
        super().__init__()
        
        self.n_codebooks = n_codebooks
        self.n_conditioning_codebooks = n_conditioning_codebooks
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.mask_token = mask_token
        self.n_heads = n_heads
        
        # Embedding
        self.embedding = VerySimpleCodebookEmbedding(
            n_codebooks=n_codebooks + n_conditioning_codebooks,
            vocab_size=vocab_size,
            d_model=d_model
        )
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.zeros(1, 2048, d_model))  # Max length 2048
        nn.init.normal_(self.pos_encoding, std=0.02)
        
        # Transformer layers
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            layer = nn.ModuleDict({
                'norm1': SimpleRMSNorm(d_model),
                'self_attn': OnnxMultiheadAttention(d_model, n_heads, dropout),
                'norm2': SimpleRMSNorm(d_model),
                'film': SimpleFiLM(d_model, d_model),
            })
            
            # FFN - use GatedFFN if VampNet uses it
            if use_gated_ffn:
                layer['ffn'] = GatedFFN(d_model)
            else:
                layer['ffn'] = nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Linear(d_model * 4, d_model)
                )
            
            self.layers.append(layer)
        
        # Final norm
        self.final_norm = SimpleRMSNorm(d_model)
        
        # Output projections
        self.output_projs = nn.ModuleList([
            nn.Linear(d_model, vocab_size + 1)  # +1 for mask token
            for _ in range(n_codebooks)
        ])
    
    def forward(self, codes, mask=None, temperature=1.0):
        """
        Forward pass through transformer.
        
        Args:
            codes: [batch, n_codebooks, seq_len]
            mask: [batch, n_codebooks, seq_len] - which positions to generate
            temperature: scalar for sampling
            
        Returns:
            output_codes: [batch, n_codebooks, seq_len]
        """
        batch_size, total_codebooks, seq_len = codes.shape
        
        # Handle conditioning codes if any
        if self.n_conditioning_codebooks > 0:
            cond_codes = codes[:, :self.n_conditioning_codebooks]
            codes = codes[:, self.n_conditioning_codebooks:]
        else:
            cond_codes = None
        
        # Apply mask token where needed
        if mask is not None:
            # Only apply to non-conditioning codes
            mask_to_apply = mask[:, self.n_conditioning_codebooks:] if self.n_conditioning_codebooks > 0 else mask
            masked_codes = codes.clone()
            masked_codes[mask_to_apply] = self.mask_token
            
            # Recombine with conditioning
            if cond_codes is not None:
                masked_codes = torch.cat([cond_codes, masked_codes], dim=1)
        else:
            masked_codes = codes
        
        # Embed
        x = self.embedding(masked_codes)  # [batch, seq_len, d_model]
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Apply transformer layers
        for layer in self.layers:
            # Self-attention with residual
            x_norm = layer['norm1'](x)
            attn_out, _ = layer['self_attn'](x_norm, x_norm, x_norm)
            x = x + attn_out
            
            # FFN with residual
            x_norm = layer['norm2'](x)
            x_norm = layer['film'](x_norm, x_norm)  # Self-modulation
            ffn_out = layer['ffn'](x_norm)
            x = x + ffn_out
        
        # Final norm
        x = self.final_norm(x)
        
        # Generate logits for each codebook
        all_logits = []
        for i in range(self.n_codebooks):
            cb_logits = self.output_projs[i](x)  # [batch, seq_len, vocab_size+1]
            all_logits.append(cb_logits)
        
        # Stack logits
        logits = torch.stack(all_logits, dim=1)  # [batch, n_codebooks, seq_len, vocab_size+1]
        
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature
        
        # Generate tokens
        predictions = torch.argmax(logits, dim=-1)  # [batch, n_codebooks, seq_len]
        
        # Apply mask - only replace masked positions
        if mask is not None:
            mask_to_apply = mask[:, self.n_conditioning_codebooks:] if self.n_conditioning_codebooks > 0 else mask
            output = codes.clone()
            output[mask_to_apply] = predictions[mask_to_apply]
        else:
            output = predictions
        
        # Add conditioning codes back
        if self.n_conditioning_codebooks > 0 and cond_codes is not None:
            output = torch.cat([cond_codes, output], dim=1)
        
        return output


def export_coarse_transformer_v2():
    """Export coarse transformer with ONNX-friendly architecture."""
    
    print("=== VampNet Coarse Transformer V2 Export ===")
    
    # Configuration
    config = {
        'n_codebooks': 4,
        'n_conditioning_codebooks': 0,
        'vocab_size': 1024,
        'd_model': 1280,
        'n_heads': 20,
        'n_layers': 20,
        'use_gated_ffn': True  # VampNet uses GatedGELU
    }
    
    print("\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # Create model
    model = VampNetTransformerV2(**config)
    model.eval()
    
    # Load existing weights if available
    weight_files = [
        "scripts/vampnet_onnx_weights.pth",
        "vampnet_onnx_weights_complete.pth"
    ]
    
    for wf in weight_files:
        if Path(wf).exists():
            print(f"\nLoading weights from {wf}")
            try:
                state_dict = torch.load(wf, map_location='cpu', weights_only=True)
                model.load_state_dict(state_dict, strict=False)
                print("✓ Loaded weights")
                break
            except Exception as e:
                print(f"  Warning: {e}")
    
    print("\nTotal parameters:", sum(p.numel() for p in model.parameters()))
    
    # Test with various shapes
    print("\nTesting shapes...")
    test_shapes = [(1, 173), (2, 256), (1, 512)]
    
    for batch, seq in test_shapes:
        codes = torch.randint(0, 1024, (batch, 4, seq))
        mask = torch.zeros(batch, 4, seq).bool()
        mask[:, :, seq//2:] = True
        temp = torch.tensor(1.0)
        
        with torch.no_grad():
            output = model(codes, mask, temp)
            print(f"  Shape ({batch}, 4, {seq}) -> {output.shape} ✓")
    
    # Export to ONNX
    print("\nExporting to ONNX...")
    output_path = Path("onnx_models_fixed/coarse_transformer_v2.onnx")
    output_path.parent.mkdir(exist_ok=True)
    
    # Dummy inputs
    dummy_codes = torch.randint(0, 1024, (1, 4, 256))
    dummy_mask = torch.zeros(1, 4, 256).bool()
    dummy_temp = torch.tensor(1.0)
    
    torch.onnx.export(
        model,
        (dummy_codes, dummy_mask, dummy_temp),
        str(output_path),
        input_names=['codes', 'mask', 'temperature'],
        output_names=['output'],
        dynamic_axes={
            'codes': {0: 'batch', 2: 'sequence'},
            'mask': {0: 'batch', 2: 'sequence'},
            'output': {0: 'batch', 2: 'sequence'}
        },
        opset_version=14,
        do_constant_folding=True,
        export_params=True,
        verbose=False
    )
    
    print(f"✓ Exported to {output_path}")
    
    # Verify with ONNX Runtime
    print("\nVerifying with ONNX Runtime...")
    ort_session = ort.InferenceSession(str(output_path))
    
    for batch, seq in test_shapes:
        test_codes = np.random.randint(0, 1024, (batch, 4, seq), dtype=np.int64)
        test_mask = np.zeros((batch, 4, seq), dtype=bool)
        test_mask[:, :, seq//2:] = True
        test_temp = np.array(1.0, dtype=np.float32)
        
        try:
            outputs = ort_session.run(
                None,
                {
                    'codes': test_codes,
                    'mask': test_mask,
                    'temperature': test_temp
                }
            )
            print(f"  Shape ({batch}, 4, {seq}) -> {outputs[0].shape} ✓")
        except Exception as e:
            print(f"  Shape ({batch}, 4, {seq}) failed: {e}")
    
    # Save info
    info = {
        'model_type': 'coarse_transformer_v2',
        'description': 'VampNet coarse transformer with ONNX-friendly custom attention',
        **config,
        'improvements': [
            'Custom ONNX-compatible multi-head attention',
            'Proper dynamic shape support',
            'GatedFFN support',
            'Correct 20 attention heads'
        ]
    }
    
    info_path = output_path.with_suffix('.json')
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"\n✓ Info saved to {info_path}")
    print("\n" + "="*60)
    print("Export successful! Use this model in your pipeline:")
    print(f"  coarse_path = '{output_path}'")
    print("="*60)


if __name__ == "__main__":
    export_coarse_transformer_v2()