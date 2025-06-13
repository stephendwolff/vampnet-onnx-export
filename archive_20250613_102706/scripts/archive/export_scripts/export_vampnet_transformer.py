"""
Export VampNet transformer to ONNX using custom operators.
This builds on our custom ops work to create a proper VampNet transformer.
"""

import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort
import vampnet
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.custom_ops.rmsnorm_onnx import SimpleRMSNorm
from scripts.custom_ops.film_onnx import SimpleFiLM
from scripts.custom_ops.codebook_embedding_onnx import VerySimpleCodebookEmbedding


class VampNetTransformerONNX(nn.Module):
    """
    ONNX-compatible VampNet transformer using our custom operators.
    This attempts to match the VampNet architecture while being ONNX-friendly.
    """
    
    def __init__(self,
                 n_codebooks: int = 4,
                 n_conditioning_codebooks: int = 0,
                 vocab_size: int = 1024,
                 d_model: int = 1280,  # VampNet uses 1280
                 n_heads: int = 20,    # VampNet uses 20
                 n_layers: int = 20,   # VampNet uses 20
                 dropout: float = 0.1,
                 mask_token: int = 1024):
        super().__init__()
        
        self.n_codebooks = n_codebooks
        self.n_conditioning_codebooks = n_conditioning_codebooks
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.mask_token = mask_token
        
        # Embedding using our custom CodebookEmbedding
        self.embedding = VerySimpleCodebookEmbedding(
            n_codebooks=n_codebooks + n_conditioning_codebooks,
            vocab_size=vocab_size,
            d_model=d_model
        )
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.zeros(1, 1000, d_model))
        nn.init.normal_(self.pos_encoding, std=0.02)
        
        # Transformer layers
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            layer = nn.ModuleDict({
                'norm1': SimpleRMSNorm(d_model),
                'self_attn': nn.MultiheadAttention(
                    d_model, 
                    n_heads, 
                    dropout=dropout,
                    batch_first=True
                ),
                'dropout1': nn.Dropout(dropout),
                'norm2': SimpleRMSNorm(d_model),
                'film': SimpleFiLM(d_model, d_model),  # VampNet uses FiLM
                'ffn': nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Linear(d_model * 4, d_model)
                ),
                'dropout2': nn.Dropout(dropout)
            })
            self.layers.append(layer)
        
        # Final norm
        self.final_norm = SimpleRMSNorm(d_model)
        
        # Output projections - only for non-conditioning codebooks
        self.output_projs = nn.ModuleList([
            nn.Linear(d_model, vocab_size)
            for _ in range(n_codebooks)
        ])
        
    def forward(self, codes, mask=None, temperature=1.0):
        """
        Forward pass through VampNet transformer.
        
        Args:
            codes: Input codes [batch, n_codebooks, seq_len]
            mask: Optional attention mask
            temperature: Temperature for sampling (fixed at 1.0 for ONNX)
            
        Returns:
            generated_codes: Output codes [batch, n_codebooks, seq_len]
        """
        batch_size, total_codebooks, seq_len = codes.shape
        
        # Split conditioning and non-conditioning codes
        if self.n_conditioning_codebooks > 0:
            cond_codes = codes[:, :self.n_conditioning_codebooks]
            codes = codes[:, self.n_conditioning_codebooks:]
        
        # Embed all codes
        x = self.embedding(codes)  # [batch, seq_len, d_model]
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len]
        
        # Apply transformer layers
        for layer in self.layers:
            # Pre-norm and self-attention
            x_norm = layer['norm1'](x)
            attn_out, _ = layer['self_attn'](x_norm, x_norm, x_norm)
            attn_out = layer['dropout1'](attn_out)
            x = x + attn_out
            
            # Pre-norm and FFN
            x_norm = layer['norm2'](x)
            
            # Apply FiLM (self-modulation in this case)
            x_norm = layer['film'](x_norm, x_norm)
            
            ffn_out = layer['ffn'](x_norm)
            ffn_out = layer['dropout2'](ffn_out)
            x = x + ffn_out
        
        # Final normalization
        x = self.final_norm(x)
        
        # Generate logits for each codebook
        all_logits = []
        for i in range(self.n_codebooks):
            cb_logits = self.output_projs[i](x)  # [batch, seq_len, vocab_size]
            all_logits.append(cb_logits)
        
        # Stack logits
        logits = torch.stack(all_logits, dim=1)  # [batch, n_codebooks, seq_len, vocab_size]
        
        # Apply temperature
        logits = logits / temperature
        
        # Generate tokens (argmax for ONNX)
        predictions = torch.argmax(logits, dim=-1)  # [batch, n_codebooks, seq_len]
        
        # Apply mask if provided
        if mask is not None:
            mask = mask[:, self.n_conditioning_codebooks:]  # Remove conditioning from mask
            output = torch.where(mask.bool(), predictions, codes)
        else:
            output = predictions
        
        # Concatenate conditioning codes back if present
        if self.n_conditioning_codebooks > 0:
            output = torch.cat([cond_codes, output], dim=1)
        
        return output


def extract_vampnet_config(vampnet_model):
    """Extract configuration from a VampNet model."""
    config = {
        'n_codebooks': 4,  # VampNet coarse uses 4
        'vocab_size': 1024,
        'd_model': 1280,   # From the architecture analysis
        'n_heads': 20,
        'n_layers': 20,
        'dropout': 0.1,
    }
    
    # Try to extract actual values from model
    if hasattr(vampnet_model, 'n_codebooks'):
        config['n_codebooks'] = vampnet_model.n_codebooks
    if hasattr(vampnet_model, 'd_model'):
        config['d_model'] = vampnet_model.d_model
    if hasattr(vampnet_model, 'n_heads'):
        config['n_heads'] = vampnet_model.n_heads
    if hasattr(vampnet_model, 'n_layers'):
        config['n_layers'] = vampnet_model.n_layers
        
    return config


def create_weight_mapping(onnx_model, vampnet_model):
    """
    Create a mapping between ONNX model and VampNet model weights.
    This is complex due to different architectures.
    """
    mapping = {}
    
    # This would need to be implemented based on actual model structures
    # For now, we'll use a placeholder
    print("Weight mapping would need to be implemented based on model analysis")
    
    return mapping


def test_vampnet_transformer_export():
    """Test exporting VampNet transformer to ONNX."""
    
    print("=== VampNet Transformer ONNX Export ===\n")
    
    # Load VampNet to get architecture details
    print("Loading VampNet models...")
    interface = vampnet.interface.Interface(
        codec_ckpt="../models/vampnet/codec.pth",
        coarse_ckpt="../models/vampnet/coarse.pth",
        wavebeat_ckpt="../models/vampnet/wavebeat.pth",
    )
    
    coarse_model = interface.coarse
    if hasattr(coarse_model, '_orig_mod'):
        original_model = coarse_model._orig_mod
    else:
        original_model = coarse_model
        
    print(f"Original model type: {type(original_model)}")
    
    # Extract configuration
    config = extract_vampnet_config(original_model)
    print(f"\nExtracted config: {config}")
    
    # Create ONNX-compatible model
    print("\nCreating ONNX-compatible model...")
    onnx_model = VampNetTransformerONNX(**config)
    
    # Count parameters
    onnx_params = sum(p.numel() for p in onnx_model.parameters())
    print(f"ONNX model parameters: {onnx_params:,}")
    
    # Try to count original model parameters
    if hasattr(original_model, 'parameters'):
        original_params = sum(p.numel() for p in original_model.parameters())
        print(f"Original model parameters: {original_params:,}")
    
    # Test with dummy input
    batch_size = 1
    seq_len = 100
    codes = torch.randint(0, 1024, (batch_size, 4, seq_len))
    mask = torch.zeros_like(codes)
    
    # Create some masked positions
    for i in range(0, seq_len, 7):
        mask[:, :, i] = 1
    
    print(f"\nTest input shape: {codes.shape}")
    print(f"Masked positions: {mask.sum().item()}")
    
    # Test forward pass
    onnx_model.eval()
    with torch.no_grad():
        output = onnx_model(codes, mask)
    
    print(f"Output shape: {output.shape}")
    print(f"Output changed at masked positions: {(output[mask.bool()] != codes[mask.bool()]).any().item()}")
    
    # Export to ONNX
    print("\nExporting to ONNX...")
    try:
        # For ONNX export, we need a wrapper that doesn't use optional arguments
        class ONNXWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                
            def forward(self, codes, mask):
                return self.model(codes, mask, temperature=1.0)
        
        wrapper = ONNXWrapper(onnx_model)
        
        torch.onnx.export(
            wrapper,
            (codes, mask),
            "vampnet_transformer.onnx",
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
        
        print("✅ Successfully exported to ONNX!")
        
        # Verify
        onnx_model_proto = onnx.load("vampnet_transformer.onnx")
        onnx.checker.check_model(onnx_model_proto)
        print("✅ ONNX model verification passed!")
        
        # Check size
        model_size = os.path.getsize("vampnet_transformer.onnx") / 1024 / 1024
        print(f"Model size: {model_size:.2f} MB")
        
        # Test ONNX inference
        print("\nTesting ONNX inference...")
        ort_session = ort.InferenceSession("vampnet_transformer.onnx")
        
        onnx_outputs = ort_session.run(
            None,
            {
                'codes': codes.numpy(),
                'mask': mask.numpy()
            }
        )
        
        onnx_output = onnx_outputs[0]
        pytorch_output = output.numpy()
        
        # Compare
        matches = np.array_equal(pytorch_output, onnx_output)
        print(f"Outputs match: {matches}")
        
        if matches:
            print("\n✅ ONNX model produces identical results!")
        else:
            max_diff = np.abs(pytorch_output - onnx_output).max()
            print(f"\n⚠️ Max difference: {max_diff}")
            
    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        
    return onnx_model


def analyze_weight_transfer_options():
    """Analyze options for transferring pretrained weights."""
    
    print("\n=== Weight Transfer Analysis ===\n")
    
    print("Options for transferring VampNet weights to ONNX model:")
    print("\n1. Direct mapping (if architectures match):")
    print("   - Map embedding weights")
    print("   - Map RMSNorm weights")
    print("   - Map attention weights")
    print("   - Map FFN weights")
    print("   - Map output projection weights")
    
    print("\n2. Layer-by-layer extraction:")
    print("   - Extract each transformer layer separately")
    print("   - Convert custom layers to our implementations")
    print("   - Handle any architectural differences")
    
    print("\n3. Knowledge distillation:")
    print("   - Use original model as teacher")
    print("   - Train ONNX model to match outputs")
    print("   - Preserves behavior even with architectural differences")
    
    print("\n4. Checkpoint conversion:")
    print("   - Load VampNet checkpoint")
    print("   - Create mapping dictionary")
    print("   - Save in ONNX-compatible format")


if __name__ == "__main__":
    # Test the export
    model = test_vampnet_transformer_export()
    
    # Analyze weight transfer options
    analyze_weight_transfer_options()
    
    print("\n=== Next Steps ===")
    print("1. Implement weight extraction from pretrained VampNet")
    print("2. Create proper weight mapping between architectures")
    print("3. Test with pretrained weights")
    print("4. Optimize for inference performance")