"""
Export VampNet transformer V3 with proper sampling instead of argmax.
This version exports models that use sampling for better audio quality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import onnxruntime as ort
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.export_vampnet_transformer_v2 import VampNetTransformerV2


class VampNetTransformerV3Sampling(VampNetTransformerV2):
    """
    Version 3 that returns logits instead of argmax tokens.
    This allows proper sampling to be done after ONNX inference.
    """
    
    def forward(self, codes, mask, temperature=None):
        """
        Forward pass that returns logits for proper sampling.
        
        Args:
            codes: Input codes [batch, n_codebooks, seq_len]
            mask: Boolean mask [batch, n_codebooks, seq_len]
            temperature: Not used in this version (for ONNX compatibility)
            
        Returns:
            logits: Raw logits [batch, n_codebooks, seq_len, vocab_size+1]
        """
        batch_size, n_codebooks, seq_len = codes.shape
        
        # For C2F models, split conditioning and generation codes
        if self.n_conditioning_codebooks > 0:
            cond_codes = codes[:, :self.n_conditioning_codebooks]
            gen_codes = codes[:, self.n_conditioning_codebooks:]
        else:
            cond_codes = None
            gen_codes = codes
        
        # Apply mask to generation codes
        masked_codes = gen_codes.clone()
        if mask is not None:
            # Only apply mask to generation codebooks
            if self.n_conditioning_codebooks > 0:
                gen_mask = mask[:, self.n_conditioning_codebooks:]
                masked_codes[gen_mask] = self.mask_token
            else:
                masked_codes[mask] = self.mask_token
        
        # Combine conditioning and masked generation codes
        if self.n_conditioning_codebooks > 0:
            if cond_codes is not None:
                masked_codes = codes
            else:
                masked_codes = gen_codes
        
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
        
        # Generate logits for each non-conditioning codebook
        all_logits = []
        n_output_codebooks = self.n_codebooks - self.n_conditioning_codebooks
        for i in range(n_output_codebooks):
            cb_logits = self.output_projs[i](x)  # [batch, seq_len, vocab_size+1]
            all_logits.append(cb_logits)
        
        # Stack logits
        logits = torch.stack(all_logits, dim=1)  # [batch, n_output_codebooks, seq_len, vocab_size+1]
        
        # Rearrange to [batch, n_codebooks, seq_len, vocab_size+1]
        # Pad with zeros for conditioning codebooks if needed
        if self.n_conditioning_codebooks > 0:
            # Create full logits tensor
            full_logits = torch.zeros(
                batch_size, n_codebooks, seq_len, self.vocab_size + 1,
                device=logits.device, dtype=logits.dtype
            )
            # Fill in the generation codebooks
            full_logits[:, self.n_conditioning_codebooks:] = logits
            logits = full_logits
        
        return logits


class VampNetTransformerV3SamplingC2F(VampNetTransformerV3Sampling):
    """
    C2F version that only accepts codes and mask (no temperature).
    """
    
    def forward(self, codes, mask):
        # Call parent with temperature=None
        # Ensure mask is used in a way that prevents constant folding
        mask = mask.bool()  # Ensure it's a boolean tensor
        return super().forward(codes, mask, temperature=None)


class SamplingWrapper(nn.Module):
    """
    Wrapper that takes ONNX logits output and applies sampling.
    Use this for inference after getting logits from ONNX.
    """
    
    def __init__(self, n_conditioning_codebooks=0):
        super().__init__()
        self.n_conditioning_codebooks = n_conditioning_codebooks
    
    def forward(self, codes, mask, logits, temperature=0.8, top_p=0.9):
        """Apply temperature and top-p sampling to logits."""
        # Apply temperature
        if temperature > 0:
            logits = logits / temperature
        
        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Apply top-p (nucleus) sampling
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # Find cutoff
            cutoff_mask = cumsum_probs > top_p
            cutoff_mask[..., 0] = False  # Keep at least one token
            
            # Zero out probabilities beyond cutoff
            sorted_probs[cutoff_mask] = 0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdims=True)
            
            # Create a new probs tensor with filtered probabilities
            probs = torch.zeros_like(probs)
            probs.scatter_(-1, sorted_indices, sorted_probs)
        
        # Sample from distribution
        batch, n_codebooks, seq_len, vocab_size = probs.shape
        probs_flat = probs.view(-1, vocab_size)
        sampled_flat = torch.multinomial(probs_flat, num_samples=1)
        sampled = sampled_flat.view(batch, n_codebooks, seq_len)
        
        # Apply mask - only use sampled tokens at masked positions
        output = codes.clone()
        if mask is not None:
            if self.n_conditioning_codebooks > 0:
                # Only update generation codebooks
                gen_mask = mask[:, self.n_conditioning_codebooks:]
                output[:, self.n_conditioning_codebooks:][gen_mask] = sampled[:, self.n_conditioning_codebooks:][gen_mask]
            else:
                output[mask] = sampled[mask]
        
        return output


def export_model_with_proper_sampling(
    model_type="coarse",
    weights_path=None,
    output_path=None
):
    """Export model that returns logits for proper sampling."""
    
    print(f"\n{'='*60}")
    print(f"Exporting {model_type.upper()} Model V3 with Proper Sampling")
    print(f"{'='*60}")
    
    # Configuration
    if model_type == "coarse":
        config = {
            'n_codebooks': 4,
            'n_conditioning_codebooks': 0,
            'vocab_size': 1024,
            'd_model': 1280,
            'n_heads': 20,
            'n_layers': 20,
            'use_gated_ffn': True
        }
        default_weights = "models/coarse_complete_v3.pth"
        default_output = "onnx_models_fixed/coarse_logits_v3.onnx"
    else:  # c2f
        config = {
            'n_codebooks': 14,
            'n_conditioning_codebooks': 4,
            'vocab_size': 1024,
            'd_model': 1280,
            'n_heads': 20,
            'n_layers': 20,
            'use_gated_ffn': True
        }
        default_weights = "models/c2f_complete_v3.pth"
        default_output = "onnx_models_fixed/c2f_logits_v3.onnx"
    
    weights_path = weights_path or default_weights
    output_path = output_path or default_output
    
    # Create model - use C2F-specific class for c2f
    if model_type == "c2f":
        model = VampNetTransformerV3SamplingC2F(**config)
    else:
        model = VampNetTransformerV3Sampling(**config)
    model.eval()
    
    # Load weights
    if Path(weights_path).exists():
        print(f"\nLoading weights from {weights_path}")
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        print("✓ Weights loaded")
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 1
    seq_len = 100
    codes = torch.randint(0, 1024, (batch_size, config['n_codebooks'], seq_len))
    # Create a more complex mask pattern to prevent constant folding
    mask = torch.zeros((batch_size, config['n_codebooks'], seq_len), dtype=torch.bool)
    # Random mask pattern
    if model_type == "c2f":
        # For C2F, mask only the fine codebooks (4-13)
        mask[:, 4:, 50:60] = True
        mask[:, 6:8, 20:30] = True
        mask[:, 10:12, 70:80] = True
    else:
        mask[:, :, 50:60] = True
        mask[:, 1:3, 20:30] = True
    
    with torch.no_grad():
        logits = model(codes, mask)
    
    print(f"Logits shape: {logits.shape}")
    print(f"Expected: ({batch_size}, {config['n_codebooks']}, {seq_len}, {config['vocab_size']+1})")
    
    # Export to ONNX
    print(f"\nExporting to {output_path}")
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    
    # Note: temperature parameter is removed for ONNX
    # For C2F model, we don't pass temperature
    if model_type == "c2f":
        torch.onnx.export(
            model,
            (codes, mask),  # No temperature
            output_path,
            input_names=['codes', 'mask'],
            output_names=['logits'],
            dynamic_axes={
                'codes': {0: 'batch', 2: 'seq_len'},
                'mask': {0: 'batch', 2: 'seq_len'},
                'logits': {0: 'batch', 2: 'seq_len'}
            },
            opset_version=14,
            do_constant_folding=False  # Disable for C2F to preserve mask input
        )
    else:
        torch.onnx.export(
            model,
            (codes, mask, None),  # temperature=None
            output_path,
            input_names=['codes', 'mask'],
            output_names=['logits'],
            dynamic_axes={
                'codes': {0: 'batch', 2: 'seq_len'},
                'mask': {0: 'batch', 2: 'seq_len'},
                'logits': {0: 'batch', 2: 'seq_len'}
            },
            opset_version=14,
            do_constant_folding=True
        )
    
    print(f"✓ Exported to {output_path}")
    
    # Verify
    print("\nVerifying ONNX model...")
    session = ort.InferenceSession(str(output_path))
    
    # Check what inputs the model actually expects
    input_names = [inp.name for inp in session.get_inputs()]
    print(f"Model expects inputs: {input_names}")
    
    # Prepare inputs based on what the model expects
    ort_inputs = {'codes': codes.numpy().astype(np.int64)}
    if 'mask' in input_names:
        ort_inputs['mask'] = mask.numpy()
    
    outputs = session.run(None, ort_inputs)
    
    onnx_logits = outputs[0]
    print(f"ONNX output shape: {onnx_logits.shape}")
    
    return output_path


def sample_from_onnx_output(codes, mask, logits, temperature=0.8, top_p=0.9, n_conditioning_codebooks=0):
    """
    Apply proper sampling to ONNX logits output.
    
    Args:
        codes: Original input codes
        mask: Mask indicating positions to generate
        logits: Raw logits from ONNX model
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        n_conditioning_codebooks: Number of conditioning codebooks (for C2F)
    
    Returns:
        Generated codes with proper sampling applied
    """
    # Convert to torch for easier manipulation
    codes_t = torch.from_numpy(codes) if isinstance(codes, np.ndarray) else codes
    mask_t = torch.from_numpy(mask) if isinstance(mask, np.ndarray) else mask
    logits_t = torch.from_numpy(logits) if isinstance(logits, np.ndarray) else logits
    
    # Create sampling wrapper and apply
    sampler = SamplingWrapper(n_conditioning_codebooks)
    with torch.no_grad():
        output = sampler(codes_t, mask_t, logits_t, temperature, top_p)
    
    # Convert back to numpy if needed
    if isinstance(codes, np.ndarray):
        output = output.numpy()
    
    return output


if __name__ == "__main__":
    # Export both models
    export_model_with_proper_sampling("coarse")
    export_model_with_proper_sampling("c2f")
    
    print("\n" + "="*60)
    print("USAGE EXAMPLE:")
    print("="*60)
    print("""
# After ONNX inference:
import onnxruntime as ort
from export_vampnet_transformer_v3_sampling import sample_from_onnx_output

# Load model
session = ort.InferenceSession("onnx_models_fixed/coarse_logits_v3.onnx")

# Run inference to get logits
logits = session.run(None, {'codes': codes, 'mask': mask})[0]

# Apply proper sampling (not ArgMax!)
output_codes = sample_from_onnx_output(
    codes, mask, logits, 
    temperature=0.8, 
    top_p=0.9
)
""")