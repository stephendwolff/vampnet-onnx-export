#!/usr/bin/env python3
"""
Step 8: Investigate C2F output format and architecture.
Understanding exactly how VampNet's C2F model outputs its predictions.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from vampnet.modules.transformer import VampNet
from lac.model.lac import LAC as DAC


def investigate_c2f_architecture():
    """Investigate the C2F model's architecture and output format."""
    print("="*80)
    print("STEP 8: INVESTIGATING C2F OUTPUT FORMAT")
    print("="*80)
    
    # Load C2F model
    print("\n1. Loading C2F model...")
    c2f = VampNet.load("models/vampnet/c2f.pth", map_location='cpu')
    c2f.eval()
    
    # Check model configuration
    print(f"\nC2F Configuration:")
    print(f"  n_codebooks: {c2f.n_codebooks}")
    print(f"  n_conditioning_codebooks: {c2f.n_conditioning_codebooks}")
    print(f"  n_predict_codebooks: {c2f.n_predict_codebooks}")
    print(f"  vocab_size: {c2f.vocab_size}")
    print(f"  n_layers: {c2f.n_layers}")
    
    # Examine classifier structure
    print(f"\n2. Classifier structure:")
    print(f"  Type: {type(c2f.classifier)}")
    print(f"  Layers: {c2f.classifier}")
    
    # Get the Conv1d layer
    conv_layer = c2f.classifier.layers[0]
    print(f"\n  Conv1d details:")
    print(f"    In channels: {conv_layer.in_channels}")
    print(f"    Out channels: {conv_layer.out_channels}")
    print(f"    Kernel size: {conv_layer.kernel_size}")
    print(f"    Expected: {c2f.vocab_size} * {c2f.n_predict_codebooks} = {c2f.vocab_size * c2f.n_predict_codebooks}")
    
    # Load codec for latents
    print("\n3. Loading codec...")
    codec = DAC.load(Path("models/vampnet/codec.pth"))
    codec.eval()
    
    # Create test input
    print("\n4. Testing forward pass...")
    batch_size = 1
    seq_len = 100
    
    # Create random codes
    codes = torch.randint(0, 1024, (batch_size, c2f.n_codebooks, seq_len))
    print(f"  Input codes shape: {codes.shape}")
    
    # Convert to latents
    with torch.no_grad():
        # LAC uses a different method
        z = codec.quantizer.from_codes(codes)
        # Flatten the quantized representation
        latents = z.view(batch_size, -1, seq_len)
        print(f"  Latents shape: {latents.shape}")
        
        # Forward pass
        output = c2f(latents)
        print(f"  Raw output shape: {output.shape}")
        
        # The output should be [batch, vocab_size * n_predict_codebooks, seq_len]
        # We need to reshape it to [batch, n_predict_codebooks, seq_len, vocab_size]
        batch, channels, seq = output.shape
        n_pred = c2f.n_predict_codebooks
        vocab = c2f.vocab_size
        
        print(f"\n5. Reshaping output:")
        print(f"  Output channels: {channels}")
        print(f"  Expected: {vocab} * {n_pred} = {vocab * n_pred}")
        print(f"  Match: {channels == vocab * n_pred}")
        
        # Reshape to separate codebooks
        output_reshaped = output.view(batch, n_pred, vocab, seq)
        output_reshaped = output_reshaped.permute(0, 1, 3, 2)  # [batch, n_pred, seq, vocab]
        print(f"  Reshaped output: {output_reshaped.shape}")
        
    # Test the generate method
    print("\n6. Testing generate method...")
    with torch.no_grad():
        # Create mask
        mask = torch.zeros_like(codes)
        mask[:, c2f.n_conditioning_codebooks:, :] = 1  # Mask non-conditioning codebooks
        
        # Generate
        generated = c2f.generate(
            codec=codec,
            time_steps=seq_len,
            start_tokens=codes,
            mask=mask,
            return_signal=False,
            _sampling_steps=2
        )
        print(f"  Generated shape: {generated.shape}")
        
    print("\n7. Analyzing generate method internals...")
    # Let's trace through what happens in generate
    with torch.no_grad():
        # Initial setup
        z = codes
        z_init = z.clone()
        n_infer_codebooks = c2f.n_codebooks - c2f.n_conditioning_codebooks
        
        print(f"  n_infer_codebooks: {n_infer_codebooks}")
        
        # Extract conditioning and non-conditioning
        z_cond = z[:, :c2f.n_conditioning_codebooks, :]
        z_infer = z[:, c2f.n_conditioning_codebooks:, :]
        
        print(f"  z_cond shape: {z_cond.shape}")
        print(f"  z_infer shape: {z_infer.shape}")
        
        # Flatten for processing
        from vampnet.util import codebook_flatten
        z_flat = codebook_flatten(z_infer)
        print(f"  z_flat shape: {z_flat.shape}")
        
        # This flattened tensor is what gets masked and processed
        # The output logits will be [batch, seq * n_infer_codebooks, vocab_size]
        
    print("\n✅ C2F Investigation Complete!")
    print("\nKey findings:")
    print("1. C2F outputs a single flattened tensor with all non-conditioning codebooks")
    print("2. Output shape: [batch, vocab_size * n_predict_codebooks, seq_len]")
    print("3. This needs to be reshaped to [batch, n_predict_codebooks, seq_len, vocab_size]")
    print("4. The generate method uses codebook_flatten/unflatten for processing")
    print("="*80)


def test_c2f_export_fix():
    """Test fixing the C2F export to match VampNet's output format."""
    print("\n\nTESTING C2F EXPORT FIX")
    print("="*80)
    
    from scripts.export_c2f_transformer_complete import C2FTransformerV11
    
    # Create a modified C2F model that outputs the correct format
    class C2FTransformerV11Fixed(C2FTransformerV11):
        def forward(self, latents):
            # Get base output
            x = self.embedding(latents)
            x = rearrange(x, "b d n -> b n d")
            
            # Apply transformer layers
            for layer in self.transformer_layers:
                residual = x
                x = layer['norm'](x)
                x = layer['self_attn'](x)
                x = torch.nn.functional.dropout(x, p=0.1, training=self.training)
                x = residual + x
                x = layer['film'](x)
                residual = x
                x = layer['ffn'](x)
                x = torch.nn.functional.dropout(x, p=0.1, training=self.training)
                x = residual + x
            
            x = self.norm_out(x)
            x = rearrange(x, "b n d -> b d n")
            
            # Single output projection like VampNet
            # Combine all output projections into one
            n_out = len(self.output_projections)
            combined_weight = torch.cat([proj.weight for proj in self.output_projections], dim=0)
            combined_bias = torch.cat([proj.bias for proj in self.output_projections], dim=0)
            
            # Apply combined projection
            x_t = x.transpose(1, 2)  # [batch, seq, d_model]
            output = torch.nn.functional.linear(x_t, combined_weight, combined_bias)  # [batch, seq, vocab * n_out]
            output = output.transpose(1, 2)  # [batch, vocab * n_out, seq]
            
            return output
    
    # Test it
    model = C2FTransformerV11Fixed()
    print(f"\nTesting fixed C2F model...")
    
    with torch.no_grad():
        test_latents = torch.randn(1, 14 * 8, 50)
        output = model(test_latents)
        print(f"Fixed model output shape: {output.shape}")
        print(f"Expected shape: [1, {1024 * 10}, 50] = [1, {1024 * 10}, 50]")
        
    print("="*80)


if __name__ == "__main__":
    investigate_c2f_architecture()
    test_c2f_export_fix()