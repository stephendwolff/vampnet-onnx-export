"""
Direct export of C2F model to ONNX, bypassing forward pass issues.
"""

import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort
from pathlib import Path
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.export_vampnet_transformer_v2 import VampNetTransformerV2


# Create a wrapper that handles C2F properly
class C2FWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
    def forward(self, codes, mask=None):
        """Simplified forward for ONNX export."""
        # Just run through the transformer layers
        batch_size, n_codebooks, seq_len = codes.shape
        
        # Apply mask to create masked codes
        masked_codes = codes.clone()
        if mask is not None:
            masked_codes[mask] = 1024  # mask token
        
        # Embedding
        x = self.base_model.embedding(masked_codes)
        
        # Add positional encoding
        x = x + self.base_model.pos_encoding[:, :seq_len, :]
        
        # Pass through transformer layers
        for layer in self.base_model.layers:
            # Self attention with residual
            x_norm = layer['norm1'](x)
            x_norm = layer['film'](x_norm.permute(0, 2, 1), x_norm.permute(0, 2, 1)).permute(0, 2, 1)
            attn_out = layer['self_attn'](x_norm, x_norm, x_norm, None)
            x = x + attn_out
            
            # FFN with residual
            x_norm = layer['norm2'](x)
            x_norm = layer['film'](x_norm.permute(0, 2, 1), x_norm.permute(0, 2, 1)).permute(0, 2, 1)
            ffn_out = layer['ffn'](x_norm)
            x = x + ffn_out
        
        # Final norm
        x = self.base_model.final_norm(x)
        
        # Generate logits for non-conditioning codebooks
        all_logits = []
        for i in range(len(self.base_model.output_projs)):
            cb_logits = self.base_model.output_projs[i](x)
            all_logits.append(cb_logits)
        
        # Stack and rearrange
        logits = torch.stack(all_logits, dim=1)
        
        # Simple argmax for predictions
        predictions = torch.argmax(logits, dim=-1)
        
        # Create output - keep conditioning, update predictions
        output = codes.clone()
        if mask is not None:
            # Only update masked positions in non-conditioning codebooks
            n_cond = 4  # C2F has 4 conditioning codebooks
            pred_mask = mask[:, n_cond:]
            output[:, n_cond:][pred_mask] = predictions[pred_mask]
        
        return output


def export_c2f_direct():
    """Export C2F model directly."""
    
    print("=== Direct C2F Export to ONNX ===")
    
    # Load the model
    print("\nLoading C2F model...")
    model = VampNetTransformerV2(
        n_codebooks=14,
        n_conditioning_codebooks=4,
        vocab_size=1024,
        d_model=1280,
        n_heads=20,
        n_layers=16,
        use_gated_ffn=True
    )
    
    # Load weights
    print("Loading weights...")
    state_dict = torch.load("models/c2f_complete_v3.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    
    # Wrap the model
    wrapped_model = C2FWrapper(model)
    wrapped_model.eval()
    
    # Create dummy inputs
    batch_size = 1
    seq_len = 100
    dummy_codes = torch.randint(0, 1024, (batch_size, 14, seq_len))
    dummy_mask = torch.zeros(batch_size, 14, seq_len).bool()
    dummy_mask[:, 4:, 40:60] = True  # Mask only non-conditioning
    
    print(f"\nInput shapes:")
    print(f"  codes: {dummy_codes.shape}")
    print(f"  mask: {dummy_mask.shape}")
    
    # Test forward pass
    print("\nTesting wrapped forward pass...")
    with torch.no_grad():
        output = wrapped_model(dummy_codes, dummy_mask)
    print(f"✓ Output shape: {output.shape}")
    
    # Export to ONNX
    output_path = "onnx_models_fixed/c2f_complete_v3.onnx"
    print(f"\nExporting to {output_path}...")
    
    torch.onnx.export(
        wrapped_model,
        (dummy_codes, dummy_mask),
        output_path,
        input_names=['codes', 'mask'],
        output_names=['output'],
        dynamic_axes={
            'codes': {0: 'batch'},
            'mask': {0: 'batch'},
            'output': {0: 'batch'}
        },
        opset_version=13,
        do_constant_folding=True,
        export_params=True,
    )
    
    print(f"✓ Exported successfully!")
    
    # Verify with ONNX Runtime
    print("\nVerifying with ONNX Runtime...")
    ort_session = ort.InferenceSession(output_path)
    
    ort_inputs = {
        'codes': dummy_codes.numpy(),
        'mask': dummy_mask.numpy(),
    }
    
    ort_outputs = ort_session.run(None, ort_inputs)
    ort_output = ort_outputs[0]
    
    print(f"✓ ONNX output shape: {ort_output.shape}")
    
    # Compare outputs
    torch_output = output.numpy()
    max_diff = np.abs(torch_output - ort_output).max()
    print(f"\nMax difference: {max_diff}")
    
    # File size
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Model size: {file_size:.2f} MB")
    
    return ort_session


if __name__ == "__main__":
    export_c2f_direct()