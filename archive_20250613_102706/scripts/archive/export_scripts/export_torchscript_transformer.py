"""
Export VampNet transformer using TorchScript as intermediate format.
This approach can preserve more of the original model's behavior.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import onnx
import os
import vampnet
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TorchScriptVampNetWrapper(nn.Module):
    """
    TorchScript-compatible wrapper for VampNet transformer.
    Uses type annotations and TorchScript decorators.
    """
    
    def __init__(self, coarse_model):
        super().__init__()
        # Extract the actual model
        if hasattr(coarse_model, '_orig_mod'):
            self.model = coarse_model._orig_mod
        else:
            self.model = coarse_model
            
        self.n_codebooks: int = self.model.n_codebooks
        self.vocab_size: int = 1024
        self.mask_token: int = getattr(self.model, 'mask_token', 1024)
    
    @torch.jit.export
    def forward_logits(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Get raw logits from the model.
        
        Args:
            codes: Input codes [batch, n_codebooks, seq_len]
            
        Returns:
            Logits [batch, n_codebooks, seq_len, vocab_size]
        """
        return self.model(codes)
    
    @torch.jit.export
    def generate_deterministic(self, 
                              codes: torch.Tensor, 
                              mask: torch.Tensor,
                              temperature: float = 1.0) -> torch.Tensor:
        """
        Deterministic generation using argmax.
        
        Args:
            codes: Input codes with mask tokens [batch, n_codebooks, seq_len]
            mask: Binary mask (1=generate, 0=keep) [batch, n_codebooks, seq_len]
            temperature: Temperature for scaling logits
            
        Returns:
            Generated codes [batch, n_codebooks, seq_len]
        """
        # Get logits
        logits = self.forward_logits(codes)
        
        # Apply temperature
        logits = logits / temperature
        
        # Get predictions using argmax
        predictions = torch.argmax(logits, dim=-1)
        
        # Apply mask
        output = torch.where(mask.bool(), predictions, codes)
        
        return output
    
    def forward(self, codes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Main forward method for ONNX export."""
        return self.generate_deterministic(codes, mask, 1.0)


class SimplifiedTorchScriptWrapper(nn.Module):
    """
    Even simpler wrapper that might work better with TorchScript.
    """
    
    def __init__(self, model_forward_func, n_codebooks: int = 4):
        super().__init__()
        self.model_forward = model_forward_func
        self.n_codebooks = n_codebooks
        
    @torch.jit.ignore
    def get_logits(self, codes: torch.Tensor) -> torch.Tensor:
        """This method will be ignored by TorchScript."""
        return self.model_forward(codes)
    
    def forward(self, codes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Simplified forward pass.
        
        Args:
            codes: Input codes [batch, n_codebooks, seq_len]
            mask: Binary mask [batch, n_codebooks, seq_len]
            
        Returns:
            Generated codes [batch, n_codebooks, seq_len]
        """
        # For TorchScript, we need to avoid complex model calls
        # This is a placeholder that shows the structure
        batch_size, n_codebooks, seq_len = codes.shape
        
        # In practice, you'd need to inline the model operations here
        # or use a pre-traced version of the model
        
        # Placeholder: return codes unchanged where not masked
        # and zeros where masked (this won't give good results)
        generated = torch.zeros_like(codes)
        output = torch.where(mask.bool(), generated, codes)
        
        return output


def trace_vampnet_forward():
    """
    Attempt to trace just the forward pass of VampNet.
    """
    print("Loading VampNet models...")
    
    # Load the interface
    interface = vampnet.interface.Interface(
        codec_ckpt="models/vampnet/codec.pth",
        coarse_ckpt="models/vampnet/coarse.pth",
    )
    
    coarse_model = interface.coarse
    device = next(coarse_model.parameters()).device
    
    print(f"Model device: {device}")
    print(f"Model type: {type(coarse_model)}")
    
    # Try to trace the model's forward pass
    print("\nAttempting to trace the model...")
    
    # Create example inputs
    example_codes = torch.randint(0, 1024, (1, 4, 100), device=device)
    
    try:
        # Extract the actual forward function
        if hasattr(coarse_model, '_orig_mod'):
            model = coarse_model._orig_mod
        else:
            model = coarse_model
        
        # Try to trace just the forward pass
        print("Tracing forward pass...")
        traced_forward = torch.jit.trace(model, example_codes)
        
        print("✅ Successfully traced forward pass!")
        
        # Save the traced model
        traced_forward.save("traced_vampnet_forward.pt")
        print("Saved traced forward pass to traced_vampnet_forward.pt")
        
        # Test the traced model
        print("\nTesting traced model...")
        with torch.no_grad():
            original_out = model(example_codes)
            traced_out = traced_forward(example_codes)
            
            max_diff = torch.max(torch.abs(original_out - traced_out)).item()
            print(f"Max difference between original and traced: {max_diff}")
            
            if max_diff < 1e-5:
                print("✅ Traced model matches original!")
            else:
                print("⚠️ Traced model has differences from original")
                
        return traced_forward
        
    except Exception as e:
        print(f"❌ Failed to trace: {e}")
        print("\nTrying alternative approaches...")
        return None


def export_with_torchscript():
    """
    Export VampNet transformer using TorchScript as intermediate.
    """
    print("=== TorchScript Export Approach ===\n")
    
    # First, try to trace the forward pass
    traced_model = trace_vampnet_forward()
    
    if traced_model is None:
        print("\nDirect tracing failed. Trying wrapper approach...")
        
        # Load models again for wrapper
        interface = vampnet.interface.Interface(
            codec_ckpt="models/vampnet/codec.pth",
            coarse_ckpt="models/vampnet/coarse.pth",
        )
        
        coarse_model = interface.coarse
        device = next(coarse_model.parameters()).device
        
        # Try the wrapper approach
        wrapper = TorchScriptVampNetWrapper(coarse_model)
        wrapper.eval()
        wrapper.to(device)
        
        # Create example inputs
        example_codes = torch.randint(0, 1024, (1, 4, 100), device=device)
        example_mask = torch.randint(0, 2, (1, 4, 100), device=device)
        
        try:
            print("\nTracing wrapper model...")
            traced_wrapper = torch.jit.trace(wrapper, (example_codes, example_mask))
            
            print("✅ Successfully traced wrapper!")
            
            # Save TorchScript model
            traced_wrapper.save("traced_vampnet_wrapper.pt")
            print("Saved to traced_vampnet_wrapper.pt")
            
            # Now try to export to ONNX
            print("\nExporting TorchScript model to ONNX...")
            
            # Move to CPU for ONNX export
            traced_wrapper.cpu()
            example_codes = example_codes.cpu()
            example_mask = example_mask.cpu()
            
            torch.onnx.export(
                traced_wrapper,
                (example_codes, example_mask),
                "vampnet_transformer_torchscript.onnx",
                input_names=['codes', 'mask'],
                output_names=['generated_codes'],
                dynamic_axes={
                    'codes': {0: 'batch', 2: 'sequence'},
                    'mask': {0: 'batch', 2: 'sequence'},
                    'generated_codes': {0: 'batch', 2: 'sequence'}
                },
                opset_version=14
            )
            
            print("✅ Successfully exported to ONNX via TorchScript!")
            
            # Verify
            onnx_model = onnx.load("vampnet_transformer_torchscript.onnx")
            onnx.checker.check_model(onnx_model)
            print("✅ ONNX model verification passed!")
            
        except Exception as e:
            print(f"❌ Failed to trace wrapper: {e}")
            
            # Try scripting instead of tracing
            print("\nTrying torch.jit.script instead of trace...")
            try:
                scripted_wrapper = torch.jit.script(wrapper)
                print("✅ Successfully scripted wrapper!")
                
                # Test it
                with torch.no_grad():
                    test_out = scripted_wrapper(example_codes, example_mask)
                    print(f"Scripted output shape: {test_out.shape}")
                    
                scripted_wrapper.save("scripted_vampnet_wrapper.pt")
                print("Saved to scripted_vampnet_wrapper.pt")
                
            except Exception as e2:
                print(f"❌ Failed to script: {e2}")


if __name__ == "__main__":
    export_with_torchscript()