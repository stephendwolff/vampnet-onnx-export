"""
TorchScript utilities for VampNet ONNX export.
"""

import torch
import torch.jit
from typing import Dict, Any, Optional, Tuple
from pathlib import Path


def trace_model(model: torch.nn.Module, 
                example_inputs: Tuple[torch.Tensor, ...],
                check_trace: bool = True) -> torch.jit.ScriptModule:
    """
    Trace a PyTorch model using TorchScript.
    
    Args:
        model: PyTorch model to trace
        example_inputs: Example inputs for tracing
        check_trace: Whether to check trace correctness
        
    Returns:
        Traced ScriptModule
    """
    model.eval()
    
    # Trace the model
    with torch.no_grad():
        traced = torch.jit.trace(model, example_inputs)
    
    if check_trace:
        # Verify the trace
        with torch.no_grad():
            original_output = model(*example_inputs)
            traced_output = traced(*example_inputs)
            
        if isinstance(original_output, torch.Tensor):
            assert torch.allclose(original_output, traced_output, rtol=1e-5), \
                "Traced model output differs from original"
        elif isinstance(original_output, tuple):
            for orig, traced in zip(original_output, traced_output):
                assert torch.allclose(orig, traced, rtol=1e-5), \
                    "Traced model output differs from original"
    
    return traced


def script_model(model: torch.nn.Module) -> torch.jit.ScriptModule:
    """
    Script a PyTorch model using TorchScript.
    
    Args:
        model: PyTorch model to script
        
    Returns:
        Scripted ScriptModule
    """
    model.eval()
    
    # Script the model
    scripted = torch.jit.script(model)
    
    return scripted


def optimize_for_inference(script_module: torch.jit.ScriptModule,
                          example_inputs: Optional[Tuple[torch.Tensor, ...]] = None) -> torch.jit.ScriptModule:
    """
    Optimize a TorchScript module for inference.
    
    Args:
        script_module: TorchScript module to optimize
        example_inputs: Optional example inputs for optimization
        
    Returns:
        Optimized ScriptModule
    """
    # Freeze the module (convert to inference mode)
    frozen = torch.jit.freeze(script_module)
    
    # Optimize for mobile/inference if example inputs provided
    if example_inputs is not None:
        frozen = torch.jit.optimize_for_inference(frozen, example_inputs)
    
    return frozen


def save_torchscript(script_module: torch.jit.ScriptModule,
                    save_path: str,
                    optimize: bool = True,
                    example_inputs: Optional[Tuple[torch.Tensor, ...]] = None) -> None:
    """
    Save a TorchScript module to disk.
    
    Args:
        script_module: TorchScript module to save
        save_path: Path to save the module
        optimize: Whether to optimize before saving
        example_inputs: Example inputs for optimization
    """
    if optimize:
        script_module = optimize_for_inference(script_module, example_inputs)
    
    # Save the module
    torch.jit.save(script_module, save_path)
    print(f"Saved TorchScript module to {save_path}")


def load_torchscript(load_path: str, 
                    map_location: Optional[str] = None) -> torch.jit.ScriptModule:
    """
    Load a TorchScript module from disk.
    
    Args:
        load_path: Path to load the module from
        map_location: Device to load the module to
        
    Returns:
        Loaded ScriptModule
    """
    if map_location is None:
        map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    module = torch.jit.load(load_path, map_location=map_location)
    return module


class TorchScriptWrapper(torch.nn.Module):
    """
    Wrapper to make models more TorchScript-friendly.
    """
    
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        
    def forward(self, *args, **kwargs):
        # TorchScript doesn't support **kwargs well, so convert to positional
        return self.model(*args)


def prepare_vampnet_for_torchscript(interface, component: str = "coarse") -> torch.nn.Module:
    """
    Prepare VampNet component for TorchScript export.
    
    Args:
        interface: VampNet interface
        component: Component to prepare ("coarse", "c2f", "codec")
        
    Returns:
        TorchScript-ready module
    """
    if component == "coarse":
        model = interface.coarse
        
        # Extract the underlying model if it's wrapped
        if hasattr(model, '_orig_mod'):
            model = model._orig_mod
            
        # Create a simplified forward pass
        class CoarseScriptWrapper(torch.nn.Module):
            def __init__(self, coarse_model):
                super().__init__()
                self.model = coarse_model
                self.n_codebooks = coarse_model.n_codebooks
                self.mask_token = coarse_model.mask_token
                
            def forward(self, codes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
                # Get logits
                logits = self.model(codes)
                
                # Simple argmax generation
                predictions = torch.argmax(logits, dim=-1)
                
                # Apply mask
                output = torch.where(mask.bool(), predictions, codes)
                
                return output
                
        return CoarseScriptWrapper(model)
        
    elif component == "c2f":
        if interface.c2f is None:
            raise ValueError("No coarse-to-fine model available")
            
        model = interface.c2f
        if hasattr(model, '_orig_mod'):
            model = model._orig_mod
            
        # C2F wrapper
        class C2FScriptWrapper(torch.nn.Module):
            def __init__(self, c2f_model):
                super().__init__()
                self.model = c2f_model
                
            def forward(self, coarse_codes: torch.Tensor) -> torch.Tensor:
                # Simplified C2F forward
                return self.model(coarse_codes)
                
        return C2FScriptWrapper(model)
        
    elif component == "codec":
        # Codec is more complex, return a simplified version
        class CodecScriptWrapper(torch.nn.Module):
            def __init__(self, codec):
                super().__init__()
                self.sample_rate = codec.sample_rate
                self.hop_length = codec.hop_length
                
            def encode(self, audio: torch.Tensor) -> torch.Tensor:
                # Placeholder - actual codec is too complex for direct export
                seq_len = audio.shape[-1] // self.hop_length
                batch_size = audio.shape[0]
                codes = torch.zeros(batch_size, 14, seq_len, dtype=torch.long)
                return codes
                
            def decode(self, codes: torch.Tensor) -> torch.Tensor:
                # Placeholder
                batch_size = codes.shape[0]
                seq_len = codes.shape[-1]
                samples = seq_len * self.hop_length
                audio = torch.zeros(batch_size, 1, samples)
                return audio
                
        return CodecScriptWrapper(interface.codec)
        
    else:
        raise ValueError(f"Unknown component: {component}")


def export_to_onnx_via_torchscript(
    model: torch.nn.Module,
    example_inputs: Tuple[torch.Tensor, ...],
    onnx_path: str,
    input_names: list,
    output_names: list,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    opset_version: int = 11,
    use_script: bool = False
) -> None:
    """
    Export model to ONNX via TorchScript.
    
    Args:
        model: PyTorch model
        example_inputs: Example inputs
        onnx_path: Path to save ONNX model
        input_names: Input tensor names
        output_names: Output tensor names
        dynamic_axes: Dynamic axes specification
        opset_version: ONNX opset version
        use_script: Use scripting instead of tracing
    """
    model.eval()
    
    # Convert to TorchScript
    if use_script:
        script_module = script_model(model)
    else:
        script_module = trace_model(model, example_inputs)
    
    # Optimize for inference
    script_module = optimize_for_inference(script_module, example_inputs)
    
    # Export to ONNX
    torch.onnx.export(
        script_module,
        example_inputs,
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        export_params=True,
        do_constant_folding=True
    )
    
    print(f"Exported model to ONNX via TorchScript: {onnx_path}")