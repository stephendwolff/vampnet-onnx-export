#!/usr/bin/env python3
"""
Fix transformer export with proper dynamic axes.
"""

import os
import torch
import torch.onnx
import onnx
from vampnet_onnx.transformer_wrapper import SimplifiedVampNetModel

def export_transformer_dynamic(output_path: str,
                              n_codebooks: int = 4,
                              vocab_size: int = 1024,
                              d_model: int = 256,
                              n_heads: int = 8,
                              n_layers: int = 4,
                              opset_version: int = 14):
    """
    Export transformer with truly dynamic sequence length.
    """
    print(f"Exporting transformer with dynamic sequence length...")
    
    # Create model
    model = SimplifiedVampNetModel(
        n_codebooks=n_codebooks,
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers
    )
    model.eval()
    
    # Example inputs with different sequence length for testing
    example_codes = torch.randint(0, vocab_size, (1, n_codebooks, 50))
    example_mask = torch.randint(0, 2, (1, n_codebooks, 50))
    
    # Wrapper to fix temperature
    class TransformerWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, codes, mask):
            return self.model(codes, mask, temperature=1.0)
    
    wrapper = TransformerWrapper(model)
    wrapper.eval()
    
    # Export with dynamic axes
    torch.onnx.export(
        wrapper,
        (example_codes, example_mask),
        output_path,
        input_names=['codes', 'mask'],
        output_names=['generated_codes'],
        dynamic_axes={
            'codes': {0: 'batch_size', 2: 'sequence_length'},
            'mask': {0: 'batch_size', 2: 'sequence_length'},
            'generated_codes': {0: 'batch_size', 2: 'sequence_length'}
        },
        opset_version=opset_version,
        export_params=True,
        do_constant_folding=True,
        verbose=False
    )
    
    print(f"Exported to {output_path}")
    
    # Verify and add shape inference
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    # Infer shapes
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    onnx.save(onnx_model, output_path)
    
    print("Model verification passed!")
    
    # Test with different sequence lengths
    import onnxruntime as ort
    
    print("\nTesting with different sequence lengths...")
    session = ort.InferenceSession(output_path)
    
    for seq_len in [25, 50, 100, 200]:
        test_codes = torch.randint(0, vocab_size, (1, n_codebooks, seq_len)).numpy()
        test_mask = torch.randint(0, 2, (1, n_codebooks, seq_len)).numpy()
        
        try:
            outputs = session.run(None, {'codes': test_codes, 'mask': test_mask})
            print(f"  ✓ Sequence length {seq_len}: OK (output shape: {outputs[0].shape})")
        except Exception as e:
            print(f"  ✗ Sequence length {seq_len}: FAILED - {str(e)}")

if __name__ == "__main__":
    # Re-export transformer for test directory
    output_dir = "onnx_models_test"
    os.makedirs(output_dir, exist_ok=True)
    
    export_transformer_dynamic(
        os.path.join(output_dir, "transformer.onnx"),
        n_codebooks=4,
        d_model=256,
        n_layers=4
    )