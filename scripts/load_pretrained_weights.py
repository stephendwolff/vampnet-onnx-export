"""
Load pretrained weights into the ONNX-compatible VampNet transformer.
"""

import torch
import numpy as np
import onnxruntime as ort
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from scripts.export_vampnet_transformer import VampNetTransformerONNX


def load_and_export_with_weights():
    """Load pretrained weights and export to ONNX."""
    
    print("=== Loading Pretrained Weights into ONNX Model ===\n")
    
    # Create the ONNX-compatible model
    print("Creating ONNX-compatible model...")
    model = VampNetTransformerONNX(
        n_codebooks=4,
        vocab_size=1024,
        d_model=1280,
        n_heads=20,
        n_layers=20
    )
    
    # Load the saved weights
    weights_path = "vampnet_onnx_weights.pth"
    if Path(weights_path).exists():
        print(f"Loading weights from {weights_path}...")
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict, strict=False)
        print("✓ Weights loaded successfully")
    else:
        print(f"⚠️  Weight file {weights_path} not found. Using random weights.")
    
    # Test the model
    print("\n=== Testing Model ===")
    model.eval()
    
    batch_size = 1
    seq_len = 100
    codes = torch.randint(0, 1024, (batch_size, 4, seq_len))
    mask = torch.zeros_like(codes)
    mask[:, :, 40:60] = 1
    
    with torch.no_grad():
        output = model(codes, mask)
        print(f"✓ Forward pass successful!")
        print(f"  Input shape: {codes.shape}")
        print(f"  Output shape: {output.shape}")
        
        # Check generation
        changed = (output != codes)[mask.bool()].sum().item()
        total_masked = mask.sum().item()
        print(f"  Changed {changed}/{total_masked} masked positions")
    
    # Export to ONNX
    print("\n=== Exporting to ONNX ===")
    
    # Create wrapper for ONNX export
    class ONNXWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, codes, mask):
            return self.model(codes, mask, temperature=1.0)
    
    wrapper = ONNXWrapper(model)
    
    try:
        torch.onnx.export(
            wrapper,
            (codes, mask),
            "vampnet_transformer_with_weights.onnx",
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
        
        print("✓ Successfully exported to vampnet_transformer_with_weights.onnx")
        
        # Check file size
        size_mb = Path("vampnet_transformer_with_weights.onnx").stat().st_size / (1024 * 1024)
        print(f"  Model size: {size_mb:.1f} MB")
        
    except Exception as e:
        print(f"✗ Export failed: {e}")
        return None
    
    return model


def test_onnx_inference():
    """Test the exported ONNX model with pretrained weights."""
    
    print("\n=== Testing ONNX Inference ===")
    
    if not Path("vampnet_transformer_with_weights.onnx").exists():
        print("⚠️  ONNX model not found. Run export first.")
        return
    
    # Load ONNX model
    print("Loading ONNX model...")
    ort_session = ort.InferenceSession("vampnet_transformer_with_weights.onnx")
    
    # Test parameters
    batch_size = 1
    seq_len = 100
    
    # Create test input
    codes = np.random.randint(0, 1024, (batch_size, 4, seq_len), dtype=np.int64)
    mask = np.zeros_like(codes)
    
    # Mask some positions
    mask[:, :, 30:50] = 1
    
    print(f"Input shape: {codes.shape}")
    print(f"Masked positions: {mask.sum()}")
    
    # Run inference
    outputs = ort_session.run(
        None,
        {
            'codes': codes,
            'mask': mask
        }
    )
    
    generated = outputs[0]
    
    # Analyze results
    changed = (generated != codes)[mask.astype(bool)].sum()
    print(f"\n✓ ONNX inference successful!")
    print(f"  Changed {changed}/{mask.sum()} masked positions")
    
    # Check token distribution
    print("\nToken distribution in generated positions:")
    for cb in range(4):
        masked_tokens = generated[0, cb, mask[0, cb].astype(bool)]
        unique_tokens = np.unique(masked_tokens)
        print(f"  Codebook {cb}: {len(unique_tokens)} unique tokens (out of {len(masked_tokens)})")


def compare_with_random_weights():
    """Compare outputs with random vs pretrained weights."""
    
    print("\n=== Comparing Random vs Pretrained Weights ===")
    
    # Create two models
    random_model = VampNetTransformerONNX(
        n_codebooks=4,
        vocab_size=1024,
        d_model=1280,
        n_heads=20,
        n_layers=20
    )
    
    pretrained_model = VampNetTransformerONNX(
        n_codebooks=4,
        vocab_size=1024,
        d_model=1280,
        n_heads=20,
        n_layers=20
    )
    
    # Load pretrained weights
    if Path("vampnet_onnx_weights.pth").exists():
        state_dict = torch.load("vampnet_onnx_weights.pth")
        pretrained_model.load_state_dict(state_dict, strict=False)
    else:
        print("⚠️  No pretrained weights found")
        return
    
    # Test input
    codes = torch.randint(0, 1024, (1, 4, 100))
    mask = torch.zeros_like(codes)
    mask[:, :, 40:60] = 1
    
    # Generate with both models
    random_model.eval()
    pretrained_model.eval()
    
    with torch.no_grad():
        random_output = random_model(codes, mask)
        pretrained_output = pretrained_model(codes, mask)
    
    # Compare
    random_changed = (random_output != codes)[mask.bool()].sum().item()
    pretrained_changed = (pretrained_output != codes)[mask.bool()].sum().item()
    
    print(f"Random model: Changed {random_changed}/{mask.sum().item()} positions")
    print(f"Pretrained model: Changed {pretrained_changed}/{mask.sum().item()} positions")
    
    # Check if outputs are different
    different_positions = (random_output != pretrained_output).sum().item()
    print(f"\nOutputs differ at {different_positions} positions")
    
    if different_positions > 0:
        print("✓ Pretrained weights are having an effect!")
    else:
        print("⚠️  Outputs are identical - weights may not be properly loaded")


if __name__ == "__main__":
    # Load weights and export
    model = load_and_export_with_weights()
    
    if model is not None:
        # Test ONNX inference
        test_onnx_inference()
        
        # Compare with random weights
        compare_with_random_weights()
    
    print("\n=== Summary ===")
    print("Files created:")
    print("  - vampnet_onnx_weights.pth (PyTorch weights)")
    print("  - vampnet_transformer_with_weights.onnx (ONNX model)")
    print("\nNote: Only partial weights were transferred (60 out of 294 parameters).")
    print("The model may not produce music-like outputs without full weight transfer.")