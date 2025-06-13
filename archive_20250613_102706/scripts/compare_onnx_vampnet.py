"""
Compare ONNX model output with original VampNet.
"""

import torch
import numpy as np
import onnxruntime as ort
from pathlib import Path
import matplotlib.pyplot as plt


def compare_models():
    """Compare ONNX and VampNet outputs."""
    
    print("=== Comparing ONNX vs VampNet ===")
    
    # Load VampNet
    try:
        from vampnet import interface as vampnet
        
        print("\nLoading VampNet...")
        interface = vampnet.Interface(
            coarse_ckpt="models/vampnet/coarse.pth",
            coarse2fine_ckpt=None,
            codec_ckpt="models/vampnet/codec.pth",
            device="cpu"
        )
        vampnet_available = True
    except Exception as e:
        print(f"VampNet not available: {e}")
        vampnet_available = False
        return
    
    # Load ONNX model
    print("\nLoading ONNX model...")
    ort_session = ort.InferenceSession("onnx_models_fixed/coarse_complete_v3.onnx")
    
    # Create test input
    batch_size = 1
    n_codebooks = 4
    seq_len = 100
    
    # Create structured input (not random)
    print("\nCreating test input...")
    codes = torch.zeros(batch_size, n_codebooks, seq_len, dtype=torch.long)
    
    # Create a pattern
    for i in range(n_codebooks):
        for j in range(seq_len):
            # Create a wave pattern
            codes[0, i, j] = 200 + i * 100 + int(50 * np.sin(j * 0.2))
    
    # Create mask
    mask = torch.zeros_like(codes, dtype=torch.bool)
    mask[:, :, 40:60] = True  # Mask middle section
    
    print(f"Input shape: {codes.shape}")
    print(f"Masked positions: {mask.sum().item()}")
    
    # Run VampNet
    print("\nRunning VampNet...")
    vampnet_model = interface.coarse
    vampnet_model.eval()
    
    with torch.no_grad():
        # Get embeddings
        masked_codes = codes.clone()
        masked_codes[mask] = 1024  # mask token
        
        # Forward pass through VampNet's embedding and transformer
        latents = vampnet_model.embedding.from_codes(masked_codes, interface.codec)
        vampnet_logits = vampnet_model(latents)
        
        # Get predictions
        vampnet_output = torch.argmax(
            vampnet_logits.reshape(batch_size, n_codebooks, seq_len, -1), 
            dim=-1
        )
        
        # Apply mask
        vampnet_final = codes.clone()
        vampnet_final[mask] = vampnet_output[mask]
    
    print(f"VampNet output shape: {vampnet_final.shape}")
    
    # Run ONNX
    print("\nRunning ONNX...")
    ort_inputs = {
        'codes': codes.numpy(),
        'mask': mask.numpy()
    }
    
    ort_outputs = ort_session.run(None, ort_inputs)
    onnx_output = torch.from_numpy(ort_outputs[0])
    
    print(f"ONNX output shape: {onnx_output.shape}")
    
    # Compare outputs
    print("\n=== Comparison Results ===")
    
    # Overall statistics
    total_diff = torch.abs(vampnet_final - onnx_output).float()
    print(f"Total differences: {(total_diff > 0).sum().item()}")
    print(f"Max difference: {total_diff.max().item()}")
    print(f"Mean difference: {total_diff.mean().item():.6f}")
    
    # Check masked positions only
    masked_vampnet = vampnet_final[mask]
    masked_onnx = onnx_output[mask]
    
    print(f"\nAt masked positions:")
    print(f"  Identical predictions: {(masked_vampnet == masked_onnx).sum().item()}/{mask.sum().item()}")
    print(f"  Agreement rate: {100*(masked_vampnet == masked_onnx).float().mean().item():.1f}%")
    
    # Token distribution
    print(f"\nToken distribution (masked positions):")
    print(f"  VampNet unique tokens: {len(torch.unique(masked_vampnet))}")
    print(f"  ONNX unique tokens: {len(torch.unique(masked_onnx))}")
    
    # Visualization
    print("\nCreating visualization...")
    fig, axes = plt.subplots(n_codebooks, 1, figsize=(12, 8))
    
    for i in range(n_codebooks):
        ax = axes[i] if n_codebooks > 1 else axes
        
        # Plot original
        ax.plot(codes[0, i].numpy(), 'k-', alpha=0.3, label='Original')
        
        # Plot outputs at masked positions
        mask_indices = torch.where(mask[0, i])[0].numpy()
        ax.scatter(mask_indices, vampnet_final[0, i][mask[0, i]].numpy(), 
                  c='blue', s=30, label='VampNet', alpha=0.7)
        ax.scatter(mask_indices, onnx_output[0, i][mask[0, i]].numpy(), 
                  c='red', s=20, label='ONNX', alpha=0.7)
        
        ax.set_ylabel(f'Codebook {i}')
        ax.legend()
    
    plt.xlabel('Time step')
    plt.suptitle('VampNet vs ONNX Model Comparison')
    plt.tight_layout()
    
    output_path = 'outputs/vampnet_onnx_comparison.png'
    plt.savefig(output_path)
    print(f"✓ Saved visualization to {output_path}")
    
    # Check if models produce same output on unmasked positions
    unmasked_check = (vampnet_final[~mask] == onnx_output[~mask]).all()
    print(f"\nUnmasked positions preserved correctly: {unmasked_check}")
    
    return vampnet_final, onnx_output


if __name__ == "__main__":
    # Create output directory
    Path("outputs").mkdir(exist_ok=True)
    
    # Run comparison
    compare_models()