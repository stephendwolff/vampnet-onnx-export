"""
Extract weights from pretrained VampNet model and create ONNX-compatible version.
"""

import torch
import torch.nn as nn
import numpy as np
import vampnet
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def analyze_vampnet_architecture():
    """Analyze VampNet model architecture to understand how to extract weights."""
    
    print("Loading VampNet models...")
    interface = vampnet.interface.Interface(
        codec_ckpt="models/vampnet/codec.pth",
        coarse_ckpt="models/vampnet/coarse.pth",
    )
    
    coarse_model = interface.coarse
    if hasattr(coarse_model, '_orig_mod'):
        model = coarse_model._orig_mod
    else:
        model = coarse_model
    
    print(f"\nModel type: {type(model)}")
    print(f"Model class name: {model.__class__.__name__}")
    
    # Print model architecture
    print("\n=== Model Architecture ===")
    print(model)
    
    # Analyze model parameters
    print("\n=== Model Parameters ===")
    total_params = 0
    param_shapes = {}
    
    for name, param in model.named_parameters():
        param_shapes[name] = param.shape
        total_params += param.numel()
        if len(param_shapes) <= 20:  # Show first 20 parameters
            print(f"{name}: {param.shape}")
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Total parameter tensors: {len(param_shapes)}")
    
    # Look for key components
    print("\n=== Key Components ===")
    
    # Check for embeddings
    embedding_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            embedding_layers.append((name, module))
            print(f"Embedding: {name} - {module.num_embeddings} x {module.embedding_dim}")
    
    # Check for transformer layers
    transformer_layers = []
    for name, module in model.named_modules():
        if 'transformer' in name.lower() or 'attention' in name.lower():
            transformer_layers.append((name, type(module).__name__))
            if len(transformer_layers) <= 10:
                print(f"Transformer component: {name} - {type(module).__name__}")
    
    # Check for output projections
    linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and 'classifier' in name:
            linear_layers.append((name, module))
            print(f"Classifier: {name} - {module.in_features} -> {module.out_features}")
    
    return model, param_shapes, embedding_layers, transformer_layers, linear_layers


def create_onnx_compatible_model(original_model, n_codebooks=4, vocab_size=1024):
    """
    Create an ONNX-compatible model and attempt to transfer weights.
    """
    
    class ONNXVampNet(nn.Module):
        def __init__(self, d_model=512, n_heads=8, n_layers=12):
            super().__init__()
            self.n_codebooks = n_codebooks
            self.vocab_size = vocab_size
            self.d_model = d_model
            
            # Create embedding layers
            self.embeddings = nn.ModuleList([
                nn.Embedding(vocab_size + 1, d_model)  # +1 for mask token
                for _ in range(n_codebooks)
            ])
            
            # Positional encoding
            self.pos_embedding = nn.Parameter(torch.zeros(1, 1000, d_model))
            
            # Transformer
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_model * 4,
                    batch_first=True,
                    norm_first=True
                ),
                num_layers=n_layers
            )
            
            # Output projections
            self.classifiers = nn.ModuleList([
                nn.Linear(d_model, vocab_size)
                for _ in range(n_codebooks)
            ])
            
        def forward(self, codes, mask):
            batch_size, n_codebooks, seq_len = codes.shape
            
            # Embed each codebook
            embedded = []
            for i in range(self.n_codebooks):
                emb = self.embeddings[i](codes[:, i])  # [batch, seq, d_model]
                embedded.append(emb)
            
            # Stack and reshape
            x = torch.stack(embedded, dim=1)  # [batch, n_codebooks, seq, d_model]
            x = x.view(batch_size, n_codebooks * seq_len, self.d_model)
            
            # Add positional encoding
            x = x + self.pos_embedding[:, :x.size(1)]
            
            # Transform
            x = self.transformer(x)
            
            # Reshape back
            x = x.view(batch_size, n_codebooks, seq_len, self.d_model)
            
            # Classify each codebook
            logits = []
            for i in range(self.n_codebooks):
                logit = self.classifiers[i](x[:, i])  # [batch, seq, vocab]
                logits.append(logit)
            
            logits = torch.stack(logits, dim=1)  # [batch, n_codebooks, seq, vocab]
            
            # Generate tokens
            predictions = torch.argmax(logits, dim=-1)
            output = torch.where(mask.bool(), predictions, codes)
            
            return output
    
    # Create the model
    onnx_model = ONNXVampNet()
    
    print("\n=== Attempting Weight Transfer ===")
    
    # Try to find matching weights
    original_state = original_model.state_dict()
    onnx_state = onnx_model.state_dict()
    
    transferred = 0
    failed = 0
    
    for onnx_name, onnx_param in onnx_state.items():
        # Try to find corresponding parameter in original model
        found = False
        
        # Direct name match
        if onnx_name in original_state and original_state[onnx_name].shape == onnx_param.shape:
            onnx_param.data.copy_(original_state[onnx_name])
            print(f"✓ Transferred: {onnx_name}")
            transferred += 1
            found = True
            continue
        
        # Try to find by partial match
        for orig_name, orig_param in original_state.items():
            if orig_param.shape == onnx_param.shape:
                # Check if names are related
                if any(part in orig_name for part in onnx_name.split('.')):
                    onnx_param.data.copy_(orig_param)
                    print(f"✓ Transferred: {onnx_name} <- {orig_name}")
                    transferred += 1
                    found = True
                    break
        
        if not found:
            print(f"✗ No match for: {onnx_name} {onnx_param.shape}")
            failed += 1
    
    print(f"\nTransferred: {transferred}")
    print(f"Failed: {failed}")
    
    return onnx_model


def main():
    # Analyze architecture
    model, param_shapes, embeddings, transformers, classifiers = analyze_vampnet_architecture()
    
    # Create ONNX-compatible model
    onnx_model = create_onnx_compatible_model(model)
    
    # Test the model
    print("\n=== Testing ONNX Model ===")
    test_codes = torch.randint(0, 1024, (1, 4, 100))
    test_mask = torch.randint(0, 2, (1, 4, 100))
    
    onnx_model.eval()
    with torch.no_grad():
        try:
            output = onnx_model(test_codes, test_mask)
            print(f"✓ Forward pass successful! Output shape: {output.shape}")
            
            # Try to export to ONNX
            print("\nExporting to ONNX...")
            torch.onnx.export(
                onnx_model,
                (test_codes, test_mask),
                "vampnet_with_weights.onnx",
                input_names=['codes', 'mask'],
                output_names=['generated_codes'],
                dynamic_axes={
                    'codes': {0: 'batch', 2: 'sequence'},
                    'mask': {0: 'batch', 2: 'sequence'},
                    'generated_codes': {0: 'batch', 2: 'sequence'}
                },
                opset_version=14
            )
            print("✓ Successfully exported to vampnet_with_weights.onnx")
            
        except Exception as e:
            print(f"✗ Error: {e}")


if __name__ == "__main__":
    main()