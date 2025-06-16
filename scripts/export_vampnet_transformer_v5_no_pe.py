#!/usr/bin/env python3
"""
Export VampNet transformer V5 WITHOUT positional encoding.
VampNet uses relative position bias in attention, not absolute positional encoding.
"""

import torch
import torch.nn as nn
import numpy as np
import onnxruntime as ort
from pathlib import Path
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.export_vampnet_transformer_v2 import VampNetTransformerV2
from scripts.codebook_embedding_correct_v2 import CodebookEmbeddingCorrectV2
from scripts.custom_ops.rmsnorm_onnx import SimpleRMSNorm
from scripts.custom_ops.film_onnx import SimpleFiLM
from scripts.custom_ops.multihead_attention_onnx import OnnxMultiheadAttention


class GatedFFN(nn.Module):
    """FFN with GatedGELU activation matching VampNet."""
    
    def __init__(self, d_model):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_model * 4)
        self.w_2 = nn.Linear(d_model * 2, d_model)
        self.activation = nn.GELU()
        
    def forward(self, x):
        hidden = self.w_1(x)
        hidden, gate = hidden.chunk(2, dim=-1)
        hidden = hidden * self.activation(gate)
        return self.w_2(hidden)


class VampNetTransformerV5NoPE(nn.Module):
    """
    VampNet transformer WITHOUT positional encoding.
    VampNet uses relative position bias in attention instead.
    """
    
    def __init__(self,
                 n_codebooks: int = 4,
                 n_conditioning_codebooks: int = 0,
                 vocab_size: int = 1024,
                 latent_dim: int = 8,
                 d_model: int = 1280,
                 n_heads: int = 20,
                 n_layers: int = 20,
                 dropout: float = 0.0,
                 mask_token: int = 1024,
                 use_gated_ffn: bool = True):
        super().__init__()
        
        self.n_codebooks = n_codebooks
        self.n_conditioning_codebooks = n_conditioning_codebooks
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.mask_token = mask_token
        self.n_heads = n_heads
        
        # CORRECT embedding layer
        self.embedding = CodebookEmbeddingCorrectV2(
            n_codebooks=n_codebooks,
            vocab_size=vocab_size,
            latent_dim=latent_dim,
            d_model=d_model
        )
        
        # NO POSITIONAL ENCODING - VampNet doesn't use it!
        
        # Transformer layers
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            layer = nn.ModuleDict({
                'norm1': SimpleRMSNorm(d_model),
                'self_attn': OnnxMultiheadAttention(d_model, n_heads, dropout),
                'norm2': SimpleRMSNorm(d_model),
                'film': SimpleFiLM(d_model, d_model),
            })
            
            if use_gated_ffn:
                layer['ffn'] = GatedFFN(d_model)
            else:
                layer['ffn'] = nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Linear(d_model * 4, d_model)
                )
            
            self.layers.append(layer)
        
        # Final norm
        self.final_norm = SimpleRMSNorm(d_model)
        
        # Output projections
        n_output_codebooks = n_codebooks - n_conditioning_codebooks
        self.output_projs = nn.ModuleList([
            nn.Linear(d_model, vocab_size + 1)
            for _ in range(n_output_codebooks)
        ])
    
    def forward(self, codes, mask=None):
        """
        Forward pass WITHOUT positional encoding.
        """
        batch_size, n_codebooks, seq_len = codes.shape
        
        # Handle conditioning/generation split
        if self.n_conditioning_codebooks > 0:
            cond_codes = codes[:, :self.n_conditioning_codebooks]
            gen_codes = codes[:, self.n_conditioning_codebooks:]
        else:
            cond_codes = None
            gen_codes = codes
        
        # Apply mask to generation codes
        masked_codes = gen_codes.clone()
        if mask is not None:
            if self.n_conditioning_codebooks > 0:
                gen_mask = mask[:, self.n_conditioning_codebooks:]
                masked_codes[gen_mask] = self.mask_token
            else:
                masked_codes[mask] = self.mask_token
        
        # Combine conditioning and masked generation codes
        if self.n_conditioning_codebooks > 0:
            masked_codes = torch.cat([cond_codes, masked_codes], dim=1)
        
        # Get embeddings using CORRECT architecture
        x = self.embedding(masked_codes)  # [batch, seq_len, d_model]
        
        # NO positional encoding added here!
        
        # Apply transformer layers
        for layer in self.layers:
            x_norm = layer['norm1'](x)
            attn_out, _ = layer['self_attn'](x_norm, x_norm, x_norm)
            x = x + attn_out
            
            x_norm = layer['norm2'](x)
            x_norm = layer['film'](x_norm, x_norm)
            ffn_out = layer['ffn'](x_norm)
            x = x + ffn_out
        
        # Final norm
        x = self.final_norm(x)
        
        # Generate logits
        all_logits = []
        n_output_codebooks = self.n_codebooks - self.n_conditioning_codebooks
        for i in range(n_output_codebooks):
            cb_logits = self.output_projs[i](x)
            all_logits.append(cb_logits)
        
        # Stack logits
        logits = torch.stack(all_logits, dim=1)  # [batch, n_output_codebooks, seq_len, vocab_size+1]
        
        # Pad with zeros for conditioning codebooks if needed
        if self.n_conditioning_codebooks > 0:
            full_logits = torch.zeros(
                batch_size, n_codebooks, seq_len, self.vocab_size + 1,
                device=logits.device, dtype=logits.dtype
            )
            full_logits[:, self.n_conditioning_codebooks:] = logits
            logits = full_logits
        
        return logits


def transfer_weights_v5(vampnet_checkpoint, model, model_type="coarse"):
    """Transfer weights from VampNet to our model WITHOUT positional encoding."""
    print(f"\nTransferring weights from {vampnet_checkpoint}")
    
    # Load VampNet
    from vampnet.modules.transformer import VampNet
    vampnet = VampNet.load(vampnet_checkpoint, map_location='cpu')
    
    # Load checkpoint for codec embeddings
    ckpt = torch.load(vampnet_checkpoint, map_location='cpu')
    
    # 1. Transfer embeddings (using correct method)
    print("Transferring embeddings...")
    
    # Transfer codec embeddings
    if 'codec' in ckpt:
        codec_state = ckpt['codec']
        for i in range(model.n_codebooks):
            key = f'quantizer.quantizers.{i}.codebook.weight'
            if key in codec_state:
                codec_emb = codec_state[key]
                model.embedding.embeddings[i].weight.data[:vampnet.vocab_size] = codec_emb
                print(f"  ✓ Codebook {i}: {codec_emb.shape}")
    
    # Transfer special token embeddings to the last position in embedding table
    if hasattr(vampnet.embedding, 'special') and 'MASK' in vampnet.embedding.special:
        mask_emb = vampnet.embedding.special['MASK']  # [n_codebooks, latent_dim]
        for i in range(model.n_codebooks):
            # Put mask embedding at index vocab_size
            model.embedding.embeddings[i].weight.data[vampnet.vocab_size] = mask_emb[i]
        print(f"  ✓ MASK embeddings: {mask_emb.shape}")
    
    # Transfer Conv1d projection
    model.embedding.out_proj.weight.data = vampnet.embedding.out_proj.weight.data
    model.embedding.out_proj.bias.data = vampnet.embedding.out_proj.bias.data
    print(f"  ✓ out_proj: {model.embedding.out_proj.weight.shape}")
    
    # 2. NO positional encoding transfer - VampNet doesn't have it!
    print("Skipping positional encoding (VampNet doesn't use it)")
    
    # 3. Transfer transformer layers
    print("Transferring transformer layers...")
    for i, (onnx_layer, vamp_layer) in enumerate(zip(model.layers, vampnet.transformer.layers)):
        # RMSNorm
        onnx_layer['norm1'].weight.data = vamp_layer.norm_1.weight.data
        onnx_layer['norm2'].weight.data = vamp_layer.norm_3.weight.data
        
        # Attention - VampNet uses w_qs, w_ks, w_vs instead of w_q, w_k, w_v
        onnx_layer['self_attn'].w_q.weight.data = vamp_layer.self_attn.w_qs.weight.data
        if hasattr(vamp_layer.self_attn.w_qs, 'bias') and vamp_layer.self_attn.w_qs.bias is not None:
            onnx_layer['self_attn'].w_q.bias.data = vamp_layer.self_attn.w_qs.bias.data
            
        onnx_layer['self_attn'].w_k.weight.data = vamp_layer.self_attn.w_ks.weight.data
        if hasattr(vamp_layer.self_attn.w_ks, 'bias') and vamp_layer.self_attn.w_ks.bias is not None:
            onnx_layer['self_attn'].w_k.bias.data = vamp_layer.self_attn.w_ks.bias.data
            
        onnx_layer['self_attn'].w_v.weight.data = vamp_layer.self_attn.w_vs.weight.data
        if hasattr(vamp_layer.self_attn.w_vs, 'bias') and vamp_layer.self_attn.w_vs.bias is not None:
            onnx_layer['self_attn'].w_v.bias.data = vamp_layer.self_attn.w_vs.bias.data
            
        onnx_layer['self_attn'].w_o.weight.data = vamp_layer.self_attn.fc.weight.data
        if hasattr(vamp_layer.self_attn.fc, 'bias') and vamp_layer.self_attn.fc.bias is not None:
            onnx_layer['self_attn'].w_o.bias.data = vamp_layer.self_attn.fc.bias.data
        
        # FiLM
        if vamp_layer.film_3.input_dim > 0:
            onnx_layer['film'].gamma_weight.data = vamp_layer.film_3.gamma.weight.data.t()
            onnx_layer['film'].gamma_bias.data = vamp_layer.film_3.gamma.bias.data
            onnx_layer['film'].beta_weight.data = vamp_layer.film_3.beta.weight.data.t()
            onnx_layer['film'].beta_bias.data = vamp_layer.film_3.beta.bias.data
        
        # FFN
        if hasattr(vamp_layer.feed_forward, 'w_1'):
            onnx_layer['ffn'].w_1.weight.data = vamp_layer.feed_forward.w_1.weight.data
            onnx_layer['ffn'].w_2.weight.data = vamp_layer.feed_forward.w_2.weight.data
    
    # 4. Transfer final norm
    if hasattr(vampnet.transformer, 'final_norm'):
        model.final_norm.weight.data = vampnet.transformer.final_norm.weight.data
    
    # 5. Transfer output projections
    print("Transferring output projections...")
    classifier_weight = vampnet.classifier.layers[0].weight.data.squeeze(-1)
    
    n_output_codebooks = model.n_codebooks - model.n_conditioning_codebooks
    for i in range(n_output_codebooks):
        start_idx = i * (model.vocab_size + 1)
        end_idx = (i + 1) * (model.vocab_size + 1)
        
        vamp_start = i * vampnet.vocab_size
        vamp_end = (i + 1) * vampnet.vocab_size
        
        model.output_projs[i].weight.data[:vampnet.vocab_size] = classifier_weight[vamp_start:vamp_end]
        model.output_projs[i].weight.data[vampnet.vocab_size] = 0
    
    print("✓ Weight transfer complete!")


def export_model_v5(
    model_type="coarse",
    checkpoint_path=None,
    output_path=None
):
    """Export model WITHOUT positional encoding."""
    
    print(f"\n{'='*60}")
    print(f"Exporting {model_type.upper()} Model V5 WITHOUT Positional Encoding")
    print(f"{'='*60}")
    
    # Configuration
    if model_type == "coarse":
        config = {
            'n_codebooks': 4,
            'n_conditioning_codebooks': 0,
            'vocab_size': 1024,
            'latent_dim': 8,
            'd_model': 1280,
            'n_heads': 20,
            'n_layers': 20,
            'use_gated_ffn': True
        }
        default_checkpoint = "../models/vampnet/coarse.pth"
        default_output = "../onnx_models_fixed/coarse_v5_no_pe.onnx"
    else:  # c2f
        config = {
            'n_codebooks': 14,
            'n_conditioning_codebooks': 4,
            'vocab_size': 1024,
            'latent_dim': 8,
            'd_model': 1280,
            'n_heads': 20,
            'n_layers': 16,
            'use_gated_ffn': True
        }
        default_checkpoint = "../models/vampnet/c2f.pth"
        default_output = "../onnx_models_fixed/c2f_v5_no_pe.onnx"
    
    checkpoint_path = checkpoint_path or default_checkpoint
    output_path = output_path or default_output
    
    print("\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # Create model
    model = VampNetTransformerV5NoPE(**config)
    
    # Transfer weights
    if Path(checkpoint_path).exists():
        transfer_weights_v5(checkpoint_path, model, model_type)
    
    model.eval()
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 1
    seq_len = 100
    codes = torch.randint(0, 1024, (batch_size, config['n_codebooks'], seq_len))
    mask = torch.zeros((batch_size, config['n_codebooks'], seq_len), dtype=torch.bool)
    mask[:, :, 50:60] = True
    
    with torch.no_grad():
        logits = model(codes, mask)
    
    print(f"Output shape: {logits.shape}")
    print(f"Output range: [{logits.min():.2f}, {logits.max():.2f}]")
    
    # Export to ONNX
    print(f"\nExporting to {output_path}")
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    
    torch.onnx.export(
        model,
        (codes, mask),
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
    
    outputs = session.run(None, {
        'codes': codes.numpy().astype(np.int64),
        'mask': mask.numpy()
    })[0]
    
    print(f"ONNX output shape: {outputs.shape}")
    print(f"✅ Verification complete!")
    
    return output_path


if __name__ == "__main__":
    # Export both models WITHOUT positional encoding
    export_model_v5("coarse")
    export_model_v5("c2f")
    
    print("\n" + "="*60)
    print("V5 MODELS EXPORTED WITHOUT POSITIONAL ENCODING!")
    print("VampNet uses relative position bias, not absolute PE")
    print("="*60)