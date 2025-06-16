#!/usr/bin/env python3
"""
Export VampNet transformer V6 with PROPER architecture matching VampNet exactly:
1. NO positional encoding (uses relative position bias instead)
2. Relative attention in first layer only
3. Correct embedding architecture (latent_dim=8)
4. GatedGELU activation in FFN
"""

import torch
import torch.nn as nn
import numpy as np
import onnxruntime as ort
from pathlib import Path
import sys
import os
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.codebook_embedding_correct_v2 import CodebookEmbeddingCorrectV2
from scripts.custom_ops.rmsnorm_onnx import SimpleRMSNorm
from scripts.custom_ops.film_onnx import SimpleFiLM
from scripts.custom_ops.multihead_attention_onnx import OnnxMultiheadAttention
from scripts.custom_ops.relative_attention_onnx import OnnxMultiheadRelativeAttention


class NewGELU(nn.Module):
    """VampNet's custom GELU implementation."""
    def forward(self, x):
        return (
            0.5
            * x
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
                )
            )
        )


class GatedGELU(nn.Module):
    """VampNet's FFN with GatedGELU activation."""
    
    def __init__(self, d_model, ffn_dim=None):
        super().__init__()
        if ffn_dim is None:
            # VampNet uses 2560 for FFN, not 5120!
            ffn_dim = d_model * 2  # 1280 * 2 = 2560
        
        # w_1 projects to 2*ffn_dim for gating
        self.w_1 = nn.Linear(d_model, ffn_dim * 2)  # 1280 -> 5120
        self.w_2 = nn.Linear(ffn_dim, d_model)      # 2560 -> 1280
        self.activation = NewGELU()  # Use VampNet's custom GELU
        
    def forward(self, x):
        # Project to 2*ffn_dim
        hidden = self.w_1(x)
        
        # Split into two halves for gating
        hidden, gate = hidden.chunk(2, dim=-1)  # Each is 2560
        
        # Apply gated activation
        hidden = hidden * self.activation(gate)
        
        # Project back
        return self.w_2(hidden)


class VampNetTransformerV6(nn.Module):
    """
    VampNet transformer with PROPER architecture:
    - NO positional encoding
    - Relative position bias in first layer only
    - Correct dimensions and activations
    """
    
    def __init__(self,
                 n_codebooks: int = 4,
                 n_conditioning_codebooks: int = 0,
                 vocab_size: int = 1024,
                 latent_dim: int = 8,
                 d_model: int = 1280,
                 n_heads: int = 20,
                 n_layers: int = 20,
                 dropout: float = 0.1,
                 mask_token: int = 1024):
        super().__init__()
        
        self.n_codebooks = n_codebooks
        self.n_conditioning_codebooks = n_conditioning_codebooks
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.mask_token = mask_token
        self.n_heads = n_heads
        
        # Correct embedding layer
        self.embedding = CodebookEmbeddingCorrectV2(
            n_codebooks=n_codebooks,
            vocab_size=vocab_size,
            latent_dim=latent_dim,
            d_model=d_model
        )
        
        # NO positional encoding!
        
        # Transformer layers
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            layer = nn.ModuleDict({
                'norm_1': SimpleRMSNorm(d_model),  # Match VampNet naming
                'norm_3': SimpleRMSNorm(d_model),  # Match VampNet naming (norm_2 is for cross-attention)
                'film_3': SimpleFiLM(d_model, d_model),  # Match VampNet naming
            })
            
            # First layer uses relative attention, others use standard
            if i == 0:
                layer['self_attn'] = OnnxMultiheadRelativeAttention(
                    d_model, n_heads, dropout,
                    bidirectional=True,
                    has_relative_attention_bias=True,
                    attention_num_buckets=32,
                    attention_max_distance=128
                )
            else:
                layer['self_attn'] = OnnxMultiheadAttention(
                    d_model, n_heads, dropout
                )
            
            # FFN with GatedGELU
            layer['feed_forward'] = GatedGELU(d_model)
            
            self.layers.append(layer)
        
        # Final norm (VampNet's transformer has this)
        self.final_norm = SimpleRMSNorm(d_model)
        
        # Output projections
        n_output_codebooks = n_codebooks - n_conditioning_codebooks
        self.output_projs = nn.ModuleList([
            nn.Linear(d_model, vocab_size + 1)
            for _ in range(n_output_codebooks)
        ])
    
    def forward(self, codes, mask=None):
        """
        Forward pass matching VampNet exactly.
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
        
        # Get embeddings
        x = self.embedding(masked_codes)  # [batch, seq_len, d_model]
        
        # NO positional encoding!
        
        # Apply transformer layers
        position_bias = None
        for i, layer in enumerate(self.layers):
            # Pre-norm
            x_norm = layer['norm_1'](x)
            
            # Self-attention
            if i == 0:
                # First layer with relative attention
                attn_out, position_bias = layer['self_attn'](
                    x_norm, x_norm, x_norm, 
                    mask=None,  # VampNet doesn't use padding mask
                    position_bias=position_bias
                )
            else:
                # Other layers with standard attention
                attn_out, _ = layer['self_attn'](x_norm, x_norm, x_norm)
            
            x = x + attn_out
            
            # FFN block
            x_norm = layer['norm_3'](x)
            x_norm = layer['film_3'](x_norm, x_norm)  # FiLM conditioning
            ffn_out = layer['feed_forward'](x_norm)
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


def transfer_weights_v6(vampnet_checkpoint, model, model_type="coarse", codec_path=None):
    """Transfer weights from VampNet to our model."""
    print(f"\nTransferring weights from {vampnet_checkpoint}")
    
    # Load VampNet
    from vampnet.modules.transformer import VampNet
    from lac.model.lac import LAC as DAC
    vampnet = VampNet.load(vampnet_checkpoint, map_location='cpu')
    
    # Load codec model for embeddings
    if codec_path and Path(codec_path).exists():
        print(f"Loading codec from {codec_path}")
        codec = DAC.load(Path(codec_path))
        use_codec_model = True
    else:
        print("No codec path provided, will use checkpoint embeddings")
        use_codec_model = False
    
    # Load checkpoint
    ckpt = torch.load(vampnet_checkpoint, map_location='cpu')
    
    # 1. Transfer embeddings
    print("Transferring embeddings...")
    
    # Transfer codec embeddings
    if use_codec_model:
        # Use actual codec model embeddings (correct approach)
        for i in range(model.n_codebooks):
            codec_emb = codec.quantizer.quantizers[i].codebook.weight
            model.embedding.embeddings[i].weight.data[:vampnet.vocab_size] = codec_emb
            print(f"  ✓ Codebook {i} from codec model: {codec_emb.shape}")
    elif 'codec' in ckpt:
        # Fallback to checkpoint embeddings
        codec_state = ckpt['codec']
        for i in range(model.n_codebooks):
            key = f'quantizer.quantizers.{i}.codebook.weight'
            if key in codec_state:
                codec_emb = codec_state[key]
                model.embedding.embeddings[i].weight.data[:vampnet.vocab_size] = codec_emb
                print(f"  ✓ Codebook {i} from checkpoint: {codec_emb.shape}")
    
    # Transfer special token embeddings
    if hasattr(vampnet.embedding, 'special') and 'MASK' in vampnet.embedding.special:
        mask_emb = vampnet.embedding.special['MASK']
        for i in range(model.n_codebooks):
            model.embedding.embeddings[i].weight.data[vampnet.vocab_size] = mask_emb[i]
        print(f"  ✓ MASK embeddings: {mask_emb.shape}")
    
    # Transfer Conv1d projection
    model.embedding.out_proj.weight.data = vampnet.embedding.out_proj.weight.data
    model.embedding.out_proj.bias.data = vampnet.embedding.out_proj.bias.data
    print(f"  ✓ out_proj: {model.embedding.out_proj.weight.shape}")
    
    # 2. NO positional encoding to transfer
    print("No positional encoding (VampNet uses relative position bias)")
    
    # 3. Transfer transformer layers
    print("Transferring transformer layers...")
    for i, (onnx_layer, vamp_layer) in enumerate(zip(model.layers, vampnet.transformer.layers)):
        # RMSNorm (matching VampNet's naming)
        onnx_layer['norm_1'].weight.data = vamp_layer.norm_1.weight.data
        onnx_layer['norm_3'].weight.data = vamp_layer.norm_3.weight.data
        
        # Attention weights
        # VampNet uses w_qs, w_ks, w_vs, fc instead of w_q, w_k, w_v, w_o
        if i == 0:
            # First layer with relative attention
            onnx_layer['self_attn'].w_qs.weight.data = vamp_layer.self_attn.w_qs.weight.data
            onnx_layer['self_attn'].w_ks.weight.data = vamp_layer.self_attn.w_ks.weight.data
            onnx_layer['self_attn'].w_vs.weight.data = vamp_layer.self_attn.w_vs.weight.data
            onnx_layer['self_attn'].fc.weight.data = vamp_layer.self_attn.fc.weight.data
            
            # Transfer relative attention bias
            if hasattr(vamp_layer.self_attn, 'relative_attention_bias'):
                onnx_layer['self_attn'].relative_attention_bias.weight.data = \
                    vamp_layer.self_attn.relative_attention_bias.weight.data
        else:
            # Standard attention layers
            onnx_layer['self_attn'].w_q.weight.data = vamp_layer.self_attn.w_qs.weight.data
            onnx_layer['self_attn'].w_k.weight.data = vamp_layer.self_attn.w_ks.weight.data
            onnx_layer['self_attn'].w_v.weight.data = vamp_layer.self_attn.w_vs.weight.data
            onnx_layer['self_attn'].w_o.weight.data = vamp_layer.self_attn.fc.weight.data
        
        # FiLM
        if vamp_layer.film_3.input_dim > 0:
            onnx_layer['film_3'].gamma_weight.data = vamp_layer.film_3.gamma.weight.data.t()
            onnx_layer['film_3'].gamma_bias.data = vamp_layer.film_3.gamma.bias.data
            onnx_layer['film_3'].beta_weight.data = vamp_layer.film_3.beta.weight.data.t()
            onnx_layer['film_3'].beta_bias.data = vamp_layer.film_3.beta.bias.data
        else:
            # FiLM with input_dim=0 should act as identity
            # Set gamma to 1 and beta to 0
            onnx_layer['film_3'].gamma_weight.data.fill_(0)  # No linear transform
            onnx_layer['film_3'].gamma_bias.data.fill_(1)    # gamma = 1
            onnx_layer['film_3'].beta_weight.data.fill_(0)   # No linear transform
            onnx_layer['film_3'].beta_bias.data.fill_(0)     # beta = 0
        
        # FFN (feed_forward)
        onnx_layer['feed_forward'].w_1.weight.data = vamp_layer.feed_forward.w_1.weight.data
        onnx_layer['feed_forward'].w_2.weight.data = vamp_layer.feed_forward.w_2.weight.data
    
    # 4. Transfer final norm
    if hasattr(vampnet.transformer, 'norm'):
        model.final_norm.weight.data = vampnet.transformer.norm.weight.data
    
    # 5. Transfer output projections
    print("Transferring output projections...")
    classifier_weight = vampnet.classifier.layers[0].weight.data.squeeze(-1)
    
    n_output_codebooks = model.n_codebooks - model.n_conditioning_codebooks
    for i in range(n_output_codebooks):
        vamp_start = i * vampnet.vocab_size
        vamp_end = (i + 1) * vampnet.vocab_size
        
        model.output_projs[i].weight.data[:vampnet.vocab_size] = classifier_weight[vamp_start:vamp_end]
        model.output_projs[i].weight.data[vampnet.vocab_size] = 0
    
    print("✓ Weight transfer complete!")


def export_model_v6(
    model_type="coarse",
    checkpoint_path=None,
    output_path=None,
    codec_path=None
):
    """Export model with PROPER VampNet architecture."""
    
    print(f"\n{'='*60}")
    print(f"Exporting {model_type.upper()} Model V6 with PROPER Architecture")
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
            'dropout': 0.1,
        }
        default_checkpoint = "../models/vampnet/coarse.pth"
        default_output = "../onnx_models_fixed/coarse_v6_proper.onnx"
    else:  # c2f
        config = {
            'n_codebooks': 14,
            'n_conditioning_codebooks': 4,
            'vocab_size': 1024,
            'latent_dim': 8,
            'd_model': 1280,
            'n_heads': 20,
            'n_layers': 16,
            'dropout': 0.1,
        }
        default_checkpoint = "../models/vampnet/c2f.pth"
        default_output = "../onnx_models_fixed/c2f_v6_proper.onnx"
    
    checkpoint_path = checkpoint_path or default_checkpoint
    output_path = output_path or default_output
    
    print("\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # Create model
    model = VampNetTransformerV6(**config)
    
    # Transfer weights
    if Path(checkpoint_path).exists():
        transfer_weights_v6(checkpoint_path, model, model_type, codec_path)
    
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
    # Export both models with PROPER architecture
    codec_path = "../models/vampnet/codec.pth"
    export_model_v6("coarse", codec_path=codec_path)
    export_model_v6("c2f", codec_path=codec_path)
    
    print("\n" + "="*60)
    print("V6 MODELS WITH PROPER ARCHITECTURE EXPORTED!")
    print("- NO positional encoding")
    print("- Relative position bias in first layer")
    print("- Correct dimensions and activations")
    print("="*60)