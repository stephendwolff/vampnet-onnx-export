#!/usr/bin/env python3
"""
Export VampNet transformer V9 with the EXACT same flow as VampNet.
Key insight: VampNet's forward() expects LATENTS, not codes!

VampNet flow:
1. codes → from_codes() → latents (using codec embeddings)
2. latents → embedding() → embeddings (using Conv1d projection)
3. embeddings → transformer → logits

Our ONNX model must match this exactly.
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


class VampNetGatedGELU(nn.Module):
    """VampNet's exact GatedGELU implementation."""
    def __init__(self):
        super().__init__()
        self.gelu = NewGELU()
    
    def forward(self, x, dim=-1):
        p1, p2 = x.chunk(2, dim=dim)
        return p1 * self.gelu(p2)


class FeedForwardGatedGELU(nn.Module):
    """VampNet's FFN with GatedGELU activation - matching exact structure."""
    
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        # VampNet uses fixed dimensions
        ffn_dim = d_model * 2  # 1280 * 2 = 2560
        
        self.w_1 = nn.Linear(d_model, ffn_dim * 2, bias=False)  # 1280 -> 5120
        self.w_2 = nn.Linear(ffn_dim, d_model, bias=False)      # 2560 -> 1280
        self.drop = nn.Dropout(dropout)
        self.act = VampNetGatedGELU()
        
    def forward(self, x):
        x = self.w_1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.w_2(x)
        return x


class VampNetEmbeddingLayer(nn.Module):
    """
    Embedding layer that matches VampNet's exact process.
    This takes LATENTS as input (not codes!), matching VampNet's forward() expectation.
    """
    
    def __init__(self, n_codebooks: int, latent_dim: int, d_model: int):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.latent_dim = latent_dim
        self.d_model = d_model
        
        # Only the projection layer - no embedding tables!
        # VampNet's embedding layer expects latents that already have the codec embeddings
        self.out_proj = nn.Conv1d(n_codebooks * latent_dim, d_model, kernel_size=1)
        
    def forward(self, latents):
        """
        Args:
            latents: [batch, n_codebooks * latent_dim, seq_len] - already embedded via codec
        Returns:
            embeddings: [batch, seq_len, d_model]
        """
        # Apply projection
        output = self.out_proj(latents)  # [batch, d_model, seq_len]
        
        # Transpose to [batch, seq_len, d_model]
        output = output.transpose(1, 2)
        
        return output


class VampNetTransformerV9(nn.Module):
    """
    VampNet transformer that expects LATENTS as input, matching the original exactly.
    """
    
    def __init__(self,
                 n_codebooks: int = 4,
                 n_conditioning_codebooks: int = 0,
                 vocab_size: int = 1024,
                 latent_dim: int = 8,
                 d_model: int = 1280,
                 n_heads: int = 20,
                 n_layers: int = 20,
                 dropout: float = 0.1):
        super().__init__()
        
        self.n_codebooks = n_codebooks
        self.n_conditioning_codebooks = n_conditioning_codebooks
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Embedding layer that expects latents
        self.embedding = VampNetEmbeddingLayer(
            n_codebooks=n_codebooks,
            latent_dim=latent_dim,
            d_model=d_model
        )
        
        # Transformer layers
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            layer = nn.ModuleDict({
                'norm_1': SimpleRMSNorm(d_model),
                'norm_3': SimpleRMSNorm(d_model),
                'film_3': SimpleFiLM(d_model, d_model),
            })
            
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
            
            layer['feed_forward'] = FeedForwardGatedGELU(d_model, dropout)
            self.layers.append(layer)
        
        # Final norm
        self.final_norm = SimpleRMSNorm(d_model)
        
        # Output projections
        n_output_codebooks = n_codebooks - n_conditioning_codebooks
        self.output_projs = nn.ModuleList([
            nn.Linear(d_model, vocab_size + 1)
            for _ in range(n_output_codebooks)
        ])
    
    def forward(self, latents):
        """
        Forward pass expecting LATENTS, not codes!
        
        Args:
            latents: [batch, n_codebooks * latent_dim, seq_len]
        Returns:
            logits: [batch, n_codebooks, seq_len, vocab_size+1]
        """
        # Get embeddings from latents
        x = self.embedding(latents)  # [batch, seq_len, d_model]
        
        # Apply transformer layers
        position_bias = None
        for i, layer in enumerate(self.layers):
            # Pre-norm
            x_norm = layer['norm_1'](x)
            
            # Self-attention
            if i == 0:
                attn_out, position_bias = layer['self_attn'](
                    x_norm, x_norm, x_norm, 
                    mask=None,
                    position_bias=position_bias
                )
            else:
                attn_out, _ = layer['self_attn'](x_norm, x_norm, x_norm)
            
            x = x + attn_out
            
            # FFN block
            x_norm = layer['norm_3'](x)
            x_norm = layer['film_3'](x_norm, x_norm)
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
            batch_size, _, seq_len, vocab_size = logits.shape
            full_logits = torch.zeros(
                batch_size, self.n_codebooks, seq_len, vocab_size,
                device=logits.device, dtype=logits.dtype
            )
            full_logits[:, self.n_conditioning_codebooks:] = logits
            logits = full_logits
        
        return logits


def create_latent_preprocessor(checkpoint_path, codec_path, n_codebooks=4, latent_dim=8):
    """
    Create a preprocessing module that converts codes to latents using codec embeddings.
    This replicates VampNet's from_codes() method.
    """
    from vampnet.modules.transformer import VampNet
    from lac.model.lac import LAC as DAC
    
    # Load models
    vampnet = VampNet.load(checkpoint_path, map_location='cpu')
    codec = DAC.load(Path(codec_path))
    
    # Create embedding lookup tables matching the codec
    embeddings = nn.ModuleList()
    for i in range(n_codebooks):
        # Create embedding with vocab_size + 1 (for mask token)
        emb = nn.Embedding(vampnet.vocab_size + 1, latent_dim)
        
        # Copy codec embeddings
        with torch.no_grad():
            codec_weight = codec.quantizer.quantizers[i].codebook.weight
            emb.weight.data[:vampnet.vocab_size] = codec_weight
            
            # Copy mask token embedding
            if hasattr(vampnet.embedding, 'special') and 'MASK' in vampnet.embedding.special:
                mask_emb = vampnet.embedding.special['MASK'][i]
                emb.weight.data[vampnet.vocab_size] = mask_emb
        
        embeddings.append(emb)
    
    return embeddings, vampnet, codec


def test_v9_model():
    """Test V9 model with proper latent input."""
    print("\n" + "="*80)
    print("TESTING V9 MODEL WITH PROPER LATENT FLOW")
    print("="*80)
    
    # Load VampNet for comparison
    from vampnet.modules.transformer import VampNet
    from lac.model.lac import LAC as DAC
    
    checkpoint_path = "models/vampnet/coarse.pth"
    codec_path = "models/vampnet/codec.pth"
    
    vampnet = VampNet.load(checkpoint_path, map_location='cpu')
    vampnet.eval()
    codec = DAC.load(Path(codec_path))
    codec.eval()
    
    # Create V9 model
    model = VampNetTransformerV9(
        n_codebooks=4,
        n_conditioning_codebooks=0,
        vocab_size=1024,
        latent_dim=8,
        d_model=1280,
        n_heads=20,
        n_layers=20
    )
    
    # Transfer weights
    transfer_weights_v9(checkpoint_path, model, codec_path)
    model.eval()
    
    # Test with proper flow
    torch.manual_seed(42)
    codes = torch.randint(0, 1024, (1, 4, 10))
    mask = torch.zeros((1, 4, 10), dtype=torch.bool)
    mask[:, :, 5:] = True
    masked_codes = codes.clone()
    masked_codes[mask] = 1024
    
    print(f"\nTest input codes: {masked_codes.shape}")
    
    with torch.no_grad():
        # VampNet flow
        print("\n1. VampNet flow:")
        vampnet_latents = vampnet.embedding.from_codes(masked_codes, codec)
        print(f"   Latents from from_codes: {vampnet_latents.shape}")
        vampnet_out = vampnet(vampnet_latents)
        print(f"   Output: {vampnet_out.shape}")
        
        # Our flow (using same latents)
        print("\n2. V9 model with same latents:")
        v9_out = model(vampnet_latents)
        print(f"   Output: {v9_out.shape}")
        
        # Compare
        print("\n3. Comparison:")
        # Reshape VampNet output
        vampnet_reshaped = vampnet_out.reshape(1, 4, 10, 1024)
        v9_truncated = v9_out[:, :, :, :1024]
        
        diff = (vampnet_reshaped - v9_truncated).abs()
        corr = np.corrcoef(vampnet_reshaped.flatten(), v9_truncated.flatten())[0,1]
        
        print(f"   Mean absolute difference: {diff.mean():.6f}")
        print(f"   Max absolute difference: {diff.max():.6f}")
        print(f"   Correlation: {corr:.4f}")
        
        if corr > 0.99:
            print("\n✅ SUCCESS! V9 model matches VampNet!")
        else:
            print("\n❌ Still not matching...")


def transfer_weights_v9(vampnet_checkpoint, model, codec_path):
    """Transfer weights to V9 model."""
    from vampnet.modules.transformer import VampNet
    from lac.model.lac import LAC as DAC
    
    vampnet = VampNet.load(vampnet_checkpoint, map_location='cpu')
    codec = DAC.load(Path(codec_path)) if codec_path else None
    
    print("Transferring weights to V9 model...")
    
    # 1. Transfer projection weights (no codec embeddings in this model!)
    model.embedding.out_proj.weight.data = vampnet.embedding.out_proj.weight.data
    model.embedding.out_proj.bias.data = vampnet.embedding.out_proj.bias.data
    print("✓ Transferred projection weights")
    
    # 2. Transfer transformer layers
    for i, (onnx_layer, vamp_layer) in enumerate(zip(model.layers, vampnet.transformer.layers)):
        # RMSNorm
        onnx_layer['norm_1'].weight.data = vamp_layer.norm_1.weight.data
        onnx_layer['norm_3'].weight.data = vamp_layer.norm_3.weight.data
        
        # Attention
        if i == 0:
            # Relative attention
            onnx_layer['self_attn'].w_qs.weight.data = vamp_layer.self_attn.w_qs.weight.data
            onnx_layer['self_attn'].w_ks.weight.data = vamp_layer.self_attn.w_ks.weight.data
            onnx_layer['self_attn'].w_vs.weight.data = vamp_layer.self_attn.w_vs.weight.data
            onnx_layer['self_attn'].fc.weight.data = vamp_layer.self_attn.fc.weight.data
            
            if hasattr(vamp_layer.self_attn, 'relative_attention_bias'):
                onnx_layer['self_attn'].relative_attention_bias.weight.data = \
                    vamp_layer.self_attn.relative_attention_bias.weight.data
        else:
            # Standard attention
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
            # Identity FiLM
            onnx_layer['film_3'].gamma_weight.data.fill_(0)
            onnx_layer['film_3'].gamma_bias.data.fill_(1)
            onnx_layer['film_3'].beta_weight.data.fill_(0)
            onnx_layer['film_3'].beta_bias.data.fill_(0)
        
        # FFN - VampNet uses bias=False
        onnx_layer['feed_forward'].w_1.weight.data = vamp_layer.feed_forward.w_1.weight.data
        onnx_layer['feed_forward'].w_2.weight.data = vamp_layer.feed_forward.w_2.weight.data
        # Copy activation weights (though GatedGELU has no parameters)
    
    # 3. Transfer final norm
    if hasattr(vampnet.transformer, 'norm'):
        model.final_norm.weight.data = vampnet.transformer.norm.weight.data
    
    # 4. Transfer output projections
    classifier_weight = vampnet.classifier.layers[0].weight.data.squeeze(-1)
    classifier_bias = vampnet.classifier.layers[0].bias.data
    n_output_codebooks = model.n_codebooks - model.n_conditioning_codebooks
    
    for i in range(n_output_codebooks):
        vamp_start = i * vampnet.vocab_size
        vamp_end = (i + 1) * vampnet.vocab_size
        
        # Transfer weights
        model.output_projs[i].weight.data[:vampnet.vocab_size] = classifier_weight[vamp_start:vamp_end]
        model.output_projs[i].weight.data[vampnet.vocab_size] = 0
        
        # Transfer bias
        model.output_projs[i].bias.data[:vampnet.vocab_size] = classifier_bias[vamp_start:vamp_end]
        model.output_projs[i].bias.data[vampnet.vocab_size] = 0
    
    print("✓ Weight transfer complete!")


if __name__ == "__main__":
    # Test the V9 model
    test_v9_model()
    
    print("\n" + "="*80)
    print("KEY INSIGHT:")
    print("VampNet's forward() expects LATENTS, not codes!")
    print("The ONNX model must receive pre-computed latents from codec embeddings.")
    print("="*80)