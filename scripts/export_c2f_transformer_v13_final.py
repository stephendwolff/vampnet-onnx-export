#!/usr/bin/env python3
"""
Export C2F (Coarse-to-Fine) Transformer V13 - Final version.
Matches VampNet exactly: expects latents as input, outputs in correct format.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys
import math
from einops import rearrange

sys.path.append(str(Path(__file__).parent.parent))


class NewGELU(nn.Module):
    """Gaussian Error Linear Unit activation function."""
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class VampNetGatedGELU(nn.Module):
    """VampNet's GatedGELU implementation."""
    def __init__(self):
        super().__init__()
        self.activation = NewGELU()
    
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * self.activation(gate)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        norm_x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * norm_x


class MultiHeadRelativeAttention(nn.Module):
    """Multi-head attention with relative position encoding."""
    def __init__(self, d_model, n_heads, dropout=0.0, bidirectional=True,
                 has_relative_attention_bias=True, attention_num_buckets=32,
                 attention_max_distance=128):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.has_relative_attention_bias = has_relative_attention_bias
        
        # Linear projections (no bias)
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.fc = nn.Linear(d_model, d_model, bias=False)
        
        # Relative attention bias (only if enabled)
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(attention_num_buckets, n_heads)
            self.attention_num_buckets = attention_num_buckets
            self.attention_max_distance = attention_max_distance
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        q = self.w_qs(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        k = self.w_ks(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        v = self.w_vs(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        
        # Transpose for attention
        q = q.transpose(1, 2)  # [batch, heads, seq, d_k]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Add relative position bias if enabled
        if self.has_relative_attention_bias:
            position_bias = self._get_relative_position_bias(seq_len)
            scores = scores + position_bias.unsqueeze(0)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.fc(attn_output)
        
        return output
    
    def _get_relative_position_bias(self, seq_len):
        """Compute relative position bias."""
        context_position = torch.arange(seq_len, dtype=torch.long)[:, None]
        memory_position = torch.arange(seq_len, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        
        # Map to buckets
        relative_buckets = self._relative_position_bucket(
            relative_position,
            bidirectional=self.bidirectional,
            num_buckets=self.attention_num_buckets,
            max_distance=self.attention_max_distance
        )
        
        # Get bias values
        values = self.relative_attention_bias(relative_buckets)
        values = values.permute([2, 0, 1])  # [heads, seq, seq]
        
        return values
    
    def _relative_position_bucket(self, relative_position, bidirectional=True,
                                  num_buckets=32, max_distance=128):
        """Map relative positions to buckets."""
        ret = 0
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))
        
        max_exact = num_buckets // 2
        is_small = n < max_exact
        
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / 
            math.log(max_distance / max_exact) * 
            (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        
        ret += torch.where(is_small, n, val_if_large)
        return ret


class FeedForward(nn.Module):
    """Feed-forward network with GatedGELU."""
    def __init__(self, d_model, d_inner, dropout=0.0):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_inner, bias=False)
        self.w_2 = nn.Linear(d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.activation = VampNetGatedGELU()
    
    def forward(self, x):
        x = self.w_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.w_2(x)
        return x


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer."""
    def __init__(self, d_model, cond_dim=0):
        super().__init__()
        self.cond_dim = cond_dim
        if cond_dim > 0:
            self.gamma = nn.Linear(cond_dim, d_model)
            self.beta = nn.Linear(cond_dim, d_model)
        else:
            # Identity FiLM
            self.register_buffer('gamma_weight', torch.zeros(d_model, cond_dim))
            self.register_buffer('gamma_bias', torch.ones(d_model))
            self.register_buffer('beta_weight', torch.zeros(d_model, cond_dim))
            self.register_buffer('beta_bias', torch.zeros(d_model))
    
    def forward(self, x, cond=None):
        if self.cond_dim > 0 and cond is not None:
            gamma = self.gamma(cond)
            beta = self.beta(cond)
            return x * gamma + beta
        else:
            # Identity transformation
            return x * self.gamma_bias + self.beta_bias


class VampNetEmbeddingLayer(nn.Module):
    """Embedding layer that takes LATENTS as input (not codes!)."""
    def __init__(self, n_codebooks: int, latent_dim: int, d_model: int):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.latent_dim = latent_dim
        self.out_proj = nn.Conv1d(n_codebooks * latent_dim, d_model, kernel_size=1)
    
    def forward(self, latents):
        """
        Args:
            latents: [batch, n_codebooks * latent_dim, seq_len]
        Returns:
            embeddings: [batch, d_model, seq_len]
        """
        return self.out_proj(latents)


class C2FTransformerV13(nn.Module):
    """C2F Transformer V13 - Final version matching VampNet exactly."""
    
    def __init__(self, 
                 n_codebooks: int = 14,  # C2F uses all 14 codebooks
                 n_conditioning_codebooks: int = 4,  # First 4 are conditioning
                 vocab_size: int = 1024,
                 d_model: int = 1280,
                 n_heads: int = 20,
                 n_layers: int = 16,  # C2F has 16 layers
                 dropout: float = 0.1,
                 latent_dim: int = 8):
        super().__init__()
        
        self.n_codebooks = n_codebooks
        self.n_conditioning_codebooks = n_conditioning_codebooks
        self.n_predict_codebooks = n_codebooks - n_conditioning_codebooks
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.latent_dim = latent_dim
        
        # Embedding layer - expects latents
        self.embedding = VampNetEmbeddingLayer(n_codebooks, latent_dim, d_model)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList()
        for i in range(n_layers):
            layer = nn.ModuleDict({
                'norm': RMSNorm(d_model),
                'self_attn': MultiHeadRelativeAttention(
                    d_model, n_heads, dropout,
                    bidirectional=True,
                    has_relative_attention_bias=(i == 0),  # Only layer 0
                    attention_num_buckets=32,
                    attention_max_distance=128
                ),
                'film': FiLMLayer(d_model, cond_dim=0),  # No conditioning
                'ffn': FeedForward(d_model, d_model * 2, dropout)
            })
            self.transformer_layers.append(layer)
        
        # Output norm
        self.norm_out = RMSNorm(d_model)
        
        # Single classifier that outputs all non-conditioning codebooks
        self.classifier = nn.Conv1d(
            d_model, 
            vocab_size * self.n_predict_codebooks,
            kernel_size=1
        )
        
        # Store mask token embeddings (for special tokens)
        self.register_buffer('mask_token_embeddings', 
                           torch.zeros(n_codebooks, latent_dim))
    
    def forward(self, latents):
        """
        Args:
            latents: [batch, n_codebooks * latent_dim, seq_len]
        Returns:
            logits: [batch, n_predict_codebooks, seq_len * n_predict_codebooks, vocab_size]
                    (matches VampNet's rearranged output format)
        """
        batch_size = latents.shape[0]
        seq_len = latents.shape[2]
        
        # Embedding
        x = self.embedding(latents)  # [batch, d_model, seq_len]
        
        # Rearrange for transformer
        x = rearrange(x, "b d n -> b n d")
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            # Pre-norm
            residual = x
            x = layer['norm'](x)
            
            # Self-attention
            x = layer['self_attn'](x)
            x = F.dropout(x, p=0.1, training=self.training)
            x = residual + x
            
            # FiLM (identity for C2F)
            x = layer['film'](x)
            
            # FFN
            residual = x
            x = layer['ffn'](x)
            x = F.dropout(x, p=0.1, training=self.training)
            x = residual + x
        
        # Output norm
        x = self.norm_out(x)
        
        # Rearrange back
        x = rearrange(x, "b n d -> b d n")
        
        # Classifier
        out = self.classifier(x)  # [batch, vocab_size * n_predict_codebooks, seq_len]
        
        # Rearrange to match VampNet's output format
        # "b (p c) t -> b p (t c)", c=self.n_predict_codebooks
        out = rearrange(out, "b (p c) t -> b p (t c)", c=self.n_predict_codebooks)
        
        return out


def transfer_weights_c2f_v13(checkpoint_path, model, codec_path):
    """Transfer weights from VampNet C2F checkpoint to our model."""
    print("Transferring weights to C2F V13 model...")
    
    # Load checkpoints
    from vampnet.modules.transformer import VampNet
    from lac.model.lac import LAC as DAC
    from torch.nn.utils import remove_weight_norm
    
    vampnet = VampNet.load(checkpoint_path, map_location='cpu')
    
    # Remove weight normalization from classifier
    try:
        remove_weight_norm(vampnet.classifier.layers[0])
        print("✓ Removed weight normalization from classifier")
    except:
        print("⚠ No weight normalization to remove")
    
    vampnet.eval()
    
    # 1. Transfer embedding projection (same as V11)
    with torch.no_grad():
        model.embedding.out_proj.weight.data = vampnet.embedding.out_proj.weight.data.clone()
        model.embedding.out_proj.bias.data = vampnet.embedding.out_proj.bias.data.clone()
    print("✓ Transferred embedding projection")
    
    # 2. Transfer mask token embeddings
    with torch.no_grad():
        if hasattr(vampnet.embedding, 'special') and 'MASK' in vampnet.embedding.special:
            model.mask_token_embeddings.data = vampnet.embedding.special['MASK'].data.clone()
            print("✓ Transferred mask token embeddings")
    
    # 3. Transfer transformer layers (same as V11)
    with torch.no_grad():
        for i in range(model.n_layers):
            src_layer = vampnet.transformer.layers[i]
            dst_layer = model.transformer_layers[i]
            
            # RMSNorm
            dst_layer['norm'].weight.data = src_layer.norm_1.weight.data.clone()
            
            # Attention (MultiHeadRelativeAttention for all layers)
            dst_layer['self_attn'].w_qs.weight.data = src_layer.self_attn.w_qs.weight.data.clone()
            dst_layer['self_attn'].w_ks.weight.data = src_layer.self_attn.w_ks.weight.data.clone()
            dst_layer['self_attn'].w_vs.weight.data = src_layer.self_attn.w_vs.weight.data.clone()
            dst_layer['self_attn'].fc.weight.data = src_layer.self_attn.fc.weight.data.clone()
            
            # Relative attention bias (only layer 0)
            if i == 0 and hasattr(src_layer.self_attn, 'relative_attention_bias'):
                dst_layer['self_attn'].relative_attention_bias.weight.data = \
                    src_layer.self_attn.relative_attention_bias.weight.data.clone()
            
            # FFN
            dst_layer['ffn'].w_1.weight.data = src_layer.feed_forward.w_1.weight.data.clone()
            dst_layer['ffn'].w_2.weight.data = src_layer.feed_forward.w_2.weight.data.clone()
    
    print(f"✓ Transferred {model.n_layers} transformer layers")
    
    # 4. Transfer output layer norm
    with torch.no_grad():
        model.norm_out.weight.data = vampnet.transformer.norm.weight.data.clone()
    print("✓ Transferred output norm")
    
    # 5. Transfer classifier weights
    with torch.no_grad():
        model.classifier.weight.data = vampnet.classifier.layers[0].weight.data.clone()
        model.classifier.bias.data = vampnet.classifier.layers[0].bias.data.clone()
    
    print("✓ Transferred classifier weights")
    print("✓ Weight transfer complete!")


def main():
    print("="*80)
    print("EXPORTING C2F TRANSFORMER V13 (FINAL)")
    print("="*80)
    
    # Create model
    model = C2FTransformerV13()
    print(f"\nModel configuration:")
    print(f"  n_codebooks: {model.n_codebooks}")
    print(f"  n_conditioning_codebooks: {model.n_conditioning_codebooks}")
    print(f"  n_predict_codebooks: {model.n_predict_codebooks}")
    print(f"  n_layers: {model.n_layers}")
    print(f"  d_model: {model.d_model}")
    print(f"  n_heads: {model.n_heads}")
    
    # Transfer weights
    transfer_weights_c2f_v13("models/vampnet/c2f.pth", model, "models/vampnet/codec.pth")
    
    # Test model
    print("\nTesting model...")
    model.eval()
    
    # Load codec to create proper latents
    from lac.model.lac import LAC as DAC
    codec = DAC.load(Path("models/vampnet/codec.pth"))
    codec.eval()
    
    with torch.no_grad():
        # Test input: create codes and convert to latents
        batch_size = 1
        seq_len = 10
        test_codes = torch.randint(0, 1024, (batch_size, 14, seq_len))
        
        # Convert codes to latents using codec
        z = codec.quantizer.from_codes(test_codes)
        # z is a tuple (quantized, latents, codes)
        # We want the latents which is z[1]
        if isinstance(z, tuple) and len(z) >= 2:
            test_latents = z[1]  # This has shape [batch, n_codebooks * latent_dim, seq_len]
        else:
            raise ValueError(f"Unexpected codec output format: {type(z)}")
        
        print(f"Input codes shape: {test_codes.shape}")
        print(f"Input latents shape: {test_latents.shape}")
        
        # Forward pass
        output = model(test_latents)
        print(f"Output shape: {output.shape}")
        print(f"Expected shape: [1, {model.n_predict_codebooks}, {seq_len * model.n_predict_codebooks}, {model.vocab_size}]")
    
    # Export to ONNX
    print("\nExporting to ONNX...")
    # Create dummy latents for export
    dummy_latents = torch.randn(1, 14 * 8, 100)
    
    torch.onnx.export(
        model,
        dummy_latents,
        "vampnet_c2f_transformer_v13.onnx",
        input_names=['latents'],
        output_names=['logits'],
        dynamic_axes={
            'latents': {0: 'batch', 2: 'sequence'},
            'logits': {0: 'batch', 2: 'sequence_expanded'}
        },
        opset_version=14,
        verbose=False
    )
    
    print("✓ Exported to vampnet_c2f_transformer_v13.onnx")
    
    # Save PyTorch checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'n_codebooks': model.n_codebooks,
            'n_conditioning_codebooks': model.n_conditioning_codebooks,
            'n_predict_codebooks': model.n_predict_codebooks,
            'vocab_size': model.vocab_size,
            'latent_dim': model.latent_dim,
            'd_model': model.d_model,
            'n_heads': model.n_heads,
            'n_layers': model.n_layers
        }
    }, 'vampnet_c2f_transformer_v13.pth')
    
    print("✓ Saved model to vampnet_c2f_transformer_v13.pth")
    
    # Test ONNX model
    print("\nTesting ONNX model...")
    import onnxruntime as ort
    ort_session = ort.InferenceSession("vampnet_c2f_transformer_v13.onnx")
    
    onnx_output = ort_session.run(None, {'latents': test_latents.numpy()})[0]
    print(f"ONNX output shape: {onnx_output.shape}")
    
    # Compare outputs
    pytorch_output = output.numpy()
    diff = np.abs(pytorch_output - onnx_output).max()
    print(f"Max difference PyTorch vs ONNX: {diff:.6f}")
    
    if diff < 1e-4:
        print("\n✅ C2F V13 export successful!")
    else:
        print("\n⚠️  Warning: Large difference between PyTorch and ONNX")
    
    # Compare with original VampNet C2F
    print("\n\nComparing with original VampNet C2F...")
    from vampnet.modules.transformer import VampNet
    vampnet_c2f = VampNet.load("models/vampnet/c2f.pth", map_location='cpu')
    vampnet_c2f.eval()
    
    with torch.no_grad():
        vampnet_output = vampnet_c2f(test_latents)
        print(f"VampNet C2F output shape: {vampnet_output.shape}")
        
        # Compare
        correlation = torch.corrcoef(torch.stack([
            vampnet_output.flatten(),
            torch.from_numpy(output.numpy()).flatten()
        ]))[0, 1]
        print(f"Correlation with VampNet: {correlation:.4f}")
    
    print("="*80)


if __name__ == "__main__":
    main()