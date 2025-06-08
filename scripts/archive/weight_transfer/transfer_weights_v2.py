"""
Transfer weights from VampNet checkpoints to v2 ONNX models.
This handles the complete weight transfer for both coarse and C2F models.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import re
import sys
import os
from collections import OrderedDict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.export_vampnet_transformer_v2 import VampNetTransformerV2
import vampnet


class WeightTransferV2:
    """Handle weight transfer from VampNet to v2 models."""
    
    def __init__(self):
        self.mapping_stats = {
            'mapped': 0,
            'skipped': 0,
            'failed': 0,
            'details': []
        }
    
    def analyze_checkpoint(self, checkpoint_path):
        """Analyze a VampNet checkpoint structure."""
        print(f"\n=== Analyzing {checkpoint_path} ===")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Get state dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        print(f"Total parameters: {len(state_dict)}")
        
        # Group by component
        components = {
            'embedding': [],
            'layers': {},
            'classifier': [],
            'other': []
        }
        
        for name, param in state_dict.items():
            if 'embedding' in name:
                components['embedding'].append((name, param.shape))
            elif 'classifier' in name or 'output' in name:
                components['classifier'].append((name, param.shape))
            elif re.search(r'layers?\.(\d+)', name):
                match = re.search(r'layers?\.(\d+)', name)
                layer_idx = int(match.group(1))
                if layer_idx not in components['layers']:
                    components['layers'][layer_idx] = []
                components['layers'][layer_idx].append((name, param.shape))
            else:
                components['other'].append((name, param.shape))
        
        # Print summary
        print(f"\nEmbedding parameters: {len(components['embedding'])}")
        for name, shape in components['embedding'][:3]:
            print(f"  {name}: {shape}")
        
        print(f"\nTransformer layers: {len(components['layers'])}")
        if components['layers']:
            print(f"  Layer 0 parameters:")
            for name, shape in list(components['layers'].get(0, []))[:5]:
                print(f"    {name}: {shape}")
        
        print(f"\nClassifier parameters: {len(components['classifier'])}")
        for name, shape in components['classifier'][:3]:
            print(f"  {name}: {shape}")
        
        return state_dict
    
    def map_embedding_weights(self, vampnet_state, onnx_model):
        """Map embedding weights."""
        print("\n--- Mapping Embedding Weights ---")
        
        # Look for VampNet embedding weights
        for vamp_name, vamp_param in vampnet_state.items():
            if 'embedding' in vamp_name and 'weight' in vamp_name:
                # Extract codebook index if present
                codebook_match = re.search(r'embeddings\.(\d+)\.weight', vamp_name)
                if codebook_match:
                    cb_idx = int(codebook_match.group(1))
                    if cb_idx < len(onnx_model.embedding.embeddings):
                        onnx_param = onnx_model.embedding.embeddings[cb_idx].weight
                        if vamp_param.shape == onnx_param.shape:
                            onnx_param.data.copy_(vamp_param)
                            self.mapping_stats['mapped'] += 1
                            self.mapping_stats['details'].append(
                                f"✓ {vamp_name} -> embedding.embeddings.{cb_idx}.weight"
                            )
                        else:
                            # Handle vocab size mismatch (1024 vs 1025)
                            if vamp_param.shape[0] == 1024 and onnx_param.shape[0] == 1025:
                                onnx_param.data[:1024].copy_(vamp_param)
                                # Initialize mask token embedding
                                onnx_param.data[1024].normal_(0, 0.02)
                                self.mapping_stats['mapped'] += 1
                                self.mapping_stats['details'].append(
                                    f"✓ {vamp_name} -> embedding.embeddings.{cb_idx}.weight (padded)"
                                )
        
        # Handle positional embeddings
        pos_candidates = [
            ('position_embed', 'pos_encoding'),
            ('pos_embed', 'pos_encoding'),
            ('positional_embedding', 'pos_encoding'),
        ]
        
        for vamp_key, onnx_attr in pos_candidates:
            for vamp_name, vamp_param in vampnet_state.items():
                if vamp_key in vamp_name:
                    onnx_param = getattr(onnx_model, onnx_attr, None)
                    if onnx_param is not None:
                        # Handle shape differences
                        min_len = min(vamp_param.shape[1], onnx_param.shape[1])
                        onnx_param.data[:, :min_len, :].copy_(vamp_param[:, :min_len, :])
                        self.mapping_stats['mapped'] += 1
                        self.mapping_stats['details'].append(
                            f"✓ {vamp_name} -> {onnx_attr} (first {min_len} positions)"
                        )
                        break
    
    def map_attention_weights(self, vampnet_state, onnx_model, layer_idx):
        """Map attention weights for a specific layer."""
        # Map attention components
        mappings = [
            # VampNet uses in_proj_weight that combines q,k,v
            ('in_proj_weight', None),  # Special handling below
            ('in_proj_bias', None),    # Special handling below
            ('out_proj.weight', 'self_attn.w_o.weight'),
            ('out_proj.bias', 'self_attn.w_o.bias'),
        ]
        
        # Handle in_proj (combined q,k,v)
        in_proj_weight_key = f'net.layers.{layer_idx}.self_attn.in_proj_weight'
        if in_proj_weight_key in vampnet_state:
            in_proj = vampnet_state[in_proj_weight_key]
            d_model = in_proj.shape[1]
            
            # Split into q, k, v
            q_weight, k_weight, v_weight = in_proj.chunk(3, dim=0)
            
            # Map to ONNX model
            onnx_layer = onnx_model.layers[layer_idx]
            onnx_layer['self_attn'].w_q.weight.data.copy_(q_weight)
            onnx_layer['self_attn'].w_k.weight.data.copy_(k_weight)
            onnx_layer['self_attn'].w_v.weight.data.copy_(v_weight)
            
            self.mapping_stats['mapped'] += 3
            self.mapping_stats['details'].append(
                f"✓ Layer {layer_idx} in_proj_weight -> w_q, w_k, w_v"
            )
        
        # Handle in_proj bias
        in_proj_bias_key = f'net.layers.{layer_idx}.self_attn.in_proj_bias'
        if in_proj_bias_key in vampnet_state:
            in_proj_bias = vampnet_state[in_proj_bias_key]
            
            # Split into q, k, v
            q_bias, k_bias, v_bias = in_proj_bias.chunk(3, dim=0)
            
            # Map to ONNX model
            onnx_layer = onnx_model.layers[layer_idx]
            onnx_layer['self_attn'].w_q.bias.data.copy_(q_bias)
            onnx_layer['self_attn'].w_k.bias.data.copy_(k_bias)
            onnx_layer['self_attn'].w_v.bias.data.copy_(v_bias)
            
            self.mapping_stats['mapped'] += 3
            self.mapping_stats['details'].append(
                f"✓ Layer {layer_idx} in_proj_bias -> w_q, w_k, w_v biases"
            )
        
        # Map output projection
        for vamp_suffix, onnx_path in mappings[2:]:  # Skip in_proj mappings
            vamp_key = f'net.layers.{layer_idx}.self_attn.{vamp_suffix}'
            if vamp_key in vampnet_state:
                # Navigate to the ONNX parameter
                parts = onnx_path.split('.')
                target = onnx_model.layers[layer_idx]
                for part in parts[:-1]:
                    target = target[part] if isinstance(target, dict) else getattr(target, part)
                
                setattr(target, parts[-1], nn.Parameter(vampnet_state[vamp_key].clone()))
                self.mapping_stats['mapped'] += 1
                self.mapping_stats['details'].append(f"✓ {vamp_key} -> layers[{layer_idx}].{onnx_path}")
    
    def map_ffn_weights(self, vampnet_state, onnx_model, layer_idx):
        """Map FFN weights for a specific layer."""
        # Check if using GatedFFN
        onnx_ffn = onnx_model.layers[layer_idx]['ffn']
        
        if hasattr(onnx_ffn, 'w_1'):  # GatedFFN
            # Map GatedFFN weights
            mappings = [
                ('mlp.w1.weight', 'ffn.w_1.weight'),
                ('mlp.w1.bias', 'ffn.w_1.bias'),
                ('mlp.w2.weight', 'ffn.w_2.weight'),
                ('mlp.w2.bias', 'ffn.w_2.bias'),
            ]
            
            for vamp_suffix, onnx_path in mappings:
                vamp_key = f'net.layers.{layer_idx}.{vamp_suffix}'
                if vamp_key in vampnet_state:
                    onnx_param = onnx_ffn.w_1 if 'w_1' in onnx_path else onnx_ffn.w_2
                    param_name = 'weight' if 'weight' in onnx_path else 'bias'
                    setattr(onnx_param, param_name, nn.Parameter(vampnet_state[vamp_key].clone()))
                    self.mapping_stats['mapped'] += 1
                    self.mapping_stats['details'].append(f"✓ {vamp_key} -> layers[{layer_idx}].{onnx_path}")
        else:
            # Standard FFN
            mappings = [
                ('mlp.0.weight', 'ffn.0.weight'),
                ('mlp.0.bias', 'ffn.0.bias'),
                ('mlp.2.weight', 'ffn.2.weight'),
                ('mlp.2.bias', 'ffn.2.bias'),
            ]
            
            for vamp_suffix, onnx_path in mappings:
                vamp_key = f'net.layers.{layer_idx}.{vamp_suffix}'
                if vamp_key in vampnet_state:
                    idx = int(onnx_path.split('.')[1])
                    param_name = onnx_path.split('.')[2]
                    setattr(onnx_ffn[idx], param_name, nn.Parameter(vampnet_state[vamp_key].clone()))
                    self.mapping_stats['mapped'] += 1
                    self.mapping_stats['details'].append(f"✓ {vamp_key} -> layers[{layer_idx}].{onnx_path}")
    
    def map_layer_norms(self, vampnet_state, onnx_model, layer_idx):
        """Map layer normalization weights."""
        mappings = [
            ('ln1.weight', 'norm1.weight'),
            ('ln2.weight', 'norm2.weight'),
        ]
        
        for vamp_suffix, onnx_suffix in mappings:
            vamp_key = f'net.layers.{layer_idx}.{vamp_suffix}'
            if vamp_key in vampnet_state:
                onnx_model.layers[layer_idx][onnx_suffix.split('.')[0]].weight.data.copy_(
                    vampnet_state[vamp_key]
                )
                self.mapping_stats['mapped'] += 1
                self.mapping_stats['details'].append(
                    f"✓ {vamp_key} -> layers[{layer_idx}].{onnx_suffix}"
                )
    
    def map_output_projections(self, vampnet_state, onnx_model):
        """Map output projection weights."""
        print("\n--- Mapping Output Projections ---")
        
        # Look for classifier weights
        for i in range(len(onnx_model.output_projs)):
            vamp_weight_key = f'classifier.{i}.weight'
            vamp_bias_key = f'classifier.{i}.bias'
            
            if vamp_weight_key in vampnet_state:
                vamp_weight = vampnet_state[vamp_weight_key]
                onnx_weight = onnx_model.output_projs[i].weight
                
                if vamp_weight.shape[0] == 1024 and onnx_weight.shape[0] == 1025:
                    # Handle vocab size difference
                    onnx_weight.data[:1024].copy_(vamp_weight)
                    onnx_weight.data[1024].normal_(0, 0.02)  # Initialize mask token row
                    self.mapping_stats['mapped'] += 1
                    self.mapping_stats['details'].append(
                        f"✓ {vamp_weight_key} -> output_projs[{i}].weight (padded)"
                    )
                elif vamp_weight.shape == onnx_weight.shape:
                    onnx_weight.data.copy_(vamp_weight)
                    self.mapping_stats['mapped'] += 1
                    self.mapping_stats['details'].append(
                        f"✓ {vamp_weight_key} -> output_projs[{i}].weight"
                    )
            
            if vamp_bias_key in vampnet_state:
                vamp_bias = vampnet_state[vamp_bias_key]
                onnx_bias = onnx_model.output_projs[i].bias
                
                if vamp_bias.shape[0] == 1024 and onnx_bias.shape[0] == 1025:
                    onnx_bias.data[:1024].copy_(vamp_bias)
                    onnx_bias.data[1024] = 0  # Initialize mask token bias
                    self.mapping_stats['mapped'] += 1
                    self.mapping_stats['details'].append(
                        f"✓ {vamp_bias_key} -> output_projs[{i}].bias (padded)"
                    )
                elif vamp_bias.shape == onnx_bias.shape:
                    onnx_bias.data.copy_(vamp_bias)
                    self.mapping_stats['mapped'] += 1
                    self.mapping_stats['details'].append(
                        f"✓ {vamp_bias_key} -> output_projs[{i}].bias"
                    )
    
    def transfer_all_weights(self, vampnet_state, onnx_model):
        """Transfer all weights from VampNet to ONNX model."""
        print("\n=== Starting Weight Transfer ===")
        
        # Reset stats
        self.mapping_stats = {
            'mapped': 0,
            'skipped': 0,
            'failed': 0,
            'details': []
        }
        
        # 1. Map embeddings
        self.map_embedding_weights(vampnet_state, onnx_model)
        
        # 2. Map transformer layers
        n_layers = len(onnx_model.layers)
        print(f"\n--- Mapping {n_layers} Transformer Layers ---")
        
        for i in range(n_layers):
            # Attention
            self.map_attention_weights(vampnet_state, onnx_model, i)
            
            # FFN
            self.map_ffn_weights(vampnet_state, onnx_model, i)
            
            # Layer norms
            self.map_layer_norms(vampnet_state, onnx_model, i)
            
            # FiLM (if present)
            film_weight = f'net.layers.{i}.film.weight'
            film_bias = f'net.layers.{i}.film.bias'
            
            if film_weight in vampnet_state:
                onnx_model.layers[i]['film'].weight.data.copy_(vampnet_state[film_weight])
                self.mapping_stats['mapped'] += 1
                
            if film_bias in vampnet_state:
                onnx_model.layers[i]['film'].bias.data.copy_(vampnet_state[film_bias])
                self.mapping_stats['mapped'] += 1
        
        # 3. Map final norm
        if 'net.final_norm.weight' in vampnet_state:
            onnx_model.final_norm.weight.data.copy_(vampnet_state['net.final_norm.weight'])
            self.mapping_stats['mapped'] += 1
            self.mapping_stats['details'].append("✓ net.final_norm.weight -> final_norm.weight")
        
        # 4. Map output projections
        self.map_output_projections(vampnet_state, onnx_model)
        
        # Print summary
        print(f"\n=== Transfer Summary ===")
        print(f"Mapped: {self.mapping_stats['mapped']}")
        print(f"Skipped: {self.mapping_stats['skipped']}")
        print(f"Failed: {self.mapping_stats['failed']}")
        
        return self.mapping_stats['mapped']


def transfer_coarse_weights():
    """Transfer weights to coarse transformer v2."""
    print("\n" + "="*60)
    print("TRANSFERRING WEIGHTS TO COARSE TRANSFORMER V2")
    print("="*60)
    
    # Load checkpoint
    checkpoint_path = Path("models/vampnet/coarse.pth")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Create model
    model = VampNetTransformerV2(
        n_codebooks=4,
        n_conditioning_codebooks=0,
        vocab_size=1024,
        d_model=1280,
        n_heads=20,
        n_layers=20,
        use_gated_ffn=True
    )
    
    # Transfer weights
    transfer = WeightTransferV2()
    vampnet_state = transfer.analyze_checkpoint(checkpoint_path)
    transferred = transfer.transfer_all_weights(vampnet_state, model)
    
    # Save model
    output_path = Path("onnx_models_fixed/coarse_transformer_v2_weights.pth")
    torch.save(model.state_dict(), output_path)
    print(f"\nSaved weighted model to {output_path}")
    
    return model, transferred


def transfer_c2f_weights():
    """Transfer weights to C2F transformer v2."""
    print("\n" + "="*60)
    print("TRANSFERRING WEIGHTS TO C2F TRANSFORMER V2")
    print("="*60)
    
    # Load checkpoint
    checkpoint_path = Path("models/vampnet/c2f.pth")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Create model
    model = VampNetTransformerV2(
        n_codebooks=14,  # Total codebooks
        n_conditioning_codebooks=0,
        vocab_size=1024,
        d_model=1280,
        n_heads=20,
        n_layers=16,  # C2F has 16 layers
        use_gated_ffn=True
    )
    
    # Transfer weights
    transfer = WeightTransferV2()
    vampnet_state = transfer.analyze_checkpoint(checkpoint_path)
    transferred = transfer.transfer_all_weights(vampnet_state, model)
    
    # Save model
    output_path = Path("onnx_models_fixed/c2f_transformer_v2_weights.pth")
    torch.save(model.state_dict(), output_path)
    print(f"\nSaved weighted model to {output_path}")
    
    return model, transferred


def main():
    """Transfer weights to both v2 models."""
    print("=== VampNet to ONNX V2 Weight Transfer ===\n")
    
    # Transfer coarse weights
    coarse_model, coarse_transferred = transfer_coarse_weights()
    
    # Transfer C2F weights
    c2f_model, c2f_transferred = transfer_c2f_weights()
    
    print("\n" + "="*60)
    print("WEIGHT TRANSFER COMPLETE")
    print("="*60)
    print(f"\nCoarse model: {coarse_transferred} weights transferred")
    print(f"C2F model: {c2f_transferred} weights transferred")
    print("\nNext step: Re-export models with transferred weights")


if __name__ == "__main__":
    main()