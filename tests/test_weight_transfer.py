"""Tests for weight transfer functionality."""

import pytest
import numpy as np
import torch
import tempfile
from vampnet_onnx import (
    WeightTransferManager,
    complete_weight_transfer,
    CoarseTransformer,
    C2FTransformer
)


@pytest.fixture
def create_dummy_checkpoint():
    """Create a dummy VampNet checkpoint for testing."""
    def _create(include_c2f=False):
        state_dict = {
            # Codec embeddings
            'codec.quantizer.quantizers.0._codebook.embed': torch.randn(1024, 128),
            'codec.quantizer.quantizers.1._codebook.embed': torch.randn(1024, 128),
            'codec.quantizer.quantizers.2._codebook.embed': torch.randn(1024, 128),
            'codec.quantizer.quantizers.3._codebook.embed': torch.randn(1024, 128),
            
            # Mask tokens
            'embedding.mask_token.0': torch.randn(128),
            'embedding.mask_token.1': torch.randn(128),
            'embedding.mask_token.2': torch.randn(128),
            'embedding.mask_token.3': torch.randn(128),
            
            # Embedding projection
            'embedding.proj': torch.randn(512, 1280),
            
            # Transformer weights (layer 0)
            'transformer.layers.0.norm1.weight': torch.randn(1280),
            'transformer.layers.0.self_attn.qkv_proj.weight': torch.randn(3840, 1280),
            'transformer.layers.0.self_attn.o_proj.weight': torch.randn(1280, 1280),
            'transformer.layers.0.norm2.weight': torch.randn(1280),
            'transformer.layers.0.ffn.w_gated.weight': torch.randn(2560, 1280),
            'transformer.layers.0.ffn.w_up.weight': torch.randn(2560, 1280),
            'transformer.layers.0.ffn.w_down.weight': torch.randn(1280, 2560),
            
            # Final norm
            'transformer.final_norm.weight': torch.randn(1280),
            
            # Output projections
            'classifier.0.weight': torch.randn(4096, 1280),
            'classifier.1.weight': torch.randn(4096, 1280),
            'classifier.2.weight': torch.randn(4096, 1280),
            'classifier.3.weight': torch.randn(4096, 1280),
        }
        
        if include_c2f:
            # Add C2F weights
            state_dict.update({
                'c2f.transformer.layers.0.norm1.weight': torch.randn(768),
                'c2f.transformer.layers.0.self_attn.qkv_proj.weight': torch.randn(2304, 768),
                'c2f.transformer.layers.0.self_attn.o_proj.weight': torch.randn(768, 768),
                'c2f.transformer.layers.0.norm2.weight': torch.randn(768),
                'c2f.transformer.layers.0.ffn.w_gated.weight': torch.randn(1536, 768),
                'c2f.transformer.layers.0.ffn.w_up.weight': torch.randn(1536, 768),
                'c2f.transformer.layers.0.ffn.w_down.weight': torch.randn(768, 1536),
                'c2f.transformer.final_norm.weight': torch.randn(768),
            })
            
            # C2F output projections (10 fine codebooks)
            for i in range(10):
                state_dict[f'c2f.classifier.{i}.weight'] = torch.randn(4096, 768)
                
        return {'state_dict': state_dict}
    return _create


class TestWeightTransferManager:
    """Test weight transfer manager functionality."""
    
    def test_manager_initialization(self, create_dummy_checkpoint):
        """Test weight transfer manager initialization."""
        checkpoint = create_dummy_checkpoint()
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            torch.save(checkpoint, tmp.name)
            
            manager = WeightTransferManager(tmp.name)
            assert manager.checkpoint_path == tmp.name
            assert manager.checkpoint is None
            assert manager.state_dict is None
            
    def test_load_checkpoint(self, create_dummy_checkpoint):
        """Test checkpoint loading."""
        checkpoint = create_dummy_checkpoint()
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            torch.save(checkpoint, tmp.name)
            
            manager = WeightTransferManager(tmp.name)
            manager.load_checkpoint()
            
            assert manager.checkpoint is not None
            assert manager.state_dict is not None
            assert len(manager.state_dict) == len(checkpoint['state_dict'])
            
    def test_extract_codec_embeddings(self, create_dummy_checkpoint):
        """Test codec embedding extraction."""
        checkpoint = create_dummy_checkpoint()
        
        # Add all 14 codebooks
        for i in range(14):
            checkpoint['state_dict'][f'codec.quantizer.quantizers.{i}._codebook.embed'] = torch.randn(1024, 128)
            checkpoint['state_dict'][f'embedding.mask_token.{i}'] = torch.randn(128)
            
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            torch.save(checkpoint, tmp.name)
            
            manager = WeightTransferManager(tmp.name)
            embeddings = manager.extract_codec_embeddings()
            
            assert 'codebooks' in embeddings
            assert embeddings['codebooks'].shape == (14, 1024, 128)
            assert 'mask_token' in embeddings
            assert embeddings['mask_token'].shape == (14, 128)
            
    def test_transfer_transformer_weights(self, create_dummy_checkpoint):
        """Test transformer weight transfer."""
        checkpoint = create_dummy_checkpoint()
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            torch.save(checkpoint, tmp.name)
            
            # Create minimal model for testing
            model = CoarseTransformer(dim=1280, n_heads=20, n_layers=1)
            
            manager = WeightTransferManager(tmp.name)
            manager.transfer_transformer_weights(model, is_c2f=False)
            
            # Check that weights were transferred
            assert torch.equal(
                model.transformer[0].norm1.weight,
                checkpoint['state_dict']['transformer.layers.0.norm1.weight']
            )
            
    def test_transfer_output_projections(self, create_dummy_checkpoint):
        """Test output projection weight transfer."""
        checkpoint = create_dummy_checkpoint()
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            torch.save(checkpoint, tmp.name)
            
            model = CoarseTransformer(dim=1280, n_heads=20, n_layers=1)
            
            manager = WeightTransferManager(tmp.name)
            manager.transfer_output_projections(model, is_c2f=False)
            
            # Check dimension handling (4096 -> 1025)
            for i in range(4):
                assert model.output_proj[i].weight.shape[0] == 1025
                assert torch.equal(
                    model.output_proj[i].weight,
                    checkpoint['state_dict'][f'classifier.{i}.weight'][:1025]
                )


class TestCompleteWeightTransfer:
    """Test complete weight transfer functionality."""
    
    def test_complete_transfer_coarse(self, create_dummy_checkpoint):
        """Test complete weight transfer for coarse model."""
        checkpoint = create_dummy_checkpoint()
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            torch.save(checkpoint, tmp.name)
            
            # Create model
            coarse_model = CoarseTransformer(dim=1280, n_heads=20, n_layers=1)
            
            # Transfer weights
            results = complete_weight_transfer(
                checkpoint_path=tmp.name,
                coarse_model=coarse_model,
                return_embeddings=True
            )
            
            assert 'coarse_model' in results
            assert 'embeddings' in results
            assert results['coarse_model'] is coarse_model
            
    def test_complete_transfer_c2f(self, create_dummy_checkpoint):
        """Test complete weight transfer for C2F model."""
        checkpoint = create_dummy_checkpoint(include_c2f=True)
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            torch.save(checkpoint, tmp.name)
            
            # Create model
            c2f_model = C2FTransformer(dim=768, n_heads=12, n_layers=1)
            
            # Transfer weights
            results = complete_weight_transfer(
                checkpoint_path=tmp.name,
                c2f_model=c2f_model,
                return_embeddings=True
            )
            
            assert 'c2f_model' in results
            assert results['c2f_model'] is c2f_model
            
    def test_complete_transfer_both_models(self, create_dummy_checkpoint):
        """Test transferring weights to both coarse and C2F models."""
        checkpoint = create_dummy_checkpoint(include_c2f=True)
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            torch.save(checkpoint, tmp.name)
            
            # Create both models
            coarse_model = CoarseTransformer(dim=1280, n_heads=20, n_layers=1)
            c2f_model = C2FTransformer(dim=768, n_heads=12, n_layers=1)
            
            # Transfer weights
            results = complete_weight_transfer(
                checkpoint_path=tmp.name,
                coarse_model=coarse_model,
                c2f_model=c2f_model,
                return_embeddings=True
            )
            
            assert 'coarse_model' in results
            assert 'c2f_model' in results
            assert 'embeddings' in results
            
    def test_weight_transfer_without_embeddings(self, create_dummy_checkpoint):
        """Test weight transfer without returning embeddings."""
        checkpoint = create_dummy_checkpoint()
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            torch.save(checkpoint, tmp.name)
            
            coarse_model = CoarseTransformer(dim=1280, n_heads=20, n_layers=1)
            
            results = complete_weight_transfer(
                checkpoint_path=tmp.name,
                coarse_model=coarse_model,
                return_embeddings=False
            )
            
            assert 'coarse_model' in results
            assert 'embeddings' not in results


class TestWeightConsistency:
    """Test weight transfer consistency and correctness."""
    
    def test_weight_shapes_preserved(self, create_dummy_checkpoint):
        """Test that weight shapes are preserved during transfer."""
        checkpoint = create_dummy_checkpoint()
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            torch.save(checkpoint, tmp.name)
            
            model = CoarseTransformer(dim=1280, n_heads=20, n_layers=1)
            original_shapes = {
                name: param.shape for name, param in model.named_parameters()
            }
            
            # Transfer weights
            complete_weight_transfer(
                checkpoint_path=tmp.name,
                coarse_model=model
            )
            
            # Check shapes unchanged
            for name, param in model.named_parameters():
                assert param.shape == original_shapes[name]
                
    def test_no_nan_weights(self, create_dummy_checkpoint):
        """Test that transferred weights don't contain NaN values."""
        checkpoint = create_dummy_checkpoint()
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            torch.save(checkpoint, tmp.name)
            
            model = CoarseTransformer(dim=1280, n_heads=20, n_layers=1)
            
            complete_weight_transfer(
                checkpoint_path=tmp.name,
                coarse_model=model
            )
            
            # Check for NaN
            for name, param in model.named_parameters():
                assert not torch.isnan(param).any(), f"NaN found in {name}"
                
    def test_weight_magnitude_reasonable(self, create_dummy_checkpoint):
        """Test that transferred weights have reasonable magnitudes."""
        checkpoint = create_dummy_checkpoint()
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            torch.save(checkpoint, tmp.name)
            
            model = CoarseTransformer(dim=1280, n_heads=20, n_layers=1)
            
            complete_weight_transfer(
                checkpoint_path=tmp.name,
                coarse_model=model
            )
            
            # Check magnitudes
            for name, param in model.named_parameters():
                mean_magnitude = torch.abs(param).mean().item()
                assert mean_magnitude < 100, f"Weight {name} has excessive magnitude"
                assert mean_magnitude > 1e-6, f"Weight {name} is too small"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])