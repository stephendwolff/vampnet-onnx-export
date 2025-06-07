"""Tests for audio processor module."""

import pytest
import numpy as np
import torch
from src.audio_processor import AudioProcessor


class TestAudioProcessor:
    """Test cases for AudioProcessor class."""
    
    def test_initialization(self):
        """Test processor initialization."""
        processor = AudioProcessor()
        assert processor.target_sample_rate == 44100
        assert processor.target_loudness == -24.0
    
    def test_stereo_to_mono(self):
        """Test stereo to mono conversion."""
        processor = AudioProcessor()
        
        # Create stereo audio
        stereo_audio = torch.randn(1, 2, 44100)
        
        # Process
        result = processor(stereo_audio)
        
        # Check shape
        assert result.shape[1] == 1  # Mono
        assert result.shape[2] == 44100  # Same length
    
    def test_mono_passthrough(self):
        """Test that mono audio passes through unchanged."""
        processor = AudioProcessor()
        
        # Create mono audio
        mono_audio = torch.randn(1, 1, 44100)
        
        # Process
        result = processor(mono_audio)
        
        # Check shape unchanged
        assert result.shape == mono_audio.shape
    
    def test_clipping(self):
        """Test audio clipping to [-1, 1] range."""
        processor = AudioProcessor()
        
        # Create audio with values outside range
        audio = torch.tensor([[[2.0, -2.0, 0.5, -0.5]]])
        
        # Process
        result = processor(audio)
        
        # Check clipping
        assert torch.all(result <= 1.0)
        assert torch.all(result >= -1.0)
        assert result[0, 0, 0] == 1.0  # Clipped from 2.0
        assert result[0, 0, 1] == -1.0  # Clipped from -2.0
    
    def test_batch_processing(self):
        """Test processing multiple audio samples."""
        processor = AudioProcessor()
        
        # Create batch
        batch_audio = torch.randn(4, 2, 22050)
        
        # Process
        result = processor(batch_audio)
        
        # Check batch size preserved
        assert result.shape[0] == 4
        assert result.shape[1] == 1  # Mono
        assert result.shape[2] == 22050
    
    def test_empty_audio(self):
        """Test handling of empty audio."""
        processor = AudioProcessor()
        
        # Create empty audio
        empty_audio = torch.zeros(1, 2, 0)
        
        # Process should handle gracefully
        result = processor(empty_audio)
        assert result.shape == (1, 1, 0)
    
    def test_export_to_onnx(self):
        """Test ONNX export compatibility."""
        processor = AudioProcessor()
        processor.eval()
        
        # Example input
        dummy_input = torch.randn(1, 2, 44100)
        
        # Export to ONNX
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
            torch.onnx.export(
                processor,
                dummy_input,
                tmp.name,
                input_names=['audio'],
                output_names=['processed_audio'],
                dynamic_axes={
                    'audio': {0: 'batch', 2: 'samples'},
                    'processed_audio': {0: 'batch', 2: 'samples'}
                },
                opset_version=14
            )
            
            # Verify export
            import onnx
            model = onnx.load(tmp.name)
            onnx.checker.check_model(model)


if __name__ == '__main__':
    pytest.main([__file__])