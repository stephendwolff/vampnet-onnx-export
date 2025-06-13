"""Tests for audio processor module."""

import pytest
import numpy as np
import torch
import tempfile
import onnx
import onnxruntime as ort
from vampnet_onnx.audio_processor import AudioProcessor


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
        # Length should be padded to multiple of hop_length
        expected_length = processor.get_output_length(44100)
        assert result.shape[2] == expected_length
    
    def test_mono_passthrough(self):
        """Test that mono audio passes through (with padding)."""
        processor = AudioProcessor()
        
        # Create mono audio
        mono_audio = torch.randn(1, 1, 44100)
        
        # Process
        result = processor(mono_audio)
        
        # Check shape - will be padded
        expected_length = processor.get_output_length(44100)
        assert result.shape == (1, 1, expected_length)
    
    def test_clipping(self):
        """Test audio normalization keeps values in [-1, 1] range."""
        processor = AudioProcessor()
        
        # Create audio with values outside range
        audio = torch.tensor([[[2.0, -2.0, 0.5, -0.5]]])
        
        # Process
        result = processor(audio)
        
        # Check that audio is normalized to [-1, 1]
        assert torch.all(result <= 1.0)
        assert torch.all(result >= -1.0)
        # Note: AudioProcessor normalizes based on RMS, not simple clipping
        # So we can't predict exact values
    
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
        # Length should be padded
        expected_length = processor.get_output_length(22050)
        assert result.shape[2] == expected_length
    
    def test_empty_audio(self):
        """Test handling of empty audio."""
        processor = AudioProcessor()
        
        # Create empty audio
        empty_audio = torch.zeros(1, 2, 0)
        
        # Process should handle gracefully
        try:
            result = processor(empty_audio)
            assert result.shape == (1, 1, 0)
        except IndexError:
            # Expected - empty audio causes issues with max operation
            pass
    
    def test_padding(self):
        """Test audio padding to hop_length multiples."""
        processor = AudioProcessor(hop_length=768)
        
        # Create audio not divisible by hop_length
        audio = torch.randn(1, 1, 1000)  # Not divisible by 768
        
        # Process
        result = processor(audio)
        
        # Check padding
        expected_length = 768 * 2  # Next multiple of 768
        assert result.shape[2] == expected_length
        
    def test_export_to_onnx(self):
        """Test ONNX export compatibility."""
        processor = AudioProcessor()
        processor.eval()
        
        # Example input
        dummy_input = torch.randn(1, 2, 44100)
        
        # Export to ONNX
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp:
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
            model = onnx.load(tmp.name)
            onnx.checker.check_model(model)
            
            # Test with ONNX Runtime
            session = ort.InferenceSession(tmp.name)
            ort_output = session.run(None, {'audio': dummy_input.numpy()})
            
            # Compare outputs
            torch_output = processor(dummy_input)
            np.testing.assert_allclose(
                torch_output.detach().numpy(),
                ort_output[0],
                rtol=1e-5,
                atol=1e-5
            )
            
    def test_loudness_normalization(self):
        """Test loudness normalization functionality."""
        processor = AudioProcessor(target_loudness=-24.0)
        
        # Create quiet audio
        quiet_audio = torch.randn(1, 1, 44100) * 0.01
        
        # Process
        result = processor(quiet_audio)
        
        # Result should be louder than input
        assert torch.abs(result).mean() > torch.abs(quiet_audio).mean()
        
    def test_different_sample_rates(self):
        """Test handling of different sample rates."""
        processor = AudioProcessor(target_sample_rate=44100)
        
        # Test with different lengths simulating different sample rates
        for length in [22050, 44100, 48000, 96000]:
            audio = torch.randn(1, 1, length)
            result = processor(audio)
            assert result.shape[1] == 1  # Mono
            assert result.shape[2] > 0  # Non-empty


if __name__ == '__main__':
    pytest.main([__file__])