"""
VampNet codec implementation for ONNX export.
This module provides wrappers for the actual VampNet codec model.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Union
import numpy as np

try:
    import vampnet
    import audiotools as at
    VAMPNET_AVAILABLE = True
except (ImportError, FileNotFoundError) as e:
    VAMPNET_AVAILABLE = False
    print(f"Warning: VampNet not available due to: {e}")
    print("Install with: pip install vampnet @ git+https://github.com/stephendwolff/vampnet.git")


class VampNetCodecEncoder(nn.Module):
    """
    Wrapper for the actual VampNet codec encoder for ONNX export.
    """
    
    def __init__(self, codec_model=None, device='cpu'):
        super().__init__()
        
        if not VAMPNET_AVAILABLE:
            raise ImportError("VampNet is not installed. Please install it first.")
        
        # Load the codec if not provided
        if codec_model is None:
            # Load VampNet interface to get codec
            interface = vampnet.interface.Interface.default()
            codec_model = interface.codec
            
        self.codec = codec_model
        self.device = device
        self.sample_rate = codec_model.sample_rate
        self.hop_length = codec_model.hop_length
        self.n_codebooks = codec_model.n_codebooks
        
        # Move codec to device
        self.codec.to(device)
        self.codec.eval()
        
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Encode audio to discrete tokens using VampNet codec.
        
        Args:
            audio: Preprocessed audio [batch, 1, samples]
            
        Returns:
            Token codes [batch, n_codebooks, sequence_length]
        """
        with torch.no_grad():
            # Ensure audio is on the correct device
            audio = audio.to(self.device)
            
            # The LAC codec expects audio in the right shape and normalized properly
            # Audio should already be preprocessed (mono, 44100Hz, normalized)
            
            # Pad audio to multiple of hop_length (matching original preprocess)
            length = audio.shape[-1]
            right_pad = ((length + self.hop_length - 1) // self.hop_length) * self.hop_length - length
            if right_pad > 0:
                audio = torch.nn.functional.pad(audio, (0, right_pad))
            
            # Use the codec's encode method directly
            # The LAC codec encode expects audio and sample_rate
            encoded = self.codec.encode(audio, self.sample_rate)
            
            # Extract codes from the encoded output
            if isinstance(encoded, dict):
                codes = encoded["codes"]
            else:
                # Some versions might return codes directly
                codes = encoded
                
        return codes


class VampNetCodecDecoder(nn.Module):
    """
    Wrapper for the actual VampNet codec decoder for ONNX export.
    """
    
    def __init__(self, codec_model=None, device='cpu'):
        super().__init__()
        
        if not VAMPNET_AVAILABLE:
            raise ImportError("VampNet is not installed. Please install it first.")
        
        # Load the codec if not provided
        if codec_model is None:
            # Load VampNet interface to get codec
            interface = vampnet.interface.Interface.default()
            codec_model = interface.codec
            
        self.codec = codec_model
        self.device = device
        self.sample_rate = codec_model.sample_rate
        self.hop_length = codec_model.hop_length
        
        # Move codec to device
        self.codec.to(device)
        self.codec.eval()
        
    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Decode tokens to audio using VampNet codec.
        
        Args:
            codes: Token codes [batch, n_codebooks, sequence_length]
            
        Returns:
            Decoded audio [batch, 1, samples]
        """
        with torch.no_grad():
            # Ensure codes are on the correct device
            codes = codes.to(self.device)
            
            # Convert to long tensor if needed
            if codes.dtype != torch.long:
                codes = codes.long()
            
            # LAC codec's decode method expects embeddings, not codes
            # First convert codes to embeddings using the quantizer
            z_q, _, _ = self.codec.quantizer.from_codes(codes)
            
            # Now decode the embeddings to audio
            output = self.codec.decode(z_q)
            
            # Extract audio from output (might be dict or tensor)
            if isinstance(output, dict):
                audio = output["audio"]
            else:
                audio = output
            
            # Ensure output shape is [batch, 1, samples]
            if audio.dim() == 2:
                audio = audio.unsqueeze(1)
                
        return audio


class VampNetCodecONNXWrapper(nn.Module):
    """
    Complete codec wrapper for ONNX export that handles both encoding and decoding.
    This is useful for testing the full codec pipeline.
    """
    
    def __init__(self, codec_model=None, device='cpu'):
        super().__init__()
        
        if not VAMPNET_AVAILABLE:
            raise ImportError("VampNet is not installed. Please install it first.")
        
        # Load the codec if not provided
        if codec_model is None:
            # Load VampNet interface to get codec
            interface = vampnet.interface.Interface.default()
            codec_model = interface.codec
            
        self.encoder = VampNetCodecEncoder(codec_model, device)
        self.decoder = VampNetCodecDecoder(codec_model, device)
        
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio to tokens."""
        return self.encoder(audio)
        
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode tokens to audio."""
        return self.decoder(codes)
        
    def forward(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full codec forward pass for testing.
        
        Args:
            audio: Input audio [batch, 1, samples]
            
        Returns:
            Tuple of (codes, reconstructed_audio)
        """
        codes = self.encode(audio)
        reconstructed = self.decode(codes)
        return codes, reconstructed


def prepare_codec_for_export(codec_model) -> Dict[str, nn.Module]:
    """
    Prepare VampNet codec components for ONNX export.
    
    Args:
        codec_model: VampNet codec model
        
    Returns:
        Dictionary with encoder and decoder modules ready for export
    """
    # Extract and prepare encoder
    encoder = VampNetCodecEncoder(codec_model)
    encoder.eval()
    
    # Extract and prepare decoder  
    decoder = VampNetCodecDecoder(codec_model)
    decoder.eval()
    
    return {
        'encoder': encoder,
        'decoder': decoder
    }


def test_vampnet_codec():
    """Test the VampNet codec wrappers."""
    if not VAMPNET_AVAILABLE:
        print("VampNet not available, skipping test")
        return
        
    print("Testing VampNet codec wrappers...")
    
    # Load codec
    interface = vampnet.interface.Interface.default()
    codec = interface.codec
    
    # Create wrappers
    encoder = VampNetCodecEncoder(codec)
    decoder = VampNetCodecDecoder(codec)
    
    # Test with dummy audio
    test_audio = torch.randn(1, 1, 44100)  # 1 second of audio
    
    # Encode
    codes = encoder(test_audio)
    print(f"Encoded shape: {codes.shape}")
    print(f"Codes dtype: {codes.dtype}")
    
    # Decode
    reconstructed = decoder(codes)
    print(f"Reconstructed shape: {reconstructed.shape}")
    
    # Check shapes
    expected_seq_len = test_audio.shape[-1] // codec.hop_length
    assert codes.shape[0] == 1  # batch
    assert codes.shape[1] == codec.n_codebooks
    assert codes.shape[2] == expected_seq_len
    
    assert reconstructed.shape[0] == 1  # batch
    assert reconstructed.shape[1] == 1  # channels
    # Audio length might be slightly different due to padding
    
    print("VampNet codec test passed!")
    
    
if __name__ == "__main__":
    test_vampnet_codec()