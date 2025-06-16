"""
VampNet ONNX Interface - provides API parity with original VampNet interface.
"""

import numpy as np
import torch
import onnxruntime as ort
from pathlib import Path
from typing import Optional, Union, Tuple, Dict
import warnings

try:
    from audiotools import AudioSignal
    AUDIOTOOLS_AVAILABLE = True
except ImportError:
    AUDIOTOOLS_AVAILABLE = False
    warnings.warn("audiotools not available. Audio loading functionality will be limited.")


class AudioSignalCompat:
    """Compatibility class when audiotools is not available."""
    
    def __init__(self, audio_data: np.ndarray, sample_rate: int):
        if audio_data.ndim == 1:
            audio_data = audio_data[np.newaxis, np.newaxis, :]  # Add batch and channel dims
        elif audio_data.ndim == 2:
            audio_data = audio_data[np.newaxis, :]  # Add batch dim
        
        self.audio_data = torch.from_numpy(audio_data).float()
        self.sample_rate = sample_rate
        self.audio_length = audio_data.shape[-1] / sample_rate
    
    def clone(self):
        return AudioSignalCompat(
            self.audio_data.numpy().copy(), 
            self.sample_rate
        )
    
    def resample(self, target_sr: int):
        if self.sample_rate != target_sr:
            try:
                import resampy
                audio = self.audio_data.numpy().squeeze()
                resampled = resampy.resample(audio, self.sample_rate, target_sr)
                self.audio_data = torch.from_numpy(resampled).float()
                if self.audio_data.ndim == 1:
                    self.audio_data = self.audio_data.unsqueeze(0).unsqueeze(0)
                elif self.audio_data.ndim == 2:
                    self.audio_data = self.audio_data.unsqueeze(0)
                self.sample_rate = target_sr
            except ImportError:
                raise ImportError("resampy required for resampling when audiotools not available")
        return self
    
    def to_mono(self):
        if self.audio_data.shape[1] > 1:
            self.audio_data = self.audio_data.mean(dim=1, keepdim=True)
        return self
    
    def normalize(self, target_loudness: float = -24.0):
        # Simple RMS normalization
        rms = torch.sqrt(torch.mean(self.audio_data ** 2))
        target_rms = 10 ** (target_loudness / 20)
        if rms > 0:
            self.audio_data = self.audio_data * (target_rms / rms)
        return self
    
    def ensure_max_of_audio(self, max_val: float = 1.0):
        max_abs = torch.max(torch.abs(self.audio_data))
        if max_abs > max_val:
            self.audio_data = self.audio_data * (max_val / max_abs)
        return self
    
    @property
    def samples(self):
        return self.audio_data.squeeze(0)  # Remove batch dim


class Interface:
    """
    ONNX-based interface that mimics the original VampNet interface.
    
    This provides a compatible API for audio processing and generation
    using exported ONNX models.
    """
    
    def __init__(
        self,
        device: str = 'cpu',
        encoder_path: Optional[str] = None,
        decoder_path: Optional[str] = None,
        coarse_path: Optional[str] = None,
        c2f_path: Optional[str] = None,
        mask_generator_path: Optional[str] = None,
        loudness: float = -24.0,
    ):
        self.device = device
        self.loudness = loudness
        
        # Default paths
        default_model_dir = Path(__file__).parent.parent / "onnx_models"
        
        # Load models
        self.encoder_session = None
        self.decoder_session = None
        self.coarse_session = None
        self.c2f_session = None
        self.mask_generator_session = None
        
        if encoder_path:
            self.encoder_session = ort.InferenceSession(encoder_path)
        
        if decoder_path:
            self.decoder_session = ort.InferenceSession(decoder_path)
            
        if coarse_path:
            self.coarse_session = ort.InferenceSession(coarse_path)
            
        if c2f_path:
            self.c2f_session = ort.InferenceSession(c2f_path)
            
        if mask_generator_path:
            self.mask_generator_session = ort.InferenceSession(mask_generator_path)
        
        # Constants matching original VampNet
        self.sample_rate = 44100
        self.codec_sample_rate = 44100
        self.hop_length = 768
        self.n_codebooks = 14
        self.n_coarse_codebooks = 4
        self.seq_len = 100  # Fixed sequence length for ONNX models
        
    @classmethod
    def from_default_models(cls, device: str = 'cpu'):
        """Load interface with default ONNX model paths."""
        model_dir = Path(__file__).parent.parent / "onnx_models_fixed"
        script_model_dir = Path(__file__).parent.parent / "scripts" / "models"
        
        # Use v8 models with all fixes
        return cls(
            device=device,
            encoder_path=str(script_model_dir / "vampnet_encoder_prepadded.onnx"),
            decoder_path=str(script_model_dir / "vampnet_codec_decoder.onnx"),
            coarse_path=str(model_dir / "coarse_v8_film_fix.onnx"),
            c2f_path=str(model_dir / "c2f_v8_film_fix.onnx"),
        )
    
    def _preprocess(self, signal: Union['AudioSignal', 'AudioSignalCompat']) -> Union['AudioSignal', 'AudioSignalCompat']:
        """Preprocess audio signal to match VampNet requirements."""
        signal = (
            signal.clone()
            .resample(self.codec_sample_rate)
            .to_mono()
            .normalize(self.loudness)
            .ensure_max_of_audio(1.0)
        )
        
        # Pad to multiple of hop_length
        samples = signal.samples
        n_samples = samples.shape[-1]
        pad_amount = (self.hop_length - (n_samples % self.hop_length)) % self.hop_length
        
        if pad_amount > 0:
            if isinstance(samples, torch.Tensor):
                samples = torch.nn.functional.pad(samples, (0, pad_amount))
            else:
                samples = np.pad(samples, ((0, 0), (0, pad_amount)), mode='constant')
        
        signal.samples = samples
        return signal
    
    def encode(self, signal: Union['AudioSignal', 'AudioSignalCompat', np.ndarray]) -> np.ndarray:
        """
        Encode audio signal to discrete tokens.
        
        Args:
            signal: Audio signal or numpy array
            
        Returns:
            tokens: Shape [batch, n_codebooks, time]
        """
        if not self.encoder_session:
            raise ValueError("Encoder model not loaded")
        
        # Handle numpy array input
        if isinstance(signal, np.ndarray):
            if signal.ndim == 1:
                signal = AudioSignalCompat(signal, self.sample_rate)
            else:
                raise ValueError("Expected 1D numpy array for audio")
        
        # Preprocess
        signal = self._preprocess(signal)
        
        # Convert to numpy
        if isinstance(signal.samples, torch.Tensor):
            audio = signal.samples.numpy()
        else:
            audio = signal.samples
        
        # Ensure correct shape [batch, channels, samples]
        if audio.ndim == 1:
            audio = audio[np.newaxis, np.newaxis, :]
        elif audio.ndim == 2:
            audio = audio[np.newaxis, :]
            
        # Encode
        tokens = self.encoder_session.run(None, {'audio_padded': audio.astype(np.float32)})[0]
        
        # Truncate or pad to seq_len if needed
        if tokens.shape[2] > self.seq_len:
            tokens = tokens[:, :, :self.seq_len]
        elif tokens.shape[2] < self.seq_len:
            pad_length = self.seq_len - tokens.shape[2]
            tokens = np.pad(tokens, ((0, 0), (0, 0), (0, pad_length)), mode='constant')
        
        return tokens
    
    def decode(self, tokens: np.ndarray) -> Union['AudioSignal', 'AudioSignalCompat']:
        """
        Decode discrete tokens to audio signal.
        
        Args:
            tokens: Shape [batch, n_codebooks, time]
            
        Returns:
            Audio signal
        """
        if not self.decoder_session:
            raise ValueError("Decoder model not loaded")
        
        # Ensure int64 and clip to valid range
        tokens = tokens.astype(np.int64)
        tokens = np.clip(tokens, 0, 1023)
        
        # Decode
        audio = self.decoder_session.run(None, {'codes': tokens})[0]
        
        # Create AudioSignal
        if AUDIOTOOLS_AVAILABLE:
            signal = AudioSignal(audio, self.sample_rate)
        else:
            signal = AudioSignalCompat(audio[0], self.sample_rate)
            
        return signal
    
    def build_mask(
        self,
        z: np.ndarray,
        signal: Optional[Union['AudioSignal', 'AudioSignalCompat']] = None,
        rand_mask_intensity: float = 1.0,
        prefix_s: float = 0.0,
        suffix_s: float = 0.0,
        periodic_prompt: int = 7,
        periodic_prompt_width: int = 1,
        onset_mask_width: int = 0,
        _dropout: float = 0.0,
        upper_codebook_mask: int = 3,
        ncc: int = 0,
        **kwargs
    ) -> np.ndarray:
        """
        Build mask for generation matching VampNet's behavior.
        
        Args:
            z: Tokens [batch, n_codebooks, time]
            signal: Audio signal (for onset detection - not yet implemented)
            rand_mask_intensity: Random masking intensity [0, 1]
            prefix_s: Seconds to preserve at start
            suffix_s: Seconds to preserve at end
            periodic_prompt: Periodic prompting interval (0 = disabled)
            periodic_prompt_width: Width of periodic preservation
            onset_mask_width: Width for onset masking (0 = disabled)
            _dropout: Additional dropout rate
            upper_codebook_mask: First codebook to mask completely
            ncc: Number of conditioning codebooks (not used in ONNX)
            
        Returns:
            mask: Boolean mask [batch, n_codebooks, time] where True = masked
        """
        # Import here to avoid circular dependency
        try:
            from .mask_generator_proper import build_mask_proper
            
            # Use proper mask generator
            return build_mask_proper(
                z,
                rand_mask_intensity=rand_mask_intensity,
                prefix_s=prefix_s,
                suffix_s=suffix_s,
                periodic_prompt=periodic_prompt,
                periodic_width=periodic_prompt_width,
                upper_codebook_mask=upper_codebook_mask,
                dropout=_dropout,
                random_roll=False,
                seed=None,  # Use None for true randomness
                hop_length=self.hop_length,
                sample_rate=self.sample_rate
            )
        except ImportError:
            # Fallback to simple masking if proper implementation not available
            warnings.warn("Proper mask generator not available, using simple fallback")
            batch_size, n_codebooks, seq_len = z.shape
            mask = np.zeros_like(z, dtype=bool)
            
            # Simple masking strategy for ONNX
            # Mask middle portion
            mask_start = seq_len // 3
            mask_end = 2 * seq_len // 3
            
            # Apply to specified codebooks
            if upper_codebook_mask > 0:
                mask[:, :upper_codebook_mask, mask_start:mask_end] = True
            else:
                mask[:, :, mask_start:mask_end] = True
                
            return mask
    
    def vamp(
        self,
        z: np.ndarray,
        mask: Optional[np.ndarray] = None,
        temperature: float = 1.0,
        top_p: float = 0.9,
        return_mask: bool = False,
        **kwargs
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate tokens at masked positions.
        
        Args:
            z: Input tokens [batch, n_codebooks, time]
            mask: Boolean mask indicating positions to generate
            temperature: Sampling temperature
            top_p: Top-p sampling threshold
            return_mask: Whether to return the mask
            
        Returns:
            Generated tokens, optionally with mask
        """
        if not self.coarse_session or not self.c2f_session:
            raise ValueError("Generation models not loaded")
        
        if mask is None:
            mask = self.build_mask(z)
        
        # Two-stage generation
        z_coarse = self.coarse_vamp(z[:, :self.n_coarse_codebooks], 
                                    mask[:, :self.n_coarse_codebooks],
                                    temperature=temperature,
                                    top_p=top_p)
        
        # Prepare for C2F
        z_full = z.copy()
        z_full[:, :self.n_coarse_codebooks] = z_coarse
        
        # Generate fine tokens
        z_generated = self.coarse_to_fine(z_full, mask, 
                                         temperature=temperature*0.8,
                                         top_p=top_p)
        
        if return_mask:
            return z_generated, mask
        return z_generated
    
    def coarse_vamp(
        self,
        z: np.ndarray,
        mask: np.ndarray,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> np.ndarray:
        """Generate coarse tokens."""
        if not self.coarse_session:
            raise ValueError("Coarse model not loaded")
            
        from .sampling import temperature_sample, top_p_sample
        
        # Get logits
        logits = self.coarse_session.run(None, {
            'codes': z.astype(np.int64),
            'mask': mask
        })[0]
        
        # Sample
        batch_size, n_codebooks, seq_len, vocab_size = logits.shape
        sampled = z.copy()
        
        for b in range(batch_size):
            for c in range(n_codebooks):
                for t in range(seq_len):
                    if mask[b, c, t]:
                        probs = temperature_sample(logits[b, c, t], temperature)
                        probs = top_p_sample(probs, top_p)
                        # Exclude mask token (1024)
                        probs = probs[:-1] / probs[:-1].sum()
                        sampled[b, c, t] = np.random.choice(len(probs), p=probs)
        
        return sampled
    
    def coarse_to_fine(
        self,
        z: np.ndarray,
        mask: np.ndarray,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> np.ndarray:
        """Generate fine tokens conditioned on coarse."""
        if not self.c2f_session:
            raise ValueError("C2F model not loaded")
            
        from .sampling import temperature_sample, top_p_sample
        
        # Create C2F mask (only for fine codebooks)
        c2f_mask = mask.copy()
        c2f_mask[:, :self.n_coarse_codebooks, :] = False
        
        # Get logits
        logits = self.c2f_session.run(None, {
            'codes': z.astype(np.int64),
            'mask': c2f_mask
        })[0]
        
        # Sample
        batch_size, n_codebooks, seq_len, vocab_size = logits.shape
        sampled = z.copy()
        
        for b in range(batch_size):
            for c in range(self.n_coarse_codebooks, n_codebooks):
                for t in range(seq_len):
                    if c2f_mask[b, c, t]:
                        probs = temperature_sample(logits[b, c, t], temperature)
                        probs = top_p_sample(probs, top_p)
                        # Exclude mask token (1024)
                        probs = probs[:-1] / probs[:-1].sum()
                        sampled[b, c, t] = np.random.choice(len(probs), p=probs)
        
        return sampled
    
    def s2t(self, seconds: float) -> int:
        """Convert seconds to tokens."""
        return int(seconds * self.sample_rate / self.hop_length)
    
    def t2s(self, tokens: int) -> float:
        """Convert tokens to seconds."""
        return tokens * self.hop_length / self.sample_rate