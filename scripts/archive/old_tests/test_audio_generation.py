"""
Test end-to-end audio generation with the ONNX transformer and codec.
"""

import torch
import numpy as np
import onnxruntime as ort
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from vampnet_onnx import VampNetONNXPipeline


def test_audio_generation_pipeline():
    """Test the complete audio generation pipeline."""
    
    print("=== Testing Audio Generation Pipeline ===\n")
    
    # Check for required models
    transformer_path = "vampnet_transformer_final.onnx"
    codec_encoder_path = "../vampnet_onnx/models/vampnet_codec_encoder.onnx"
    codec_decoder_path = "../vampnet_onnx/models/vampnet_codec_decoder.onnx"
    
    missing_models = []
    for path, name in [(transformer_path, "Transformer"), 
                       (codec_encoder_path, "Codec Encoder"),
                       (codec_decoder_path, "Codec Decoder")]:
        if not Path(path).exists():
            missing_models.append(f"{name}: {path}")
    
    if missing_models:
        print("❌ Missing required models:")
        for model in missing_models:
            print(f"   {model}")
        print("\nPlease ensure all models are exported first.")
        return
    
    # Initialize pipeline
    print("Initializing VampNet ONNX Pipeline...")
    try:
        pipeline = VampNetONNXPipeline(
            transformer_path=transformer_path,
            codec_encoder_path=codec_encoder_path,
            codec_decoder_path=codec_decoder_path
        )
        print("✓ Pipeline initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize pipeline: {e}")
        return
    
    # Test with different audio inputs
    test_generation_from_noise(pipeline)
    test_generation_from_audio(pipeline)
    test_iterative_refinement(pipeline)
    test_variation_generation(pipeline)


def test_generation_from_noise(pipeline):
    """Test generation from random noise."""
    
    print("\n=== Test 1: Generation from Noise ===")
    
    # Generate random codes
    batch_size = 1
    n_codebooks = 4
    seq_len = 100
    
    codes = np.random.randint(0, 1024, (batch_size, n_codebooks, seq_len), dtype=np.int64)
    
    # Create mask for full generation
    mask = np.ones_like(codes)  # Mask everything
    
    print(f"Input: Random codes {codes.shape}")
    print(f"Mask: Full generation (all positions masked)")
    
    # Generate
    try:
        generated_codes = pipeline.generate(codes, mask, temperature=0.95)
        print(f"✓ Generated codes: {generated_codes.shape}")
        
        # Decode to audio
        audio = pipeline.decode_codes(generated_codes)
        print(f"✓ Decoded audio: {audio.shape}")
        
        # Save audio
        output_path = "test_from_noise.wav"
        sf.write(output_path, audio[0].T, 16000)
        print(f"✓ Saved to {output_path}")
        
        # Analyze
        analyze_audio(audio[0], "Generation from Noise")
        
    except Exception as e:
        print(f"✗ Generation failed: {e}")
        import traceback
        traceback.print_exc()


def test_generation_from_audio(pipeline):
    """Test generation from existing audio."""
    
    print("\n\n=== Test 2: Generation from Audio ===")
    
    # Load test audio
    test_audio_path = "../assets/example.wav"
    if not Path(test_audio_path).exists():
        print(f"⚠️  Test audio not found: {test_audio_path}")
        # Create synthetic test audio
        print("Creating synthetic test audio...")
        duration = 3.0  # seconds
        sample_rate = 16000
        t = np.linspace(0, duration, int(duration * sample_rate))
        # Create a simple melody
        frequencies = [440, 494, 523, 587]  # A, B, C, D
        audio = np.zeros_like(t)
        for i, freq in enumerate(frequencies):
            start = int(i * len(t) / len(frequencies))
            end = int((i + 1) * len(t) / len(frequencies))
            audio[start:end] = 0.3 * np.sin(2 * np.pi * freq * t[start:end])
        audio = audio.reshape(1, -1)
    else:
        audio, sr = sf.read(test_audio_path)
        if sr != 16000:
            print(f"⚠️  Resampling from {sr}Hz to 16000Hz")
            # Simple decimation (not ideal but quick)
            audio = audio[::sr//16000]
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)
        elif audio.ndim == 2:
            audio = audio.T
    
    print(f"Input audio shape: {audio.shape}")
    
    # Encode to codes
    try:
        codes = pipeline.encode_audio(audio)
        print(f"✓ Encoded to codes: {codes.shape}")
        
        # Create partial mask (regenerate middle section)
        mask = np.zeros_like(codes)
        seq_len = codes.shape[2]
        mask[:, :, seq_len//3:2*seq_len//3] = 1
        
        print(f"Mask: Regenerating middle third ({mask.sum()} positions)")
        
        # Generate
        generated_codes = pipeline.generate(codes, mask, temperature=0.8)
        
        # Decode both original and generated
        original_audio = pipeline.decode_codes(codes)
        generated_audio = pipeline.decode_codes(generated_codes)
        
        # Save both
        sf.write("test_from_audio_original.wav", original_audio[0].T, 16000)
        sf.write("test_from_audio_generated.wav", generated_audio[0].T, 16000)
        print("✓ Saved original and generated audio")
        
        # Compare
        compare_audio(original_audio[0], generated_audio[0], "Original vs Generated")
        
    except Exception as e:
        print(f"✗ Generation failed: {e}")
        import traceback
        traceback.print_exc()


def test_iterative_refinement(pipeline):
    """Test iterative refinement process."""
    
    print("\n\n=== Test 3: Iterative Refinement ===")
    
    # Start with random codes
    codes = np.random.randint(0, 1024, (1, 4, 100), dtype=np.int64)
    
    print("Starting iterative refinement...")
    refinement_steps = 5
    audio_history = []
    
    for step in range(refinement_steps):
        # Decreasing mask ratio
        mask_ratio = 0.5 * (1 - step / refinement_steps)
        mask = np.zeros_like(codes)
        
        # Random masking
        for cb in range(4):
            positions = np.random.choice(100, int(100 * mask_ratio), replace=False)
            mask[0, cb, positions] = 1
        
        print(f"\nStep {step + 1}: Masking {mask_ratio*100:.0f}% of positions")
        
        # Generate
        codes = pipeline.generate(codes, mask, temperature=0.9 - step * 0.1)
        
        # Decode and save
        audio = pipeline.decode_codes(codes)
        audio_history.append(audio[0])
        
        sf.write(f"test_refinement_step_{step}.wav", audio[0].T, 16000)
    
    print("\n✓ Saved all refinement steps")
    
    # Visualize refinement
    visualize_refinement(audio_history)


def test_variation_generation(pipeline):
    """Test generating variations of the same input."""
    
    print("\n\n=== Test 4: Variation Generation ===")
    
    # Create a simple pattern
    codes = np.random.randint(0, 1024, (1, 4, 100), dtype=np.int64)
    
    # Make it somewhat repetitive (musical)
    pattern_len = 25
    for i in range(pattern_len, 100, pattern_len):
        if i + pattern_len <= 100:
            codes[:, :, i:i+pattern_len] = codes[:, :, :pattern_len]
    
    print("Generating variations with different temperatures...")
    
    temperatures = [0.7, 0.85, 1.0, 1.2]
    variations = []
    
    for temp in temperatures:
        # Mask some positions
        mask = np.zeros_like(codes)
        mask[:, :, ::3] = 1  # Every third position
        
        # Generate
        generated = pipeline.generate(codes.copy(), mask, temperature=temp)
        audio = pipeline.decode_codes(generated)
        variations.append(audio[0])
        
        sf.write(f"test_variation_temp_{temp}.wav", audio[0].T, 16000)
        print(f"✓ Generated variation with temperature {temp}")
    
    # Compare variations
    compare_variations(variations, temperatures)


def analyze_audio(audio, title):
    """Analyze and visualize audio properties."""
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    # Waveform
    axes[0].plot(audio[0])
    axes[0].set_title(f"{title} - Waveform")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_xlim(0, len(audio[0]))
    
    # Spectrogram
    from scipy import signal
    f, t, Sxx = signal.spectrogram(audio[0], 16000, nperseg=512)
    axes[1].pcolormesh(t, f[:256], 10 * np.log10(Sxx[:256] + 1e-10))
    axes[1].set_ylabel("Frequency [Hz]")
    axes[1].set_title("Spectrogram")
    
    # Energy over time
    window = 512
    energy = np.array([np.sum(audio[0][i:i+window]**2) 
                      for i in range(0, len(audio[0])-window, window)])
    axes[2].plot(energy)
    axes[2].set_title("Energy over Time")
    axes[2].set_xlabel("Time (windows)")
    axes[2].set_ylabel("Energy")
    
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}_analysis.png", dpi=150)
    plt.close()


def compare_audio(audio1, audio2, title):
    """Compare two audio signals."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Waveforms
    axes[0, 0].plot(audio1[0], alpha=0.7, label="Audio 1")
    axes[0, 0].plot(audio2[0], alpha=0.7, label="Audio 2")
    axes[0, 0].set_title("Waveform Comparison")
    axes[0, 0].legend()
    axes[0, 0].set_xlim(0, min(len(audio1[0]), len(audio2[0])))
    
    # Difference
    min_len = min(len(audio1[0]), len(audio2[0]))
    diff = audio1[0][:min_len] - audio2[0][:min_len]
    axes[0, 1].plot(diff)
    axes[0, 1].set_title("Difference Signal")
    
    # Spectrograms
    from scipy import signal
    for i, (audio, ax) in enumerate([(audio1, axes[1, 0]), (audio2, axes[1, 1])]):
        f, t, Sxx = signal.spectrogram(audio[0], 16000, nperseg=512)
        ax.pcolormesh(t, f[:256], 10 * np.log10(Sxx[:256] + 1e-10))
        ax.set_ylabel("Frequency [Hz]")
        ax.set_title(f"Spectrogram {i+1}")
    
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}_comparison.png", dpi=150)
    plt.close()


def visualize_refinement(audio_history):
    """Visualize the refinement process."""
    
    n_steps = len(audio_history)
    fig, axes = plt.subplots(n_steps, 1, figsize=(12, 2*n_steps))
    
    for i, audio in enumerate(audio_history):
        axes[i].plot(audio[0])
        axes[i].set_title(f"Refinement Step {i+1}")
        axes[i].set_ylabel("Amplitude")
        if i == n_steps - 1:
            axes[i].set_xlabel("Sample")
    
    plt.tight_layout()
    plt.savefig("refinement_process.png", dpi=150)
    plt.close()


def compare_variations(variations, temperatures):
    """Compare generated variations."""
    
    fig, axes = plt.subplots(len(variations), 1, figsize=(12, 2*len(variations)))
    
    for i, (audio, temp) in enumerate(zip(variations, temperatures)):
        axes[i].plot(audio[0], alpha=0.8)
        axes[i].set_title(f"Temperature {temp}")
        axes[i].set_ylabel("Amplitude")
        if i == len(variations) - 1:
            axes[i].set_xlabel("Sample")
    
    plt.tight_layout()
    plt.savefig("temperature_variations.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    # Test the complete pipeline
    test_audio_generation_pipeline()
    
    print("\n\n=== Summary ===")
    print("Audio generation tests completed!")
    print("\nGenerated files:")
    print("  - test_from_noise.wav")
    print("  - test_from_audio_*.wav")
    print("  - test_refinement_step_*.wav")
    print("  - test_variation_temp_*.wav")
    print("\nAnalysis plots:")
    print("  - *_analysis.png")
    print("  - *_comparison.png")
    print("  - refinement_process.png")
    print("  - temperature_variations.png")