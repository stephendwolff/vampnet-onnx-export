"""
Comprehensive test of VampNet generation with ONNX models.
"""

import numpy as np
import onnxruntime as ort
import soundfile as sf
from pathlib import Path
import matplotlib.pyplot as plt
import sys

sys.path.append(str(Path(__file__).parent.parent))


def create_periodic_mask(shape, period=7, n_codebooks_to_mask=1):
    """Create VampNet-style periodic mask."""
    batch, n_codebooks, seq_len = shape
    mask = np.zeros(shape, dtype=np.int64)
    
    # Mask periodic positions in coarse codebooks
    for cb in range(min(n_codebooks_to_mask, n_codebooks)):
        for i in range(0, seq_len, period):
            mask[:, cb, i] = 1
    
    return mask


def create_prefix_suffix_mask(shape, prefix_len=10, suffix_len=10):
    """Create mask that preserves prefix and suffix."""
    batch, n_codebooks, seq_len = shape
    mask = np.ones(shape, dtype=np.int64)
    
    # Don't mask prefix and suffix
    mask[:, :, :prefix_len] = 0
    mask[:, :, -suffix_len:] = 0
    
    return mask


def create_onset_mask(shape, onset_positions):
    """Create mask focused on musical onsets."""
    batch, n_codebooks, seq_len = shape
    mask = np.zeros(shape, dtype=np.int64)
    
    # Mask around onset positions
    for pos in onset_positions:
        start = max(0, pos - 2)
        end = min(seq_len, pos + 3)
        mask[:, :2, start:end] = 1  # Focus on first two codebooks
    
    return mask


def test_vampnet_generation():
    """Test VampNet generation with different masking strategies."""
    
    print("=== VampNet Generation Test ===\n")
    
    # Load models
    transformer_path = "vampnet_transformer_final.onnx"
    encoder_path = "../models/codec_encoder.onnx"
    decoder_path = "../models/codec_decoder.onnx"
    
    if not all(Path(p).exists() for p in [transformer_path, encoder_path, decoder_path]):
        print("❌ Missing required models!")
        return
    
    print("Loading models...")
    transformer = ort.InferenceSession(transformer_path)
    encoder = ort.InferenceSession(encoder_path)
    decoder = ort.InferenceSession(decoder_path)
    print("✓ All models loaded\n")
    
    # Test 1: Music continuation
    print("1. Testing music continuation...")
    test_music_continuation(transformer, encoder, decoder)
    
    # Test 2: Variation generation
    print("\n2. Testing variation generation...")
    test_variation_generation(transformer, encoder, decoder)
    
    # Test 3: Rhythm transfer
    print("\n3. Testing rhythm transfer...")
    test_rhythm_transfer(transformer, encoder, decoder)
    
    # Test 4: Progressive generation
    print("\n4. Testing progressive generation...")
    test_progressive_generation(transformer, encoder, decoder)


def test_music_continuation(transformer, encoder, decoder):
    """Test continuing a musical phrase."""
    
    # Create a simple musical pattern
    duration = 3.0
    sr = 16000
    t = np.linspace(0, duration, int(duration * sr))
    
    # Create a melody that stops halfway
    melody = np.zeros_like(t)
    notes = [440, 493.88, 523.25, 587.33]  # A, B, C, D
    note_duration = 0.25  # seconds
    
    for i, freq in enumerate(notes * 2):  # Repeat pattern
        start_time = i * note_duration
        if start_time < duration / 2:  # Only first half
            start_idx = int(start_time * sr)
            end_idx = int((start_time + note_duration * 0.8) * sr)
            if end_idx < len(t):
                melody[start_idx:end_idx] = 0.3 * np.sin(2 * np.pi * freq * t[start_idx:end_idx])
    
    # Encode
    audio = melody.reshape(1, 1, -1).astype(np.float32)
    codes = encoder.run(None, {'audio': audio})[0]
    
    # Adjust sequence length
    seq_len = codes.shape[2]
    if seq_len != 100:
        if seq_len < 100:
            codes = np.pad(codes, ((0, 0), (0, 0), (0, 100 - seq_len)), constant_values=0)
        else:
            codes = codes[:, :, :100]
    
    # Mask the second half (continue the melody)
    mask = np.zeros_like(codes[:, :4, :])
    mask[:, :, 50:] = 1  # Mask second half
    
    # Generate continuation
    generated = transformer.run(None, {'codes': codes[:, :4, :], 'mask': mask})[0]
    
    # Combine with fine codes
    full_codes = np.concatenate([generated, codes[:, 4:, :]], axis=1)
    
    # Decode
    if seq_len < 100:
        full_codes = full_codes[:, :, :seq_len]
    
    output = decoder.run(None, {'codes': full_codes})[0]
    
    # Save
    sf.write("test_continuation.wav", output[0].T, sr)
    print("✓ Saved music continuation to test_continuation.wav")


def test_variation_generation(transformer, encoder, decoder):
    """Test generating variations with different temperatures."""
    
    # Create base pattern
    base_codes = np.random.randint(200, 400, (1, 4, 100), dtype=np.int64)
    
    # Make it somewhat repetitive
    pattern = base_codes[:, :, :25].copy()
    for i in range(25, 100, 25):
        base_codes[:, :, i:i+25] = pattern
    
    temperatures = [0.7, 0.9, 1.1]
    
    for temp_idx, temp in enumerate(temperatures):
        # Periodic mask
        mask = create_periodic_mask(base_codes.shape, period=8, n_codebooks_to_mask=2)
        
        # Generate with temperature (simplified - just adds noise)
        codes_with_noise = base_codes.copy()
        noise = np.random.randn(*base_codes.shape) * temp * 50
        codes_with_noise = np.clip(codes_with_noise + noise.astype(int), 0, 1023)
        
        # Generate
        generated = transformer.run(None, {
            'codes': codes_with_noise.astype(np.int64),
            'mask': mask
        })[0]
        
        # Create full codes (pad with zeros for fine codes)
        full_codes = np.zeros((1, 14, 100), dtype=np.int64)
        full_codes[:, :4, :] = generated
        
        # Decode
        output = decoder.run(None, {'codes': full_codes})[0]
        
        # Save
        sf.write(f"test_variation_temp_{temp}.wav", output[0].T, 16000)
    
    print("✓ Saved variations with different temperatures")


def test_rhythm_transfer(transformer, encoder, decoder):
    """Test transferring rhythm pattern to new content."""
    
    # Create rhythm pattern (drums-like)
    duration = 2.0
    sr = 16000
    t = np.linspace(0, duration, int(duration * sr))
    
    # Simple kick-snare pattern
    rhythm = np.zeros_like(t)
    beat_duration = 0.125  # eighth note at 120 BPM
    
    for i in range(0, int(duration / beat_duration)):
        start_idx = int(i * beat_duration * sr)
        end_idx = int((i + 0.05) * beat_duration * sr)  # Short burst
        
        if i % 4 == 0:  # Kick on 1
            rhythm[start_idx:end_idx] = 0.5 * np.sin(2 * np.pi * 60 * t[start_idx:end_idx])
        elif i % 4 == 2:  # Snare on 3
            rhythm[start_idx:end_idx] = 0.3 * np.random.randn(end_idx - start_idx)
    
    # Encode rhythm
    audio = rhythm.reshape(1, 1, -1).astype(np.float32)
    rhythm_codes = encoder.run(None, {'audio': audio})[0]
    
    # Create new melodic content
    melody_codes = np.random.randint(400, 700, rhythm_codes.shape, dtype=np.int64)
    
    # Adjust to 100 length
    if rhythm_codes.shape[2] != 100:
        target_len = min(rhythm_codes.shape[2], 100)
        rhythm_codes = rhythm_codes[:, :, :target_len]
        melody_codes = melody_codes[:, :, :target_len]
        
        if target_len < 100:
            pad_len = 100 - target_len
            rhythm_codes = np.pad(rhythm_codes, ((0, 0), (0, 0), (0, pad_len)), constant_values=0)
            melody_codes = np.pad(melody_codes, ((0, 0), (0, 0), (0, pad_len)), constant_values=0)
    
    # Transfer rhythm (keep rhythm codebooks, replace melody)
    combined = rhythm_codes.copy()
    combined[:, 2:4, :] = melody_codes[:, 2:4, :]  # Replace upper codebooks
    
    # Mask to refine
    mask = np.zeros((1, 4, 100), dtype=np.int64)
    mask[:, 2:4, ::4] = 1  # Sparse mask on melody codebooks
    
    # Generate
    generated = transformer.run(None, {
        'codes': combined[:, :4, :],
        'mask': mask
    })[0]
    
    # Full codes
    full_codes = np.zeros((1, 14, 100), dtype=np.int64)
    full_codes[:, :4, :] = generated
    
    # Decode
    output = decoder.run(None, {'codes': full_codes})[0]
    
    # Save
    sf.write("test_rhythm_transfer.wav", output[0].T, sr)
    print("✓ Saved rhythm transfer result")


def test_progressive_generation(transformer, encoder, decoder):
    """Test progressive refinement from noise to music."""
    
    # Start with random codes
    codes = np.random.randint(0, 1024, (1, 4, 100), dtype=np.int64)
    
    refinement_steps = 4
    outputs = []
    
    for step in range(refinement_steps):
        # Decreasing mask density
        mask_ratio = 0.8 * (1 - step / refinement_steps)
        mask = np.zeros((1, 4, 100), dtype=np.int64)
        
        # Random positions to mask
        n_positions = int(100 * mask_ratio)
        for cb in range(4):
            positions = np.random.choice(100, n_positions, replace=False)
            mask[0, cb, positions] = 1
        
        # Generate
        codes = transformer.run(None, {'codes': codes, 'mask': mask})[0]
        
        # Decode this step
        full_codes = np.zeros((1, 14, 100), dtype=np.int64)
        full_codes[:, :4, :] = codes
        
        output = decoder.run(None, {'codes': full_codes})[0]
        outputs.append(output)
        
        # Save intermediate
        sf.write(f"test_progressive_step_{step}.wav", output[0].T, 16000)
    
    print("✓ Saved progressive generation steps")
    
    # Visualize progression
    visualize_progression(outputs)


def visualize_progression(outputs):
    """Visualize the progressive generation."""
    
    n_steps = len(outputs)
    fig, axes = plt.subplots(n_steps, 1, figsize=(12, 2*n_steps))
    
    for i, output in enumerate(outputs):
        audio = output[0, 0]
        
        # Simple envelope
        window_size = 1000
        envelope = np.array([
            np.sqrt(np.mean(audio[j:j+window_size]**2))
            for j in range(0, len(audio) - window_size, window_size)
        ])
        
        axes[i].plot(envelope)
        axes[i].set_title(f'Step {i+1} - Energy Envelope')
        axes[i].set_ylabel('RMS')
        
        if i == n_steps - 1:
            axes[i].set_xlabel('Time (windows)')
    
    plt.tight_layout()
    plt.savefig('progressive_generation.png', dpi=150)
    plt.close()
    print("✓ Saved visualization to progressive_generation.png")


def analyze_generation_quality():
    """Analyze the quality of generated audio."""
    
    print("\n=== Generation Quality Analysis ===")
    
    # Check if files exist
    files = [
        "test_continuation.wav",
        "test_variation_temp_0.7.wav", 
        "test_rhythm_transfer.wav",
        "test_progressive_step_3.wav"
    ]
    
    for file in files:
        if Path(file).exists():
            audio, sr = sf.read(file)
            
            # Basic statistics
            print(f"\n{file}:")
            print(f"  Duration: {len(audio)/sr:.2f}s")
            print(f"  RMS: {np.sqrt(np.mean(audio**2)):.4f}")
            print(f"  Peak: {np.max(np.abs(audio)):.4f}")
            
            # Check for silence
            silence_threshold = 0.001
            silence_ratio = np.sum(np.abs(audio) < silence_threshold) / len(audio)
            print(f"  Silence ratio: {silence_ratio*100:.1f}%")


if __name__ == "__main__":
    test_vampnet_generation()
    analyze_generation_quality()
    
    print("\n✅ All VampNet generation tests completed!")
    print("\nGenerated files:")
    print("  - test_continuation.wav")
    print("  - test_variation_temp_*.wav")
    print("  - test_rhythm_transfer.wav")
    print("  - test_progressive_step_*.wav")
    print("  - progressive_generation.png")