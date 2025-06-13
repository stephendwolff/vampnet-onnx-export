"""Create a simple test audio file for testing."""

import numpy as np
import scipy.io.wavfile as wavfile
from pathlib import Path

# Create output directory
Path("test_audio").mkdir(exist_ok=True)

# Generate a simple sine wave
sample_rate = 44100
duration = 3.0  # seconds
frequency = 440.0  # A4 note

t = np.linspace(0, duration, int(sample_rate * duration))
audio = 0.5 * np.sin(2 * np.pi * frequency * t)

# Add some harmonics for more interesting sound
audio += 0.2 * np.sin(2 * np.pi * frequency * 2 * t)
audio += 0.1 * np.sin(2 * np.pi * frequency * 3 * t)

# Add envelope
envelope = np.exp(-t * 0.5)
audio = audio * envelope

# Normalize
audio = audio / np.max(np.abs(audio)) * 0.8

# Convert to 16-bit int
audio_int16 = (audio * 32767).astype(np.int16)

# Save
output_path = "test_audio/test_sine.wav"
wavfile.write(output_path, sample_rate, audio_int16)
print(f"Created test audio file: {output_path}")
print(f"Duration: {duration}s, Sample rate: {sample_rate}Hz")