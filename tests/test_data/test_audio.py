#!/usr/bin/env python3
"""
Generate a simple test audio file for testing.
"""

import numpy as np
import soundfile as sf
from pathlib import Path

# Create test_data directory
test_data_dir = Path(__file__).parent
test_data_dir.mkdir(exist_ok=True)

# Generate a simple sine wave
duration = 3.0  # seconds
sample_rate = 44100
frequency = 440.0  # A4 note

t = np.linspace(0, duration, int(sample_rate * duration))
audio = 0.5 * np.sin(2 * np.pi * frequency * t)

# Add some harmonics to make it more interesting
audio += 0.2 * np.sin(2 * np.pi * frequency * 2 * t)  # Octave
audio += 0.1 * np.sin(2 * np.pi * frequency * 3 * t)  # Fifth

# Add envelope
envelope = np.exp(-t * 0.5)
audio = audio * envelope

# Normalize
audio = audio / np.abs(audio).max() * 0.8

# Save
output_path = test_data_dir / "short_audio.wav"
sf.write(output_path, audio, sample_rate)

print(f"Created test audio file: {output_path}")
print(f"Duration: {duration} seconds")
print(f"Sample rate: {sample_rate} Hz")