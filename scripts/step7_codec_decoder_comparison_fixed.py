#!/usr/bin/env python3
"""
Step 7: Codec Decoder Comparison - tokens to audio conversion.
Compare how VampNet and ONNX decoder convert tokens back to audio.
"""

import torch
import numpy as np
from pathlib import Path
import sys
import soundfile as sf
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))

from vampnet.modules.transformer import VampNet
from lac.model.lac import LAC as DAC
import onnxruntime as ort

print("="*80)
print("STEP 7: CODEC DECODER COMPARISON")
print("="*80)

# Load models
print("\n1. Loading models...")
torch.manual_seed(42)

# Load VampNet to access its embedding layer
vampnet = VampNet.load("models/vampnet/coarse.pth", map_location='cpu')
vampnet.eval()

# VampNet codec
codec = DAC.load(Path("models/vampnet/codec.pth"))
codec.eval()

# Check for ONNX decoder
onnx_decoder_path = "scripts/models/vampnet_codec_decoder.onnx"
if not Path(onnx_decoder_path).exists():
    onnx_decoder_path = "onnx_models/vampnet_codec_decoder.onnx"
    if not Path(onnx_decoder_path).exists():
        print(f"❌ ONNX decoder not found at {onnx_decoder_path}")
        print("   Please run export scripts first")
        sys.exit(1)

print(f"✓ Found ONNX decoder at {onnx_decoder_path}")
ort_session = ort.InferenceSession(onnx_decoder_path)

# Understand the decoder
print("\n2. Understanding the decoder pipeline...")
print("   VampNet decoding process:")
print("   1. codes → latents (via vampnet.embedding.from_codes)")
print("   2. latents → quantized (via codec.quantizer.from_latents)")
print("   3. quantized → audio (via codec.decode)")

# Check ONNX decoder inputs/outputs
print("\n3. ONNX decoder info:")
for i, input_info in enumerate(ort_session.get_inputs()):
    print(f"   Input {i}: {input_info.name} - shape {input_info.shape}")
for i, output_info in enumerate(ort_session.get_outputs()):
    print(f"   Output {i}: {output_info.name} - shape {output_info.shape}")

# Create test cases
print("\n4. Creating test cases...")
test_cases = []

# Test case 1: Simple pattern
codes1 = torch.randint(0, 1024, (1, 4, 20))
test_cases.append(("Simple random codes", codes1))

# Test case 2: Repeated pattern
codes2 = torch.zeros((1, 4, 30), dtype=torch.long)
codes2[:, :, ::2] = 100  # Alternating pattern
test_cases.append(("Alternating pattern", codes2))

# Test case 3: Real-like codes
codes3 = torch.randint(0, 1024, (1, 4, 50))
# Add some structure
codes3[:, 0, :] = torch.randint(0, 200, (1, 50))  # Lower values in first codebook
codes3[:, 1, :] = torch.randint(200, 400, (1, 50))  # Mid values in second
test_cases.append(("Structured codes", codes3))

# Run comparisons
print("\n5. Running decoder comparisons...")
results = []

for test_name, codes in test_cases:
    print(f"\n{test_name}:")
    print(f"  Input codes shape: {codes.shape}")
    
    with torch.no_grad():
        # VampNet decoding process
        # Step 1: codes → latents
        latents = vampnet.embedding.from_codes(codes, codec)
        print(f"  Latents shape: {latents.shape}")
        
        # Step 2: latents → quantized 
        quantized, _, _ = codec.quantizer.from_latents(latents)
        print(f"  Quantized shape: {quantized.shape}")
        
        # Step 3: quantized → audio
        vampnet_audio_dict = codec.decode(quantized)
        vampnet_audio = vampnet_audio_dict["audio"]
        print(f"  VampNet audio shape: {vampnet_audio.shape}")
        
        # ONNX decoding
        # The ONNX decoder expects codes directly and does the full pipeline
        try:
            # ONNX decoder takes 14 codebooks (VampNet uses 4)
            # Pad with zeros for missing codebooks
            codes_padded = torch.zeros((1, 14, codes.shape[2]), dtype=torch.long)
            codes_padded[:, :4, :] = codes
            
            onnx_audio = ort_session.run(None, {'codes': codes_padded.numpy()})[0]
            print(f"  ONNX audio shape: {onnx_audio.shape}")
        except Exception as e:
            print(f"  ❌ ONNX error: {e}")
            # Try different input formats
            input_name = ort_session.get_inputs()[0].name
            print(f"  ONNX expects input named: {input_name}")
            continue
        
        # Convert to numpy for comparison
        vampnet_np = vampnet_audio.squeeze().cpu().numpy()
        onnx_np = onnx_audio.squeeze()
        
        # Ensure same length for comparison
        min_len = min(len(vampnet_np), len(onnx_np))
        vampnet_np = vampnet_np[:min_len]
        onnx_np = onnx_np[:min_len]
        
        # Compute metrics
        mse = np.mean((vampnet_np - onnx_np) ** 2)
        max_diff = np.max(np.abs(vampnet_np - onnx_np))
        
        # Handle edge case where one signal might be zero
        if np.std(vampnet_np) > 0 and np.std(onnx_np) > 0:
            correlation = np.corrcoef(vampnet_np.flatten(), onnx_np.flatten())[0, 1]
        else:
            correlation = 0.0
        
        # Signal-to-noise ratio
        signal_power = np.mean(vampnet_np ** 2)
        noise_power = np.mean((vampnet_np - onnx_np) ** 2)
        snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
        
        results.append({
            'name': test_name,
            'shape': codes.shape,
            'mse': mse,
            'max_diff': max_diff,
            'correlation': correlation,
            'snr': snr,
            'audio_length': min_len
        })
        
        print(f"  MSE: {mse:.6f}")
        print(f"  Max difference: {max_diff:.6f}")
        print(f"  Correlation: {correlation:.4f}")
        print(f"  SNR: {snr:.2f} dB")

# Detailed analysis with one example
print("\n\n6. Detailed analysis with audio samples...")
codes = torch.randint(0, 1024, (1, 4, 100))

with torch.no_grad():
    # VampNet full decoding
    latents = vampnet.embedding.from_codes(codes, codec)
    quantized, _, _ = codec.quantizer.from_latents(latents)
    vampnet_audio_dict = codec.decode(quantized)
    vampnet_audio = vampnet_audio_dict["audio"]
    
    # ONNX decoding
    codes_padded = torch.zeros((1, 14, codes.shape[2]), dtype=torch.long)
    codes_padded[:, :4, :] = codes
    onnx_audio = ort_session.run(None, {'codes': codes_padded.numpy()})[0]
    
    # Convert to numpy
    vampnet_np = vampnet_audio.squeeze().cpu().numpy()
    onnx_np = onnx_audio.squeeze()
    
    # Ensure same length
    min_len = min(len(vampnet_np), len(onnx_np))
    vampnet_np = vampnet_np[:min_len]
    onnx_np = onnx_np[:min_len]
    
    # Save audio samples
    print("\n  Saving audio samples...")
    sf.write('vampnet_decoder_output.wav', vampnet_np, codec.sample_rate)
    sf.write('onnx_decoder_output.wav', onnx_np, codec.sample_rate)
    print("  ✓ Saved vampnet_decoder_output.wav")
    print("  ✓ Saved onnx_decoder_output.wav")
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    # Plot waveforms
    plt.subplot(3, 1, 1)
    time_samples = min(5000, len(vampnet_np))
    time = np.arange(time_samples) / codec.sample_rate
    plt.plot(time, vampnet_np[:time_samples], label='VampNet', alpha=0.7)
    plt.plot(time, onnx_np[:time_samples], label='ONNX', alpha=0.7)
    plt.title('Waveform Comparison (first 0.3s)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    
    # Plot difference
    plt.subplot(3, 1, 2)
    diff = vampnet_np[:time_samples] - onnx_np[:time_samples]
    plt.plot(time, diff)
    plt.title('Difference (VampNet - ONNX)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    # Plot spectrograms
    plt.subplot(3, 1, 3)
    # Simple spectrogram using FFT
    from scipy import signal
    f, t, Sxx = signal.spectrogram(vampnet_np, codec.sample_rate, nperseg=512)
    max_freq_idx = min(len(f[f<4000]), Sxx.shape[0])
    max_time_idx = min(int(2/t[-1]*len(t)), Sxx.shape[1])
    plt.pcolormesh(t[:max_time_idx], f[:max_freq_idx], 
                   10 * np.log10(Sxx[:max_freq_idx, :max_time_idx] + 1e-10))
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title('VampNet Spectrogram')
    plt.colorbar(label='dB')
    
    plt.tight_layout()
    plt.savefig('codec_decoder_comparison.png', dpi=150)
    print("  ✓ Saved codec_decoder_comparison.png")

# Summary
print("\n\n7. SUMMARY")
print("="*80)

if results:
    print("\nDecoding accuracy across test cases:")
    for r in results:
        print(f"\n{r['name']}:")
        print(f"  Correlation: {r['correlation']:.4f}")
        print(f"  SNR: {r['snr']:.2f} dB")
        print(f"  MSE: {r['mse']:.6f}")
    
    valid_results = [r for r in results if r['correlation'] != 0.0]
    if valid_results:
        avg_corr = np.mean([r['correlation'] for r in valid_results])
        avg_snr = np.mean([r['snr'] for r in valid_results if r['snr'] != float('inf')])
        print(f"\nAverage correlation: {avg_corr:.4f}")
        if not np.isnan(avg_snr):
            print(f"Average SNR: {avg_snr:.2f} dB")

print("\n✅ Key Findings:")
print("  - VampNet decoding: codes → latents → quantized → audio")
print("  - Uses LAC codec with 14 codebooks (VampNet uses 4)")
print("  - ONNX decoder handles full pipeline internally")
print("  - Need to pad codes to 14 codebooks for ONNX")

print("\n📝 Decoder Pipeline Details:")
print("  1. vampnet.embedding.from_codes(codes, codec)")
print("  2. codec.quantizer.from_latents(latents)")
print("  3. codec.decode(quantized)")
print(f"  - Output sample rate: {codec.sample_rate}Hz")

print("\n⚠️  Note:")
print("  - VampNet uses 4 codebooks, LAC codec expects 14")
print("  - Extra codebooks are padded with zeros")
print("  - This may affect audio quality comparison")

print("\n✅ Step 7 Complete!")
print("   Next: Step 8 - End-to-End Pipeline comparison")