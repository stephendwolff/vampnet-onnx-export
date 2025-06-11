#!/usr/bin/env python3
"""
Compare VampNet vs ONNX audio generation using the pre-padded encoder.
This version handles the fixed-size ONNX encoder output correctly.
"""

import torch
import numpy as np
import onnxruntime as ort
from pathlib import Path
import soundfile as sf
from audiotools import AudioSignal
import vampnet
from vampnet import mask as pmask
import matplotlib.pyplot as plt
from scipy import signal
import warnings
warnings.filterwarnings("ignore")


def pad_audio_for_encoder(audio, hop_length=768):
    """Pad audio to multiple of hop_length for the ONNX encoder."""
    if audio.ndim == 1:
        audio = audio[np.newaxis, np.newaxis, :]
    elif audio.ndim == 2:
        audio = audio[np.newaxis, :]
    
    batch, channels, samples = audio.shape
    padded_samples = ((samples + hop_length - 1) // hop_length) * hop_length
    pad_amount = padded_samples - samples
    
    if pad_amount > 0:
        audio = np.pad(audio, ((0, 0), (0, 0), (0, pad_amount)), mode='constant')
    
    return audio, samples


def create_matching_mask(mask_ratio, n_codebooks, seq_len, periodic_prompt=30):
    """Create a mask matching VampNet's style."""
    mask = np.ones((1, n_codebooks, seq_len), dtype=np.int64)
    
    if periodic_prompt > 0:
        # Keep every periodic_prompt-th token
        for i in range(seq_len):
            if i % periodic_prompt == 0:
                mask[:, :, i] = 0  # 0 means keep, 1 means mask
    
    # Additionally mask some random positions
    n_mask = int(seq_len * mask_ratio)
    for cb in range(n_codebooks):
        # Get positions that aren't already unmasked
        available_positions = []
        for i in range(seq_len):
            if mask[0, cb, i] == 1:  # Currently masked
                available_positions.append(i)
        
        # Randomly select positions to unmask
        if len(available_positions) > n_mask:
            positions_to_mask = np.random.choice(
                available_positions, 
                size=len(available_positions) - n_mask, 
                replace=False
            )
            for pos in positions_to_mask:
                mask[0, cb, pos] = 0
    
    return mask


def compare_audio_generation():
    """Compare VampNet and ONNX audio generation."""
    
    print("=== VampNet vs ONNX Audio Generation Comparison ===\n")
    
    # Load test audio
    audio_path = Path("assets/stargazing.wav")
    if not audio_path.exists():
        audio_path = Path("assets/example.wav")
    
    audio_np, sr = sf.read(audio_path)
    duration = len(audio_np) / sr
    print(f"Test audio: {audio_path.name} ({duration:.1f}s at {sr}Hz)")
    
    # Ensure audio is the right sample rate
    if sr != 44100:
        import resampy
        audio_np = resampy.resample(audio_np, sr, 44100)
        sr = 44100
    
    # 1. Setup VampNet
    print("\n1. Setting up VampNet...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    interface = vampnet.interface.Interface(
        device=device,
        codec_ckpt="models/vampnet/codec.pth",
        coarse_ckpt="models/vampnet/coarse.pth",
        coarse2fine_ckpt="models/vampnet/c2f.pth",
        wavebeat_ckpt="models/vampnet/wavebeat.pth"
    )
    
    print(f"  Device: {device}")
    print(f"  Codec hop length: {interface.codec.hop_length}")
    
    # 2. Encode with VampNet
    print("\n2. Encoding with VampNet...")
    sig = AudioSignal(audio_np[np.newaxis, :], sample_rate=sr)
    sig = interface._preprocess(sig)
    
    with torch.no_grad():
        z_vampnet = interface.encode(sig)
    
    print(f"  Encoded shape: {z_vampnet.shape}")
    print(f"  Token range: [{z_vampnet.min().item()}, {z_vampnet.max().item()}]")
    
    # 3. Create mask for generation
    mask_ratio = 0.7
    periodic_prompt = 30
    
    # VampNet mask
    mask_vampnet = interface.build_mask(
        z_vampnet,
        sig,
        periodic_prompt=periodic_prompt,
        upper_codebook_mask=3
    )
    
    masked_positions = (mask_vampnet == 0).sum().item()
    total_positions = mask_vampnet.numel()
    print(f"\n3. Masking:")
    print(f"  Masked positions: {masked_positions}/{total_positions} ({masked_positions/total_positions:.1%})")
    
    # 4. Generate with VampNet
    print("\n4. Generating with VampNet...")
    import time
    start_time = time.time()
    
    with torch.no_grad():
        z_generated_vampnet, _ = interface.vamp(
            z_vampnet,
            mask=mask_vampnet,
            temperature=1.0,
            top_p=0.9,
            return_mask=True
        )
    
    vampnet_time = time.time() - start_time
    print(f"  Generation time: {vampnet_time:.2f}s")
    
    # Decode VampNet
    audio_vampnet = interface.decode(z_generated_vampnet)
    audio_vampnet_np = audio_vampnet.audio_data.squeeze(0).cpu().numpy()
    
    # 5. Process with ONNX
    print("\n5. Processing with ONNX...")
    
    # Load ONNX models
    encoder_path = Path("scripts/models/vampnet_encoder_prepadded.onnx")
    decoder_path = Path("scripts/models/vampnet_codec_decoder.onnx")
    coarse_path = Path("scripts/onnx_models_fixed/coarse_transformer_v2_weighted.onnx")
    c2f_path = Path("scripts/onnx_models_fixed/c2f_transformer_v2_weighted.onnx")
    
    if not encoder_path.exists():
        print(f"ERROR: Encoder not found at {encoder_path}")
        return
    
    # Create sessions
    encoder_session = ort.InferenceSession(str(encoder_path))
    decoder_session = ort.InferenceSession(str(decoder_path))
    coarse_session = ort.InferenceSession(str(coarse_path))
    c2f_session = ort.InferenceSession(str(c2f_path))
    
    # Encode with ONNX
    print("\n  5a. ONNX Encoding...")
    audio_padded, original_length = pad_audio_for_encoder(audio_np.astype(np.float32))
    
    start_time = time.time()
    codes_onnx = encoder_session.run(None, {'audio_padded': audio_padded})[0]
    encode_time = time.time() - start_time
    
    print(f"    Padded shape: {audio_padded.shape}")
    print(f"    Encoded shape: {codes_onnx.shape}")
    print(f"    Encoding time: {encode_time:.3f}s")
    
    # Handle size mismatch
    if codes_onnx.shape[2] != z_vampnet.shape[2]:
        print(f"\n  ⚠️ Size mismatch: ONNX {codes_onnx.shape[2]} vs VampNet {z_vampnet.shape[2]}")
        print(f"    Using first {min(codes_onnx.shape[2], 100)} tokens")
        seq_len = min(codes_onnx.shape[2], 100)
        codes_onnx = codes_onnx[:, :, :seq_len]
    else:
        seq_len = codes_onnx.shape[2]
    
    # Coarse generation with ONNX
    print("\n  5b. ONNX Coarse Generation...")
    
    # Prepare for transformer input (flatten codebooks)
    batch_size = 1
    n_codebooks = 14
    
    # Take only coarse codebooks
    coarse_codes = codes_onnx[:, :4, :].copy()
    
    # Create mask for ONNX
    mask_onnx = create_matching_mask(mask_ratio, 4, seq_len, periodic_prompt)
    
    # Keep 3D shape for transformer input [batch, n_codebooks, seq_len]
    start_time = time.time()
    coarse_output = coarse_session.run(None, {
        'codes': coarse_codes.astype(np.int64),
        'mask': mask_onnx.astype(bool)
    })[0]
    coarse_time = time.time() - start_time
    
    # Output is already [batch, n_codebooks, seq_len]
    coarse_generated = coarse_output
    
    print(f"    Coarse output shape: {coarse_generated.shape}")
    print(f"    Coarse generation time: {coarse_time:.3f}s")
    
    # C2F generation
    print("\n  5c. ONNX C2F Generation...")
    
    # C2F expects all 14 codebooks - combine coarse with empty fine
    # Initialize with coarse codes
    c2f_input = np.zeros((batch_size, 14, seq_len), dtype=np.int64)
    c2f_input[:, :4, :] = coarse_generated
    
    # Create mask for C2F - mask the fine codebooks
    c2f_mask = np.zeros((batch_size, 14, seq_len), dtype=bool)
    c2f_mask[:, 4:, :] = True  # Mask all fine codebooks
    
    start_time = time.time()
    c2f_output = c2f_session.run(None, {
        'codes': c2f_input,
        'mask': c2f_mask
    })[0]
    c2f_time = time.time() - start_time
    
    # Output is [batch, n_codebooks, seq_len]
    complete_codes = c2f_output
    
    print(f"    C2F output shape: {complete_codes.shape}")
    print(f"    C2F generation time: {c2f_time:.3f}s")
    
    # Fix any mask tokens (1024) before decoding
    complete_codes = np.clip(complete_codes, 0, 1023)
    
    # Decode with ONNX
    print("\n  5d. ONNX Decoding...")
    start_time = time.time()
    audio_onnx = decoder_session.run(None, {'codes': complete_codes.astype(np.int64)})[0]
    decode_time = time.time() - start_time
    
    audio_onnx_np = audio_onnx.squeeze()
    print(f"    Decoded shape: {audio_onnx_np.shape}")
    print(f"    Decoding time: {decode_time:.3f}s")
    
    onnx_total_time = encode_time + coarse_time + c2f_time + decode_time
    
    # 6. Save outputs
    print("\n6. Saving outputs...")
    output_dir = Path("outputs/comparison_fixed")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Ensure arrays are 1D
    if audio_vampnet_np.ndim > 1:
        audio_vampnet_np = audio_vampnet_np.squeeze()
    if audio_onnx_np.ndim > 1:
        audio_onnx_np = audio_onnx_np.squeeze()
    
    # Trim to same length
    min_len = min(len(audio_np), len(audio_vampnet_np), len(audio_onnx_np))
    
    sf.write(output_dir / "01_original.wav", audio_np[:min_len], sr)
    sf.write(output_dir / "02_vampnet.wav", audio_vampnet_np[:min_len], sr)
    sf.write(output_dir / "03_onnx.wav", audio_onnx_np[:min_len], sr)
    
    print(f"  Saved to {output_dir}")
    
    # 7. Compare results
    print("\n7. Comparison Summary:")
    print(f"  VampNet time: {vampnet_time:.2f}s")
    print(f"  ONNX time: {onnx_total_time:.2f}s (encode: {encode_time:.2f}s, coarse: {coarse_time:.2f}s, c2f: {c2f_time:.2f}s, decode: {decode_time:.2f}s)")
    print(f"  Speedup: {vampnet_time / onnx_total_time:.1f}x")
    
    # Token comparison
    print("\n  Token statistics:")
    print(f"    VampNet tokens shape: {z_generated_vampnet.shape}")
    print(f"    ONNX tokens shape: {complete_codes.shape}")
    
    # Plot spectrograms
    print("\n8. Creating spectrograms...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (audio, title) in enumerate([
        (audio_np[:min_len], "Original"),
        (audio_vampnet_np[:min_len], "VampNet"),
        (audio_onnx_np[:min_len], "ONNX")
    ]):
        f, t, Sxx = signal.spectrogram(audio, sr, nperseg=1024)
        axes[idx].pcolormesh(t, f[:1000], 10 * np.log10(Sxx[:1000] + 1e-10))
        axes[idx].set_title(title)
        axes[idx].set_ylabel('Frequency [Hz]')
        axes[idx].set_xlabel('Time [s]')
    
    plt.tight_layout()
    plt.savefig(output_dir / "spectrograms.png", dpi=150)
    plt.close()
    
    print(f"  Saved spectrograms to {output_dir}/spectrograms.png")
    
    print("\n✅ Comparison complete!")
    print(f"\nListen to the outputs:")
    print(f"  - {output_dir}/01_original.wav")
    print(f"  - {output_dir}/02_vampnet.wav") 
    print(f"  - {output_dir}/03_onnx.wav")


if __name__ == "__main__":
    compare_audio_generation()