"""
Complete example of exporting VampNet models to ONNX format.

This example demonstrates:
1. Exporting individual components (encoder, decoder, transformer)
2. Transferring weights from pretrained checkpoints
3. Exporting complete models with weight transfer
4. Running inference with exported models
"""

import torch
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import VampNet ONNX modules
from vampnet_onnx import (
    # Export functions
    export_audio_processor,
    export_codec_encoder,
    export_codec_decoder,
    export_mask_generator,
    export_complete_transformer,
    export_pretrained_encoder,
    export_all_components,
    
    # Models
    CoarseTransformer,
    C2FTransformer,
    
    # Weight transfer
    complete_weight_transfer,
    
    # Embeddings
    extract_and_convert_embeddings,
    
    # Validation
    validate_onnx_model,
    create_onnx_session
)


def example_basic_export():
    """Example: Export basic components without pretrained weights."""
    logger.info("=== Basic Component Export ===")
    
    output_dir = Path("onnx_models/basic")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Export audio processor
    logger.info("Exporting audio processor...")
    export_audio_processor(
        output_path=str(output_dir / "audio_processor.onnx"),
        target_sample_rate=44100,
        target_loudness=-24.0,
        hop_length=768
    )
    
    # 2. Export codec encoder
    logger.info("Exporting codec encoder...")
    export_codec_encoder(
        output_path=str(output_dir / "codec_encoder.onnx"),
        n_codebooks=14,
        vocab_size=1024,
        hop_length=768,
        use_simplified=True  # Use simplified version for testing
    )
    
    # 3. Export codec decoder
    logger.info("Exporting codec decoder...")
    export_codec_decoder(
        output_path=str(output_dir / "codec_decoder.onnx"),
        n_codebooks=14,
        vocab_size=1024,
        hop_length=768,
        use_simplified=True
    )
    
    # 4. Export mask generator
    logger.info("Exporting mask generator...")
    export_mask_generator(
        output_path=str(output_dir / "mask_generator.onnx"),
        n_codebooks=14,
        mask_token=1024,
        use_onnx_compatible=True  # Use ONNX-compatible version
    )
    
    logger.info(f"Basic components exported to {output_dir}")


def example_pretrained_export(checkpoint_path: str, codec_path: str):
    """Example: Export models with pretrained weights."""
    logger.info("=== Pretrained Model Export ===")
    
    output_dir = Path("onnx_models/pretrained")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Export pretrained encoder
    logger.info("Exporting pretrained encoder...")
    export_pretrained_encoder(
        codec_path=codec_path,
        output_path=str(output_dir / "vampnet_encoder_pretrained.onnx"),
        use_prepadded=True  # Use pre-padded for fixed input sizes
    )
    
    # 2. Export coarse transformer with weights
    logger.info("Exporting coarse transformer with weights...")
    export_complete_transformer(
        checkpoint_path=checkpoint_path,
        output_path=str(output_dir / "coarse_transformer_weighted.onnx"),
        model_type="coarse",
        codec_path=codec_path,
        verify_export=True
    )
    
    # 3. Export C2F transformer with weights
    logger.info("Exporting C2F transformer with weights...")
    export_complete_transformer(
        checkpoint_path=checkpoint_path,
        output_path=str(output_dir / "c2f_transformer_weighted.onnx"),
        model_type="c2f",
        codec_path=codec_path,
        verify_export=True
    )
    
    logger.info(f"Pretrained models exported to {output_dir}")


def example_weight_transfer(checkpoint_path: str, codec_path: str):
    """Example: Manual weight transfer process."""
    logger.info("=== Manual Weight Transfer Example ===")
    
    # 1. Create models
    coarse_model = CoarseTransformer(
        vocab_size=1025,
        dim=1280,
        n_heads=20,
        n_layers=48
    )
    
    c2f_model = C2FTransformer(
        vocab_size=1025,
        dim=768,
        n_heads=12,
        n_layers=24
    )
    
    # 2. Transfer weights from checkpoint
    logger.info("Transferring weights from checkpoint...")
    results = complete_weight_transfer(
        checkpoint_path=checkpoint_path,
        coarse_model=coarse_model,
        c2f_model=c2f_model,
        return_embeddings=True
    )
    
    # 3. Extract and load codec embeddings
    logger.info("Extracting codec embeddings...")
    embeddings = extract_and_convert_embeddings(
        codec_path=codec_path,
        model_dim=1280,  # For coarse model
        n_codebooks=14,
        save_path="embeddings/codec_embeddings.pth"
    )
    
    logger.info("Weight transfer complete!")
    logger.info(f"Embeddings shape: {embeddings['combined_embeddings'].shape}")
    
    # 4. Export the weighted models
    output_dir = Path("onnx_models/manual_transfer")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export coarse model
    torch.onnx.export(
        coarse_model,
        torch.randint(0, 1025, (1, 100, 4)),
        str(output_dir / "coarse_manual.onnx"),
        input_names=['tokens'],
        output_names=['logits'],
        dynamic_axes={
            'tokens': {0: 'batch', 1: 'sequence'},
            'logits': {0: 'batch', 1: 'sequence'}
        }
    )
    
    logger.info(f"Manually transferred models exported to {output_dir}")


def example_inference():
    """Example: Run inference with exported models."""
    logger.info("=== ONNX Inference Example ===")
    
    import onnxruntime as ort
    
    # 1. Load audio processor
    audio_session = create_onnx_session("onnx_models/basic/audio_processor.onnx")
    
    # Process audio
    test_audio = np.random.randn(1, 2, 44100).astype(np.float32)  # 1 second stereo
    processed = audio_session.run(None, {'audio': test_audio})[0]
    logger.info(f"Processed audio shape: {processed.shape}")
    
    # 2. Load encoder
    encoder_session = create_onnx_session("onnx_models/basic/codec_encoder.onnx")
    
    # Encode audio
    codes = encoder_session.run(None, {'audio': processed})[0]
    logger.info(f"Encoded codes shape: {codes.shape}")
    
    # 3. Load mask generator
    mask_session = create_onnx_session("onnx_models/basic/mask_generator.onnx")
    
    # Generate mask
    mask_output = mask_session.run(None, {'codes': codes})
    mask, masked_codes = mask_output[0], mask_output[1]
    logger.info(f"Mask shape: {mask.shape}, Masked codes shape: {masked_codes.shape}")
    
    # 4. Load decoder
    decoder_session = create_onnx_session("onnx_models/basic/codec_decoder.onnx")
    
    # Decode back to audio
    reconstructed = decoder_session.run(None, {'codes': codes})[0]
    logger.info(f"Reconstructed audio shape: {reconstructed.shape}")


def example_complete_pipeline():
    """Example: Complete export pipeline with all components."""
    logger.info("=== Complete Pipeline Export ===")
    
    # Export all components at once
    exported_models = export_all_components(
        output_dir="onnx_models/complete",
        checkpoint_path=None,  # Set to actual checkpoint path
        codec_path=None,       # Set to actual codec path
        export_weighted_models=False,  # Set to True with valid paths
        audio_processor={
            'target_sample_rate': 44100,
            'hop_length': 768
        },
        codec_encoder={
            'n_codebooks': 14,
            'vocab_size': 1024,
            'use_simplified': True
        },
        codec_decoder={
            'n_codebooks': 14,
            'vocab_size': 1024,
            'use_simplified': True
        },
        mask_generator={
            'n_codebooks': 14,
            'use_onnx_compatible': True
        },
        transformer={
            'n_codebooks': 4,
            'vocab_size': 1024,
            'd_model': 512,
            'n_heads': 8,
            'n_layers': 6,
            'use_simplified': True
        }
    )
    
    logger.info("Exported models:")
    for name, path in exported_models.items():
        logger.info(f"  {name}: {path}")
        
        # Validate each model
        is_valid, info = validate_onnx_model(path)
        logger.info(f"    Valid: {is_valid}, Info: {info}")


def example_model_optimization():
    """Example: Optimize exported models for deployment."""
    logger.info("=== Model Optimization Example ===")
    
    from onnxruntime.quantization import quantize_dynamic, QuantType
    import onnx
    
    # Load a model
    model_path = "onnx_models/basic/codec_encoder.onnx"
    optimized_path = "onnx_models/optimized/codec_encoder_int8.onnx"
    
    Path("onnx_models/optimized").mkdir(parents=True, exist_ok=True)
    
    # Quantize to INT8
    logger.info(f"Quantizing {model_path}...")
    quantize_dynamic(
        model_path,
        optimized_path,
        weight_type=QuantType.QInt8
    )
    
    # Compare model sizes
    original_size = Path(model_path).stat().st_size / 1024 / 1024  # MB
    optimized_size = Path(optimized_path).stat().st_size / 1024 / 1024  # MB
    
    logger.info(f"Original size: {original_size:.2f} MB")
    logger.info(f"Optimized size: {optimized_size:.2f} MB")
    logger.info(f"Compression ratio: {original_size / optimized_size:.2f}x")
    
    # Validate optimized model
    is_valid, info = validate_onnx_model(optimized_path)
    logger.info(f"Optimized model valid: {is_valid}")


def main():
    """Run all examples."""
    # Note: Set these paths to your actual model files
    checkpoint_path = "path/to/vampnet_checkpoint.pth"
    codec_path = "path/to/codec_model.pth"
    
    # Run basic export example
    example_basic_export()
    
    # Run inference example
    example_inference()
    
    # Run complete pipeline
    example_complete_pipeline()
    
    # Run optimization example
    example_model_optimization()
    
    # If you have actual model files, uncomment these:
    # example_pretrained_export(checkpoint_path, codec_path)
    # example_weight_transfer(checkpoint_path, codec_path)
    
    logger.info("All examples completed!")


if __name__ == "__main__":
    main()