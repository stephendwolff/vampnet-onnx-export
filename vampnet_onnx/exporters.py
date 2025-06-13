"""
ONNX export functions for VampNet components.

This module provides comprehensive export functionality for all VampNet components,
including weight transfer, codec export, and transformer export with full support
for pretrained models.
"""

import torch
import torch.onnx
import onnx
import onnxruntime as ort
import os
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from .audio_processor import AudioProcessor, AudioPostProcessor
from .codec_wrapper import CodecEncoder, CodecDecoder, SimplifiedCodec, SimplifiedCodecEncoder, SimplifiedCodecDecoder
from .mask_generator import MaskGenerator, AdvancedMaskGenerator
from .mask_generator_onnx import ONNXMaskGenerator, FlexibleONNXMaskGenerator
from .transformer_wrapper import TransformerWrapper, SimplifiedVampNetModel
from .models import CoarseTransformer, C2FTransformer
from .weight_transfer import complete_weight_transfer
from .embeddings import extract_and_convert_embeddings, load_embeddings_into_model

# Try to import VampNet codec wrappers
try:
    from .vampnet_codec import VampNetCodecEncoder, VampNetCodecDecoder, VAMPNET_AVAILABLE
except ImportError:
    VAMPNET_AVAILABLE = False

logger = logging.getLogger(__name__)


def export_audio_processor(
    output_path: str,
    target_sample_rate: int = 44100,
    target_loudness: float = -24.0,
    hop_length: int = 768,
    example_batch_size: int = 1,
    example_audio_length: int = 44100,
    opset_version: int = 14
) -> None:
    """
    Export audio processor to ONNX.
    
    Args:
        output_path: Path to save ONNX model
        target_sample_rate: Target sample rate
        target_loudness: Target loudness in dB
        hop_length: Hop length for padding
        example_batch_size: Batch size for example input
        example_audio_length: Audio length for example input
        opset_version: ONNX opset version
    """
    # Create model
    model = AudioProcessor(
        target_sample_rate=target_sample_rate,
        target_loudness=target_loudness,
        hop_length=hop_length
    )
    model.eval()
    
    # Example input
    example_input = torch.randn(example_batch_size, 2, example_audio_length)
    
    # Export
    torch.onnx.export(
        model,
        example_input,
        output_path,
        input_names=['audio'],
        output_names=['processed_audio'],
        dynamic_axes={
            'audio': {0: 'batch', 2: 'samples'},
            'processed_audio': {0: 'batch', 2: 'samples'}
        },
        opset_version=opset_version,
        export_params=True,
        do_constant_folding=True
    )
    
    print(f"Exported audio processor to {output_path}")
    
    # Verify
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("Model verification passed!")


def export_codec_encoder(
    output_path: str,
    n_codebooks: int = 14,
    vocab_size: int = 1024,
    sample_rate: int = 44100,
    hop_length: int = 768,
    example_batch_size: int = 1,
    example_audio_length: int = 44100,
    use_simplified: bool = True,
    use_vampnet: bool = False,
    codec_model=None,
    device: str = 'cpu',
    opset_version: int = 14
) -> None:
    """
    Export codec encoder to ONNX.
    
    Args:
        output_path: Path to save ONNX model
        n_codebooks: Number of codebooks
        vocab_size: Vocabulary size
        sample_rate: Sample rate
        hop_length: Hop length
        example_batch_size: Batch size for example
        example_audio_length: Audio length for example
        use_simplified: Use simplified codec for testing
        use_vampnet: Use actual VampNet codec
        codec_model: Pre-loaded VampNet codec model (optional)
        device: Device to use for VampNet codec
        opset_version: ONNX opset version
    """
    if use_vampnet and VAMPNET_AVAILABLE:
        # Use actual VampNet codec
        print("Using VampNet codec encoder...")
        model = VampNetCodecEncoder(codec_model=codec_model, device=device)
        model.eval()
        example_input = torch.randn(example_batch_size, 1, example_audio_length)
        
        torch.onnx.export(
            model,
            example_input,
            output_path,
            input_names=['audio'],
            output_names=['codes'],
            dynamic_axes={
                'audio': {0: 'batch', 2: 'samples'},
                'codes': {0: 'batch', 2: 'sequence'}
            },
            opset_version=opset_version
        )
    elif use_simplified:
        codec = SimplifiedCodec(
            n_codebooks=n_codebooks,
            vocab_size=vocab_size,
            hop_length=hop_length
        )
        # Use the wrapper for ONNX export
        model = SimplifiedCodecEncoder(codec)
        model.eval()
        example_input = torch.randn(example_batch_size, 1, example_audio_length)
        
        torch.onnx.export(
            model,
            example_input,
            output_path,
            input_names=['audio'],
            output_names=['codes'],
            dynamic_axes={
                'audio': {0: 'batch', 2: 'samples'},
                'codes': {0: 'batch', 2: 'sequence'}
            },
            opset_version=opset_version
        )
    else:
        model = CodecEncoder(
            n_codebooks=n_codebooks,
            vocab_size=vocab_size,
            sample_rate=sample_rate,
            hop_length=hop_length
        )
        model.eval()
        
        example_input = torch.randn(example_batch_size, 1, example_audio_length)
        
        torch.onnx.export(
            model,
            example_input,
            output_path,
            input_names=['audio'],
            output_names=['codes'],
            dynamic_axes={
                'audio': {0: 'batch', 2: 'samples'},
                'codes': {0: 'batch', 2: 'sequence'}
            },
            opset_version=opset_version
        )
    
    print(f"Exported codec encoder to {output_path}")
    
    # Verify and infer shapes
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    # Infer shapes to ensure all tensors have type information
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    onnx.save(onnx_model, output_path)
    
    print("Model verification passed!")


def export_codec_decoder(
    output_path: str,
    n_codebooks: int = 14,
    vocab_size: int = 1024,
    sample_rate: int = 44100,
    hop_length: int = 768,
    example_batch_size: int = 1,
    example_sequence_length: int = 100,
    use_simplified: bool = True,
    use_vampnet: bool = False,
    codec_model=None,
    device: str = 'cpu',
    opset_version: int = 14
) -> None:
    """
    Export codec decoder to ONNX.
    """
    if use_vampnet and VAMPNET_AVAILABLE:
        # Use actual VampNet codec
        print("Using VampNet codec decoder...")
        model = VampNetCodecDecoder(codec_model=codec_model, device=device)
        model.eval()
        example_input = torch.randint(
            0, vocab_size,
            (example_batch_size, n_codebooks, example_sequence_length)
        )
        
        torch.onnx.export(
            model,
            example_input,
            output_path,
            input_names=['codes'],
            output_names=['audio'],
            dynamic_axes={
                'codes': {0: 'batch', 2: 'sequence'},
                'audio': {0: 'batch', 2: 'samples'}
            },
            opset_version=opset_version
        )
    elif use_simplified:
        codec = SimplifiedCodec(
            n_codebooks=n_codebooks,
            vocab_size=vocab_size,
            hop_length=hop_length
        )
        # Use the wrapper for ONNX export
        model = SimplifiedCodecDecoder(codec)
        model.eval()
        example_input = torch.randint(
            0, vocab_size,
            (example_batch_size, n_codebooks, example_sequence_length)
        )
        
        torch.onnx.export(
            model,
            example_input,
            output_path,
            input_names=['codes'],
            output_names=['audio'],
            dynamic_axes={
                'codes': {0: 'batch', 2: 'sequence'},
                'audio': {0: 'batch', 2: 'samples'}
            },
            opset_version=opset_version
        )
    else:
        model = CodecDecoder(
            n_codebooks=n_codebooks,
            vocab_size=vocab_size,
            sample_rate=sample_rate,
            hop_length=hop_length
        )
        model.eval()
        
        example_input = torch.randint(
            0, vocab_size,
            (example_batch_size, n_codebooks, example_sequence_length)
        )
        
        torch.onnx.export(
            model,
            example_input,
            output_path,
            input_names=['codes'],
            output_names=['audio'],
            dynamic_axes={
                'codes': {0: 'batch', 2: 'sequence'},
                'audio': {0: 'batch', 2: 'samples'}
            },
            opset_version=opset_version
        )
    
    print(f"Exported codec decoder to {output_path}")
    
    # Verify and infer shapes
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    # Infer shapes to ensure all tensors have type information
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    onnx.save(onnx_model, output_path)
    
    print("Model verification passed!")


def export_mask_generator(
    output_path: str,
    n_codebooks: int = 14,
    mask_token: int = 1024,
    example_batch_size: int = 1,
    example_sequence_length: int = 100,
    use_advanced: bool = False,
    use_onnx_compatible: bool = True,
    opset_version: int = 14
) -> None:
    """
    Export mask generator to ONNX.
    """
    if use_onnx_compatible:
        # Use ONNX-compatible version that avoids conditionals
        model = ONNXMaskGenerator(
            n_codebooks=n_codebooks,
            mask_token=mask_token,
            periodic_prompt=7,  # Fixed for export
            upper_codebook_mask=3  # Fixed for export
        )
    elif use_advanced:
        model = AdvancedMaskGenerator(
            n_codebooks=n_codebooks,
            mask_token=mask_token
        )
    else:
        model = MaskGenerator(
            n_codebooks=n_codebooks,
            mask_token=mask_token
        )
    model.eval()
    
    # Example inputs
    example_codes = torch.randint(
        0, 1024,
        (example_batch_size, n_codebooks, example_sequence_length)
    )
    
    # Export based on model type
    if use_onnx_compatible:
        # ONNXMaskGenerator only takes codes as input
        torch.onnx.export(
            model,
            example_codes,
            output_path,
            input_names=['codes'],
            output_names=['mask', 'masked_codes'],
            dynamic_axes={
                'codes': {0: 'batch', 2: 'sequence'},
                'mask': {0: 'batch', 2: 'sequence'},
                'masked_codes': {0: 'batch', 2: 'sequence'}
            },
            opset_version=opset_version
        )
    elif not use_advanced:
        # Regular mask generator needs wrapper
        class MaskGeneratorWrapper(torch.nn.Module):
            def __init__(self, mask_gen):
                super().__init__()
                self.mask_gen = mask_gen
                
            def forward(self, codes):
                # Use fixed values for ONNX export
                return self.mask_gen(codes, 7, 3, 0)
        
        wrapper = MaskGeneratorWrapper(model)
        wrapper.eval()
        
        torch.onnx.export(
            wrapper,
            example_codes,
            output_path,
            input_names=['codes'],
            output_names=['mask', 'masked_codes'],
            dynamic_axes={
                'codes': {0: 'batch', 2: 'sequence'},
                'mask': {0: 'batch', 2: 'sequence'},
                'masked_codes': {0: 'batch', 2: 'sequence'}
            },
            opset_version=opset_version
        )
    else:
        # For advanced generator, export periodic mask as example
        class AdvancedMaskWrapper(torch.nn.Module):
            def __init__(self, mask_gen):
                super().__init__()
                self.mask_gen = mask_gen
                
            def forward(self, codes):
                return self.mask_gen.periodic_mask(codes, period=7, offset=0, upper_codebook_mask=3)
        
        wrapper = AdvancedMaskWrapper(model)
        wrapper.eval()
        
        torch.onnx.export(
            wrapper,
            example_codes,
            output_path,
            input_names=['codes'],
            output_names=['mask', 'masked_codes'],
            dynamic_axes={
                'codes': {0: 'batch', 2: 'sequence'},
                'mask': {0: 'batch', 2: 'sequence'},
                'masked_codes': {0: 'batch', 2: 'sequence'}
            },
            opset_version=opset_version
        )
    
    print(f"Exported mask generator to {output_path}")
    
    # Verify
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("Model verification passed!")


def export_transformer(
    output_path: str,
    n_codebooks: int = 4,
    vocab_size: int = 1024,
    d_model: int = 512,
    n_heads: int = 8,
    n_layers: int = 6,
    mask_token: int = 1024,
    example_batch_size: int = 1,
    example_sequence_length: int = 100,
    use_simplified: bool = True,
    opset_version: int = 14
) -> None:
    """
    Export transformer to ONNX.
    """
    if use_simplified:
        model = SimplifiedVampNetModel(
            n_codebooks=n_codebooks,
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            mask_token=mask_token
        )
    else:
        model = TransformerWrapper(
            n_codebooks=n_codebooks,
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            mask_token=mask_token
        )
    model.eval()
    
    # Example inputs
    example_codes = torch.randint(
        0, vocab_size,
        (example_batch_size, n_codebooks, example_sequence_length)
    )
    example_mask = torch.randint(
        0, 2,
        (example_batch_size, n_codebooks, example_sequence_length)
    )
    
    if use_simplified:
        # Simplified model takes mask and temperature
        class SimplifiedWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                
            def forward(self, codes, mask):
                return self.model(codes, mask, temperature=1.0)
        
        wrapper = SimplifiedWrapper(model)
        wrapper.eval()
        
        torch.onnx.export(
            wrapper,
            (example_codes, example_mask),
            output_path,
            input_names=['codes', 'mask'],
            output_names=['generated_codes'],
            dynamic_axes={
                'codes': {0: 'batch', 1: 'n_codebooks', 2: 'sequence'},
                'mask': {0: 'batch', 1: 'n_codebooks', 2: 'sequence'},
                'generated_codes': {0: 'batch', 1: 'n_codebooks', 2: 'sequence'}
            },
            opset_version=opset_version
        )
    else:
        # Regular transformer just outputs logits
        torch.onnx.export(
            model,
            example_codes,
            output_path,
            input_names=['codes'],
            output_names=['logits'],
            dynamic_axes={
                'codes': {0: 'batch', 2: 'sequence'},
                'logits': {0: 'batch', 2: 'sequence'}
            },
            opset_version=opset_version
        )
    
    print(f"Exported transformer to {output_path}")
    
    # Verify and infer shapes
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    # Infer shapes to ensure all tensors have type information
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    onnx.save(onnx_model, output_path)
    
    print("Model verification passed!")


def export_complete_transformer(
    checkpoint_path: str,
    output_path: str,
    model_type: str = "coarse",
    codec_path: Optional[str] = None,
    opset_version: int = 14,
    verify_export: bool = True
) -> Dict[str, Any]:
    """
    Export a complete VampNet transformer with weight transfer from checkpoint.
    
    Args:
        checkpoint_path: Path to VampNet checkpoint file
        output_path: Path to save ONNX model
        model_type: "coarse" or "c2f" 
        codec_path: Path to codec model for embedding extraction
        opset_version: ONNX opset version
        verify_export: Whether to verify the exported model
        
    Returns:
        Dictionary with export results and statistics
    """
    logger.info(f"Exporting {model_type} transformer from {checkpoint_path}")
    
    # Create model based on type
    if model_type == "coarse":
        model = CoarseTransformer()
    elif model_type == "c2f":
        model = C2FTransformer()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    # Transfer weights from checkpoint
    results = complete_weight_transfer(
        checkpoint_path=checkpoint_path,
        coarse_model=model if model_type == "coarse" else None,
        c2f_model=model if model_type == "c2f" else None,
        return_embeddings=True
    )
    
    # Load embeddings if codec path provided
    if codec_path and 'embeddings' in results:
        embeddings = extract_and_convert_embeddings(
            codec_path=codec_path,
            model_dim=model.dim,
            n_codebooks=model.n_codebooks
        )
        load_embeddings_into_model(model, embeddings)
        
    # Get the weighted model
    weighted_model = results.get(f'{model_type}_model', model)
    weighted_model.eval()
    
    # Example input
    batch_size = 1
    seq_len = 100
    example_tokens = torch.randint(
        0, model.vocab_size,
        (batch_size, seq_len, model.n_codebooks),
        dtype=torch.long
    )
    
    # Export to ONNX
    torch.onnx.export(
        weighted_model,
        example_tokens,
        output_path,
        input_names=['tokens'],
        output_names=['logits'],
        dynamic_axes={
            'tokens': {0: 'batch', 1: 'sequence'},
            'logits': {0: 'batch', 1: 'sequence'}
        },
        opset_version=opset_version,
        export_params=True,
        do_constant_folding=True
    )
    
    logger.info(f"Exported {model_type} transformer to {output_path}")
    
    # Verify if requested
    if verify_export:
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        # Test with ONNX Runtime
        session = ort.InferenceSession(output_path)
        ort_outputs = session.run(None, {'tokens': example_tokens.numpy()})
        
        logger.info(f"ONNX Runtime verification passed. Output shape: {ort_outputs[0].shape}")
        
    return {
        'model_path': output_path,
        'model_type': model_type,
        'embeddings': results.get('embeddings'),
        'verified': verify_export
    }


def export_pretrained_encoder(
    codec_path: str,
    output_path: str,
    use_prepadded: bool = True,
    opset_version: int = 14
) -> None:
    """
    Export a pretrained VampNet encoder with proper weight loading.
    
    Args:
        codec_path: Path to VampNet codec checkpoint
        output_path: Path to save ONNX model
        use_prepadded: Whether to use pre-padded encoder for fixed sizes
        opset_version: ONNX opset version
    """
    if not VAMPNET_AVAILABLE:
        raise ImportError("VampNet codec not available. Please install vampnet.")
        
    # Load codec
    from lac import LAC
    codec = LAC.load(codec_path)
    
    if use_prepadded:
        # Create pre-padded encoder wrapper
        class PrePaddedEncoder(torch.nn.Module):
            def __init__(self, codec, target_length: int = 76800):
                super().__init__()
                self.codec = codec
                self.target_length = target_length
                
            def forward(self, audio: torch.Tensor) -> torch.Tensor:
                # Pad to target length
                batch_size, channels, length = audio.shape
                if length < self.target_length:
                    padding = self.target_length - length
                    audio = torch.nn.functional.pad(audio, (0, padding))
                    
                # Encode
                with torch.no_grad():
                    _, codes, _, _, _ = self.codec.encode(audio)
                    
                return codes.permute(0, 2, 1)  # [batch, n_codebooks, seq_len]
                
        model = PrePaddedEncoder(codec)
    else:
        model = VampNetCodecEncoder(codec_model=codec)
        
    model.eval()
    
    # Example input
    example_audio = torch.randn(1, 1, 76800)
    
    # Export
    torch.onnx.export(
        model,
        example_audio,
        output_path,
        input_names=['audio'],
        output_names=['codes'],
        dynamic_axes={
            'audio': {0: 'batch'},
            'codes': {0: 'batch'}
        },
        opset_version=opset_version
    )
    
    logger.info(f"Exported encoder to {output_path}")
    
    # Verify
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    logger.info("Encoder verification passed!")


def export_all_components(
    output_dir: str = "onnx_models",
    checkpoint_path: Optional[str] = None,
    codec_path: Optional[str] = None,
    export_weighted_models: bool = True,
    **kwargs
) -> Dict[str, str]:
    """
    Export all VampNet components to ONNX.
    
    Args:
        output_dir: Directory to save ONNX models
        checkpoint_path: Path to VampNet checkpoint for weight transfer
        codec_path: Path to codec model
        export_weighted_models: Whether to export models with transferred weights
        **kwargs: Additional arguments for exporters
        
    Returns:
        Dictionary of component names to file paths
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    exported_models = {}
    
    # Export audio processor
    audio_processor_path = os.path.join(output_dir, "audio_processor.onnx")
    export_audio_processor(audio_processor_path, **kwargs.get('audio_processor', {}))
    exported_models['audio_processor'] = audio_processor_path
    
    # Export codec encoder
    if codec_path and export_weighted_models:
        # Export pretrained encoder
        encoder_path = os.path.join(output_dir, "vampnet_encoder_pretrained.onnx")
        export_pretrained_encoder(codec_path, encoder_path)
        exported_models['codec_encoder'] = encoder_path
    else:
        codec_encoder_path = os.path.join(output_dir, "codec_encoder.onnx")
        export_codec_encoder(codec_encoder_path, **kwargs.get('codec_encoder', {}))
        exported_models['codec_encoder'] = codec_encoder_path
    
    # Export codec decoder
    codec_decoder_path = os.path.join(output_dir, "codec_decoder.onnx")
    export_codec_decoder(codec_decoder_path, **kwargs.get('codec_decoder', {}))
    exported_models['codec_decoder'] = codec_decoder_path
    
    # Export mask generator
    mask_generator_path = os.path.join(output_dir, "mask_generator.onnx")
    export_mask_generator(mask_generator_path, **kwargs.get('mask_generator', {}))
    exported_models['mask_generator'] = mask_generator_path
    
    # Export transformers with weights if checkpoint provided
    if checkpoint_path and export_weighted_models:
        # Export coarse transformer
        coarse_path = os.path.join(output_dir, "coarse_transformer_weighted.onnx")
        export_complete_transformer(
            checkpoint_path=checkpoint_path,
            output_path=coarse_path,
            model_type="coarse",
            codec_path=codec_path
        )
        exported_models['coarse_transformer'] = coarse_path
        
        # Export C2F transformer
        c2f_path = os.path.join(output_dir, "c2f_transformer_weighted.onnx")
        export_complete_transformer(
            checkpoint_path=checkpoint_path,
            output_path=c2f_path,
            model_type="c2f",
            codec_path=codec_path
        )
        exported_models['c2f_transformer'] = c2f_path
    else:
        # Export regular transformer
        transformer_path = os.path.join(output_dir, "transformer.onnx")
        export_transformer(transformer_path, **kwargs.get('transformer', {}))
        exported_models['transformer'] = transformer_path
    
    print(f"\nAll models exported to {output_dir}")
    return exported_models