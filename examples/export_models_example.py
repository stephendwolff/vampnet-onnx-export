"""
Example script showing how to export VampNet models to ONNX using the integrated API.
"""

import torch
from vampnet_onnx import (
    export_complete_transformer,
    export_pretrained_encoder,
    export_codec_decoder,
    complete_weight_transfer,
    CoarseTransformer,
    C2FTransformer
)


def export_models_from_checkpoint():
    """Example: Export models from a VampNet checkpoint."""
    
    # Paths
    checkpoint_path = "path/to/vampnet/checkpoint.pth"
    output_dir = "exported_models/"
    
    # 1. Export transformer models with weights
    print("Exporting Coarse Transformer...")
    coarse_result = export_complete_transformer(
        checkpoint_path=checkpoint_path,
        output_path=f"{output_dir}/coarse_transformer.onnx",
        model_type="coarse",
        verify_export=True
    )
    print(f"Coarse transformer exported: {coarse_result['output_path']}")
    
    print("\nExporting C2F Transformer...")
    c2f_result = export_complete_transformer(
        checkpoint_path=checkpoint_path,
        output_path=f"{output_dir}/c2f_transformer.onnx",
        model_type="c2f",
        verify_export=True
    )
    print(f"C2F transformer exported: {c2f_result['output_path']}")
    
    # 2. Export codec (if available)
    print("\nExporting Codec Encoder...")
    try:
        encoder_path = export_pretrained_encoder(
            codec_path="path/to/codec.pth",
            output_path=f"{output_dir}/codec_encoder.onnx",
            use_prepadded=True
        )
        print(f"Encoder exported: {encoder_path}")
    except Exception as e:
        print(f"Could not export encoder: {e}")
        print("Using simplified encoder instead...")
        export_codec_encoder(
            output_path=f"{output_dir}/codec_encoder_simple.onnx",
            use_simplified=True
        )
    
    # 3. Export decoder
    print("\nExporting Codec Decoder...")
    export_codec_decoder(
        output_path=f"{output_dir}/codec_decoder.onnx",
        use_simplified=True
    )
    print(f"Decoder exported: {output_dir}/codec_decoder.onnx")


def manual_weight_transfer_example():
    """Example: Manually transfer weights to custom models."""
    
    checkpoint_path = "path/to/vampnet/checkpoint.pth"
    
    # Create models
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
    
    # Transfer weights
    print("Transferring weights...")
    results = complete_weight_transfer(
        checkpoint_path=checkpoint_path,
        coarse_model=coarse_model,
        c2f_model=c2f_model,
        return_embeddings=True
    )
    
    print("Weight transfer complete!")
    print(f"Embeddings shape: {results['embeddings']['codebooks'].shape}")
    
    # Export models with transferred weights
    dummy_input = torch.randint(0, 1025, (1, 100, 4))
    
    torch.onnx.export(
        coarse_model,
        dummy_input,
        "exported_models/coarse_with_weights.onnx",
        input_names=['tokens'],
        output_names=['logits'],
        dynamic_axes={
            'tokens': {0: 'batch', 1: 'sequence'},
            'logits': {0: 'batch', 1: 'sequence'}
        },
        opset_version=14
    )
    print("Exported coarse model with weights!")


def extract_embeddings_example():
    """Example: Extract and save codec embeddings."""
    
    from vampnet_onnx import extract_and_convert_embeddings
    
    # Extract embeddings from codec
    embeddings = extract_and_convert_embeddings(
        codec_path="path/to/codec.pth",
        model_dim=1280,
        n_codebooks=14,
        save_path="exported_models/embeddings.pth"
    )
    
    print(f"Extracted embeddings:")
    print(f"  Codebooks: {embeddings['codebooks'].shape}")
    print(f"  Projection: {embeddings['projection'].shape}")
    print(f"  Mask tokens: {embeddings['mask_tokens'].shape}")
    print(f"  Combined: {embeddings['combined_embeddings'].shape}")


if __name__ == "__main__":
    print("VampNet ONNX Export Examples")
    print("=" * 50)
    
    # Uncomment to run examples:
    
    # export_models_from_checkpoint()
    # manual_weight_transfer_example()
    # extract_embeddings_example()
    
    print("\nFor more examples, see the notebooks/ directory.")