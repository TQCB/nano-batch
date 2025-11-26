"""
Example: Load and test Mistral 7B model.

This script demonstrates how to load a Mistral model and run basic inference
without the continuous batching engine (for testing purposes).
"""

import torch
from nano_batch_models import MistralForCausalLM, MistralConfig
from nano_batch_models.utils import load_mistral_model, get_model_memory_footprint, generate_tokens


def main():
    """Load Mistral and run a simple generation test."""
    
    # Configuration
    model_name = "mistralai/Mistral-7B-v0.1"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    print(f"Loading Mistral model from: {model_name}")
    print(f"Device: {device}, Dtype: {dtype}")
    
    # Load model
    model = load_mistral_model(
        model_name,
        device=device,
        dtype=dtype,
    )
    
    # Print memory footprint
    memory_stats = get_model_memory_footprint(model)
    print(f"\nModel Memory Footprint:")
    print(f"  Total Parameters: {memory_stats['total_params']:,}")
    print(f"  Size: {memory_stats['size_gb']:.2f} GB")
    
    # Example prompt
    # Note: For actual tokenization, you should use the Mistral tokenizer
    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # input_ids = tokenizer.encode("Hello, my name is", return_tensors="pt")
    
    # For this example, we'll use dummy token IDs
    print("\n" + "="*50)
    print("NOTE: This is a basic example.")
    print("For production use, integrate with the continuous batching engine")
    print("and implement PagedAttention kernels.")
    print("="*50)
    
    # Dummy input (replace with actual tokenization)
    input_ids = torch.randint(0, model.config.vocab_size, (1, 10), device=device)
    
    print(f"\nRunning forward pass with input shape: {input_ids.shape}")
    
    # Forward pass
    with torch.no_grad():
        logits, _ = model(input_ids)
    
    print(f"Output logits shape: {logits.shape}")
    print(f"✓ Model loaded and tested successfully!")
    
    # Optional: Run generation (will be slow without optimizations)
    # output_ids = generate_tokens(model, input_ids, max_new_tokens=20)
    # print(f"Generated sequence length: {output_ids.shape[1]}")


if __name__ == "__main__":
    main()
