"""
Simple end-to-end example using the nano_batch inference engine.

This demonstrates how to use the complete pipeline with PagedAttention and
continuous batching.
"""

import torch
from nano_batch_models import (
    MistralForCausalLM,
    PagedMistralForCausalLM,
    MistralTokenizer,
    InferenceEngine,
    GenerationConfig,
)


def main():
    """Run a simple generation example."""
    # Configuration
    model_name = "mistralai/Mistral-7B-v0.1"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Memory configuration
    num_blocks = 100  # Total KV cache blocks
    block_size = 16   # Tokens per block
    
    print("="*60)
    print("nano_batch Inference Engine - Simple Example")
    print("="*60)
    
    # Load model
    print(f"\n1. Loading Mistral model: {model_name}")
    standard_model = MistralForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16
    )
    
    print("2. Converting to PagedMistral...")
    paged_model = PagedMistralForCausalLM.from_standard_mistral(standard_model)
    del standard_model
    
    # Load tokenizer
    print("3. Loading tokenizer...")
    tokenizer = MistralTokenizer(model_name)
    
    # Create inference engine
    print(f"4. Initializing inference engine (blocks={num_blocks}, block_size={block_size})...")
    engine = InferenceEngine(
        model=paged_model,
        tokenizer=tokenizer,
        num_blocks=num_blocks,
        block_size=block_size,
        device=device,
    )
    
    # Generation config
    gen_config = GenerationConfig(
        max_tokens=50,
        temperature=0.8,
        top_p=0.9,
        top_k=50,
    )
    
    # Example prompts
    prompts = [
        "The future of artificial intelligence is",
        "Once upon a time in a distant galaxy,",
        "Explain quantum computing in simple terms:",
    ]
    
    print("\n" + "="*60)
    print("Generating Responses")
    print("="*60)
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n[Prompt {i}]: {prompt}")
        
        # Generate
        output = engine.generate(prompt, gen_config)
        
        print(f"[Output {i}]: {output}")
        print("-"*60)
    
    print("\n✓ Generation complete!")
    
    # Memory stats
    print("\n" + "="*60)
    print("Memory Statistics")
    print("="*60)
    for i, cache in enumerate(engine.kv_caches):
        stats = cache.get_memory_usage()
        print(f"Layer {i}: {stats['total_gb']:.2f} GB")
    
    total_memory = sum(cache.get_memory_usage()['total_gb'] for cache in engine.kv_caches)
    print(f"Total KV Cache: {total_memory:.2f} GB")
    print("="*60)


if __name__ == "__main__":
    main()
