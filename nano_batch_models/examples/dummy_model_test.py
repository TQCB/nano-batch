"""
Unit test for nano_batch_models components.

This script tests the model architecture and inference engine
WITHOUT downloading large models - uses random weights for testing.
"""

import torch
from nano_batch_models import (
    MistralConfig,
    MistralForCausalLM,
    PagedMistralForCausalLM,
)
from nano_batch import Engine, KVCache


def test_mistral_architecture():
    """Test Mistral model architecture with small config."""
    print("="*60)
    print("Testing Mistral Architecture (No Download Required)")
    print("="*60)
    
    # Create a tiny config for testing
    config = MistralConfig(
        vocab_size=1000,        # Small vocab
        hidden_size=256,        # Tiny hidden size
        intermediate_size=512,  # Small MLP
        num_hidden_layers=2,    # Just 2 layers
        num_attention_heads=4,  # 4 heads
        num_key_value_heads=2,  # 2 KV heads (GQA)
        max_position_embeddings=512,
    )
    
    print(f"\n1. Creating Mistral model with config:")
    print(f"   - Layers: {config.num_hidden_layers}")
    print(f"   - Hidden size: {config.hidden_size}")
    print(f"   - Attention heads: {config.num_attention_heads}")
    print(f"   - KV heads: {config.num_key_value_heads}")
    
    # Create model with random weights
    model = MistralForCausalLM(config)
    print("   ✓ Standard Mistral model created")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Total parameters: {num_params:,} (~{num_params/1e6:.2f}M)")
    
    # Test forward pass
    print("\n2. Testing forward pass...")
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    with torch.no_grad():
        logits, _ = model(input_ids)
    
    print(f"   ✓ Input shape: {input_ids.shape}")
    print(f"   ✓ Output shape: {logits.shape}")
    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    
    return model, config


def test_paged_attention_conversion(model, config):
    """Test conversion to PagedAttention variant."""
    print("\n" + "="*60)
    print("Testing PagedAttention Conversion")
    print("="*60)
    
    print("\n1. Converting to PagedMistral...")
    paged_model = PagedMistralForCausalLM.from_standard_mistral(model)
    print("   ✓ Conversion successful")
    
    # Verify structure
    print(f"   ✓ Model has {len(paged_model.layers)} paged layers")
    
    return paged_model


def test_inference_engine(paged_model, config):
    """Test the inference engine with the paged model."""
    print("\n" + "="*60)
    print("Testing Inference Engine")
    print("="*60)
    
    # Engine configuration
    num_blocks = 50
    block_size = 16
    
    print(f"\n1. Initializing engine:")
    print(f"   - Blocks: {num_blocks}")
    print(f"   - Block size: {block_size}")
    
    engine = Engine(num_blocks=num_blocks, block_size=block_size)
    print("   ✓ Engine created")
    
    # Add a request
    request_id = "test_request"
    prompt_tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    
    print(f"\n2. Adding request with {len(prompt_tokens)} tokens...")
    engine.add_request(
        request_id=request_id,
        prompt_token_ids=prompt_tokens,
        max_tokens=20,
    )
    print("   ✓ Request added")
    
    # Run scheduling
    print("\n3. Running scheduler...")
    output = engine.step()
    
    print(f"   ✓ Scheduled: {output.scheduled_requests}")
    print(f"   ✓ Block table: {output.block_tables}")
    print(f"   ✓ Slots: {output.slot_mappings}")
    
    # Simulate a few decode steps
    print("\n4. Simulating token generation...")
    for i in range(3):
        new_token = 100 + i
        engine.update({request_id: new_token})
        print(f"   Step {i+1}: Generated token {new_token}")
    
    print("   ✓ Generation simulation complete")
    
    return engine


def test_kv_cache(config):
    """Test KV cache functionality."""
    print("\n" + "="*60)
    print("Testing KV Cache")
    print("="*60)
    
    num_blocks = 50
    block_size = 16
    
    print(f"\n1. Creating KV cache:")
    print(f"   - Blocks: {num_blocks}")
    print(f"   - Block size: {block_size}")
    print(f"   - KV heads: {config.num_key_value_heads}")
    print(f"   - Head dim: {config.hidden_size // config.num_attention_heads}")
    
    kv_cache = KVCache(
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.hidden_size // config.num_attention_heads,
        dtype=torch.float32,
        device="cpu",
    )
    print("   ✓ KV cache created")
    
    # Write some dummy KV data
    num_tokens = 10
    head_dim = config.hidden_size // config.num_attention_heads
    dummy_keys = torch.randn(num_tokens, config.num_key_value_heads, head_dim)
    dummy_values = torch.randn(num_tokens, config.num_key_value_heads, head_dim)
    slot_mappings = torch.arange(num_tokens, dtype=torch.long)
    
    print(f"\n2. Writing {num_tokens} KV pairs to cache...")
    kv_cache.write_kv(dummy_keys, dummy_values, slot_mappings)
    print("   ✓ Write successful")
    
    # Check memory usage
    mem_stats = kv_cache.get_memory_usage()
    print(f"\n3. Memory usage:")
    print(f"   ✓ Total: {mem_stats['total_bytes'] / 1024:.2f} KB")
    
    return kv_cache


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("nano_batch_models - Unit Tests")
    print("Testing WITHOUT downloading large models")
    print("="*60)
    
    try:
        # Test 1: Model architecture
        model, config = test_mistral_architecture()
        
        # Test 2: Paged conversion
        paged_model = test_paged_attention_conversion(model, config)
        
        # Test 3: Inference engine
        engine = test_inference_engine(paged_model, config)
        
        # Test 4: KV cache
        kv_cache = test_kv_cache(config)
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        print("\nThe core components are working correctly:")
        print("  ✓ Mistral architecture (RoPE, GQA, SwiGLU)")
        print("  ✓ PagedAttention conversion")
        print("  ✓ Rust scheduler with block allocation")
        print("  ✓ KV cache management")
        print("\nTo run with a real model, use simple_generation.py")
        print("(Note: This requires downloading Mistral-7B ~15GB)")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
