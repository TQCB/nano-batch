"""
Minimal demonstration of the nano_batch core engine.

This example shows how to use the Rust scheduler and PagedAttention primitives
with synthetic tensors (no real model required).
"""

import torch
from nano_batch import Engine, paged_attention_fwd, KVCache

def main():
    num_blocks = 100
    block_size = 16
    num_kv_heads = 8
    head_dim = 128
    num_heads = 32  # GQA: num_heads > num_kv_heads
    
    engine = Engine(num_blocks=num_blocks, block_size=block_size)
    kv_cache = KVCache(
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=torch.float32,
        device="cpu",  # Use CPU for compatibility
    )
    
    request_id = "demo_request_1"
    prompt_tokens = [1, 2, 3, 4, 5]
    print(f"\nAdding request '{request_id}' with {len(prompt_tokens)} prompt tokens")
    
    engine.add_request(
        request_id=request_id,
        prompt_token_ids=prompt_tokens,
        max_tokens=10,
    )
    
    # run scheduling step
    output = engine.step()
    
    print(f"Scheduled requests: {output.scheduled_requests}")
    print(f"Block tables: {output.block_tables}")
    print(f"Slot mappings: {output.slot_mappings}")
    print(f"Tokens per request: {output.num_tokens_per_request}")
    
    # simulate writing to KV cache
    if output.scheduled_requests:
        num_tokens = sum(output.num_tokens_per_request.values())
        
        # create dummy KVs
        dummy_keys = torch.randn(num_tokens, num_kv_heads, head_dim)
        dummy_values = torch.randn(num_tokens, num_kv_heads, head_dim)
        slot_mappings_tensor = torch.tensor(output.slot_mappings, dtype=torch.long)
        
        kv_cache.write_kv(dummy_keys, dummy_values, slot_mappings_tensor)
        print(f"Wrote {num_tokens} token KV pairs to cache")
        
        mem_usage = kv_cache.get_memory_usage()
        print(f"KV cache memory: {mem_usage['total_bytes'] / 1024 / 1024:.2f} MB")
    
    # decode phase paged attention
    batch_size = 1
    seq_len = 1  # decode phase
    
    query = torch.randn(batch_size, num_heads, seq_len, head_dim)
    key_cache, value_cache = kv_cache.get_kv_cache()
    
    # create block table and context lengths
    if request_id in output.block_tables:
        block_table_list = output.block_tables[request_id]
        max_blocks = max(len(bt) for bt in output.block_tables.values())
        
        # pad block table to max length
        padded_blocks = block_table_list + [0] * (max_blocks - len(block_table_list))
        block_table = torch.tensor([padded_blocks], dtype=torch.long)
        context_lens = torch.tensor([len(prompt_tokens)], dtype=torch.long)
        
        attn_output = paged_attention_fwd(
            query,
            key_cache,
            value_cache,
            block_table,
            context_lens,
            block_size,
        )
    
    # token generation
    for step in range(5):
        # update with generated token
        new_token = 100 + step
        engine.update({request_id: new_token})
        print(f"Step {step + 1}: Generated token {new_token}")
        
        # schedule next step
        output = engine.step()
        if output.scheduled_requests:
            print(f"  Block table size: {len(output.block_tables[request_id])} blocks")
            
    print([k for k in dir(output) if not k.startswith('_')])
    print(f"\nOutput:")
    [print(f"{k}: {getattr(output, k)}")
    for k in dir(output)
    if not k.startswith('_')]

if __name__ == "__main__":
    main()