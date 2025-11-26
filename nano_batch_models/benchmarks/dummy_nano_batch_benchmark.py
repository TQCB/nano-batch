"""
nano_batch engine benchmark with PagedAttention using dummy model.

This script benchmarks the custom nano_batch inference engine with Triton
PagedAttention kernels and continuous batching, using a randomly initialized
model to avoid downloading pretrained weights.
"""

import time
import torch
import json
from typing import List
from dataclasses import asdict

import sys
import os

sys.path.insert(0, "./")

from nano_batch_models.models.config import MistralConfig
from nano_batch_models.models.mistral import MistralForCausalLM
from nano_batch_models.tokenizer import MistralTokenizer
from nano_batch_models.inference_engine import PagedMistralForCausalLM, InferenceEngine, GenerationConfig
from dummy_baseline_hf import BenchmarkResult, load_test_prompts


def run_nano_batch_benchmark(
    prompts: List[str],
    max_new_tokens: int = 100,
    num_blocks: int = 100,
    block_size: int = 16,
    device: str = "cuda",
    hidden_size: int = 512,
    num_hidden_layers: int = 8,
) -> BenchmarkResult:
    """
    Run benchmark with nano_batch engine using a dummy model.
    
    Args:
        prompts: List of input prompts
        max_new_tokens: Tokens to generate per prompt
        num_blocks: Number of KV cache blocks
        block_size: Tokens per block
        device: Device to run on
        hidden_size: Model hidden size (default matches Mistral-7B)
        num_hidden_layers: Number of transformer layers
        
    Returns:
        BenchmarkResult with performance metrics
    """
    print("Creating dummy Mistral model with random weights...")
    
    # Create lightweight model config for fast testing
    config = MistralConfig(
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 4,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=2048,
        vocab_size=32000,
    )
    
    # Initialize model with random weights
    print("Creating standard Mistral model...")
    standard_model = MistralForCausalLM(config)
    standard_model = standard_model.to(torch.bfloat16).to(device)
    
    print("Converting to PagedMistral...")
    paged_model = PagedMistralForCausalLM.from_standard_mistral(standard_model)
    del standard_model  # Free memory
    
    # Load tokenizer (we still need a real tokenizer)
    print("Loading tokenizer...")
    tokenizer = MistralTokenizer("mistralai/Mistral-7B-v0.1")
    
    # Create inference engine
    print(f"Initializing inference engine (blocks={num_blocks}, block_size={block_size})...")
    engine = InferenceEngine(
        model=paged_model,
        tokenizer=tokenizer,
        num_blocks=num_blocks,
        block_size=block_size,
        device=device,
    )
    
    # Warm up - COMMENTED OUT: PagedAttention only supports decode phase (seq_len=1)
    # The warmup would fail because "Hello" has multiple tokens (prefill phase)
    # print("Warming up...")
    # gen_config = GenerationConfig(max_tokens=10)
    # _ = engine.generate("Hello", gen_config)    

    # if device == "cuda":
    #     torch.cuda.synchronize()
    #     torch.cuda.reset_peak_memory_stats()
    # elif device == "cpu":
    #     torch.cpu.synchronize()
    
    # Benchmark
    print(f"\nRunning benchmark with {len(prompts)} prompts...")
    total_tokens = 0
    first_token_times = []
    
    gen_config = GenerationConfig(
        max_tokens=max_new_tokens,
        temperature=1.0,
        top_p=1.0,
    )
    
    start_time = time.time()
    
    # Submit all requests
    print(f"Submitting {len(prompts)} requests...")
    request_ids = []
    request_start_times = {}
    
    for i, prompt in enumerate(prompts):
        req_id = engine.add_request(prompt, gen_config)
        request_ids.append(req_id)
        request_start_times[req_id] = time.time()
    
    # Run continuous batching loop
    print("Running continuous batching loop...")
    active_requests = set(request_ids)
    completed_requests = set()
    
    # Track metrics
    tokens_generated_per_request = {req_id: 0 for req_id in request_ids}
    first_token_latencies = {}
    
    while active_requests:
        step_results = engine.step()
        current_time = time.time()
        
        # Update metrics
        for req_id, token_text in step_results.items():
            tokens_generated_per_request[req_id] += 1
            
            # Record first token latency
            if tokens_generated_per_request[req_id] == 1:
                latency_ms = (current_time - request_start_times[req_id]) * 1000
                first_token_latencies[req_id] = latency_ms
            
            # Check for completion
            if tokens_generated_per_request[req_id] >= max_new_tokens:
                if req_id in active_requests:
                    active_requests.remove(req_id)
                    completed_requests.add(req_id)
                    
                    if len(completed_requests) % 10 == 0:
                        print(f"  Completed {len(completed_requests)}/{len(prompts)} requests")
    
    # Calculate total tokens
    total_tokens = sum(tokens_generated_per_request.values())
    
    # Collect first token times in order
    first_token_times = [first_token_latencies.get(req_id, 0.0) for req_id in request_ids]
    
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "cpu":
        torch.cpu.synchronize()
        
    end_time = time.time()
    
    # Calculate metrics
    total_time = end_time - start_time
    tokens_per_second = total_tokens / total_time
    avg_latency_per_token = (total_time / total_tokens) * 1000  # ms
    avg_first_token_latency = sum(first_token_times) / len(first_token_times)
    
    # Memory stats
    memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
    memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)  # GB
    
    return BenchmarkResult(
        model_name=f"Mistral (dummy, h={hidden_size}, layers={num_hidden_layers}, nano_batch)",
        num_prompts=len(prompts),
        total_tokens_generated=total_tokens,
        total_time_seconds=total_time,
        tokens_per_second=tokens_per_second,
        average_latency_per_token_ms=avg_latency_per_token,
        first_token_latency_ms=avg_first_token_latency,
        memory_allocated_gb=memory_allocated,
        memory_reserved_gb=memory_reserved,
    )


def main():
    """Run nano_batch benchmark."""
    # Get the directory of this script to find test_prompts.json
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompts_file = os.path.join(script_dir, "test_prompts.json")
    prompts = load_test_prompts(prompts_file)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    result = run_nano_batch_benchmark(
        prompts=prompts,
        max_new_tokens=200,
        num_blocks=1000,
        block_size=32,
        device=device,
    )
    
    # Print results
    print("\n" + "="*60)
    print("nano_batch Engine Benchmark Results (Dummy Model)")
    print("="*60)
    print(f"Model: {result.model_name}")
    print(f"Prompts: {result.num_prompts}")
    print(f"Total Tokens Generated: {result.total_tokens_generated}")
    print(f"Total Time: {result.total_time_seconds:.2f}s")
    print(f"Throughput: {result.tokens_per_second:.2f} tokens/s")
    print(f"Avg Latency per Token: {result.average_latency_per_token_ms:.2f}ms")
    print(f"Avg First Token Latency: {result.first_token_latency_ms:.2f}ms")
    print(f"Memory Allocated: {result.memory_allocated_gb:.2f} GB")
    print(f"Memory Reserved: {result.memory_reserved_gb:.2f} GB")
    print("="*60)
    
    # Save results
    with open("dummy_nano_batch_results.json", "w") as f:
        json.dump(asdict(result), f, indent=2)
    
    print("\nResults saved to dummy_nano_batch_results.json")


if __name__ == "__main__":
    main()
