"""
nano_batch engine benchmark with PagedAttention.

This script benchmarks the custom nano_batch inference engine with Triton
PagedAttention kernels and continuous batching.
"""

import time
import torch
import json
from typing import List
from dataclasses import asdict

import sys
sys.path.insert(0, "../")

from nano_batch import MistralForCausalLM, PagedMistralForCausalLM, MistralTokenizer
from nano_batch.inference_engine import InferenceEngine, GenerationConfig
from benchmarks.baseline_hf import BenchmarkResult, load_test_prompts


def run_nano_batch_benchmark(
    model_name: str,
    prompts: List[str],
    max_new_tokens: int = 100,
    num_blocks: int = 100,
    block_size: int = 16,
    device: str = "cuda",
) -> BenchmarkResult:
    """
    Run benchmark with nano_batch engine.
    
    Args:
        model_name: HuggingFace model ID
        prompts: List of input prompts
        max_new_tokens: Tokens to generate per prompt
        num_blocks: Number of KV cache blocks
        block_size: Tokens per block
        device: Device to run on
        
    Returns:
        BenchmarkResult with performance metrics
    """
    print(f"Loading model: {model_name}")
    
    # Load standard model and convert to paged version
    print("Loading standard Mistral model...")
    standard_model = MistralForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    )
    
    print("Converting to PagedMistral...")
    paged_model = PagedMistralForCausalLM.from_standard_mistral(standard_model)
    del standard_model  # Free memory
    
    # Load tokenizer
    tokenizer = MistralTokenizer(model_name)
    
    # Create inference engine
    print(f"Initializing inference engine (blocks={num_blocks}, block_size={block_size})...")
    engine = InferenceEngine(
        model=paged_model,
        tokenizer=tokenizer,
        num_blocks=num_blocks,
        block_size=block_size,
        device=device,
    )
    
    # Warm up
    print("Warming up...")
    gen_config = GenerationConfig(max_tokens=10)
    _ = engine.generate("Hello", gen_config)
    
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
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
    
    for i, prompt in enumerate(prompts):
        ft_start = time.time()
        
        # Generate
        output = engine.generate(prompt, gen_config)
        
        ft_end = time.time()
        first_token_times.append((ft_end - ft_start) * 1000)  # ms
        
        # Count tokens (simplified)
        num_generated = len(tokenizer.encode(output))
        total_tokens += num_generated
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(prompts)} prompts")
    
    torch.cuda.synchronize()
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
        model_name=f"{model_name} (nano_batch)",
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
    model_name = "mistralai/Mistral-7B-v0.2"
    prompts = load_test_prompts("test_prompts.json")
    
    result = run_nano_batch_benchmark(
        model_name=model_name,
        prompts=prompts,
        max_new_tokens=100,
        num_blocks=100,
        block_size=16,
    )
    
    # Print results
    print("\n" + "="*60)
    print("nano_batch Engine Benchmark Results")
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
    with open("nano_batch_results.json", "w") as f:
        json.dump(asdict(result), f, indent=2)
    
    print("\nResults saved to nano_batch_results.json")


if __name__ == "__main__":
    main()
