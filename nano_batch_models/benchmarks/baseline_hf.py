"""
Baseline benchmark using standard HuggingFace transformers.

This script measures the performance of the vanilla HuggingFace Mistral implementation
to establish a baseline for comparison with the nano_batch engine.
"""

import time
import torch
import json
from typing import List, Dict
from dataclasses import dataclass, asdict

from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    model_name: str
    num_prompts: int
    total_tokens_generated: int
    total_time_seconds: float
    tokens_per_second: float
    average_latency_per_token_ms: float
    first_token_latency_ms: float
    memory_allocated_gb: float
    memory_reserved_gb: float


def run_hf_baseline_benchmark(
    model_name: str,
    prompts: List[str],
    max_new_tokens: int = 100,
    device: str = "cuda",
) -> BenchmarkResult:
    """
    Run benchmark with HuggingFace baseline.
    
    Args:
        model_name: HuggingFace model ID
        prompts: List of input prompts
        max_new_tokens: Tokens to generate per prompt
        device: Device to run on
        
    Returns:
        BenchmarkResult with performance metrics
    """
    print(f"Loading model: {model_name}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map=device,
    )
    model.save_pretrained("model")
    model.eval()
    
    # Warm up
    print("Warming up...")
    with torch.no_grad():
        dummy_input = tokenizer.encode("Hello", return_tensors="pt").to(device)
        _ = model.generate(dummy_input, max_new_tokens=10)
    
    if device == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    else:
        torch.cpu.synchronize()
    
    # Benchmark
    print(f"\nRunning benchmark with {len(prompts)} prompts...")
    total_tokens = 0
    first_token_times = []
    
    start_time = time.time()
    
    for i, prompt in enumerate(prompts):
        # Encode
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        # Measure first token latency
        ft_start = time.time()
        
        with torch.no_grad():
            # Generate
            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
            )
        
        ft_end = time.time()
        first_token_times.append((ft_end - ft_start) * 1000)  # ms
        
        # Count tokens
        num_generated = output.shape[1] - input_ids.shape[1]
        total_tokens += num_generated
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(prompts)} prompts")
    
    if device == "cuda":
        torch.cuda.synchronize()
    else:
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
        model_name=model_name,
        num_prompts=len(prompts),
        total_tokens_generated=total_tokens,
        total_time_seconds=total_time,
        tokens_per_second=tokens_per_second,
        average_latency_per_token_ms=avg_latency_per_token,
        first_token_latency_ms=avg_first_token_latency,
        memory_allocated_gb=memory_allocated,
        memory_reserved_gb=memory_reserved,
    )


def load_test_prompts(prompt_file: str = None) -> List[str]:
    """Load test prompts from file or use defaults."""
    if prompt_file:
        with open(prompt_file) as f:
            data = json.load(f)
        return data["prompts"]
    else:
        # Default test prompts
        return [
            "The future of artificial intelligence is",
            "In a world where technology advances rapidly,",
            "Once upon a time in a distant galaxy,",
            "The key to understanding quantum mechanics",
            "Climate change is one of the most pressing",
        ]


def main():
    """Run baseline benchmark."""
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    prompts = load_test_prompts()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    result = run_hf_baseline_benchmark(
        model_name=model_name,
        prompts=prompts,
        max_new_tokens=100,
        device=device,
    )
    
    # Print results
    print("\n" + "="*60)
    print("HuggingFace Baseline Benchmark Results")
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
    with open("hf_baseline_results.json", "w") as f:
        json.dump(asdict(result), f, indent=2)
    
    print("\nResults saved to hf_baseline_results.json")


if __name__ == "__main__":
    main()
