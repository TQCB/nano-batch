"""
Compare benchmark results between HuggingFace baseline and nano_batch engine.

This script loads the results from both benchmarks and generates a comparison
report with visualizations and speedup calculations.
"""

import json
import matplotlib.pyplot as plt
from typing import Dict
from dataclasses import dataclass


@dataclass
class ComparisonMetrics:
    """Metrics comparing two benchmark runs."""
    throughput_speedup: float
    latency_improvement_percent: float
    memory_reduction_percent: float
    first_token_improvement_percent: float


def load_results(baseline_file: str, nano_batch_file: str) -> tuple:
    """Load both benchmark results."""
    with open(baseline_file) as f:
        baseline = json.load(f)
    
    with open(nano_batch_file) as f:
        nano_batch = json.load(f)
    
    return baseline, nano_batch


def calculate_comparison(baseline: Dict, nano_batch: Dict) -> ComparisonMetrics:
    """Calculate comparison metrics."""
    throughput_speedup = nano_batch["tokens_per_second"] / baseline["tokens_per_second"]
    
    latency_improvement = (
        (baseline["average_latency_per_token_ms"] - nano_batch["average_latency_per_token_ms"]) /
        baseline["average_latency_per_token_ms"] * 100
    )
    
    memory_reduction = (
        (baseline["memory_allocated_gb"] - nano_batch["memory_allocated_gb"]) /
        baseline["memory_allocated_gb"] * 100
    )
    
    first_token_improvement = (
        (baseline["first_token_latency_ms"] - nano_batch["first_token_latency_ms"]) /
        baseline["first_token_latency_ms"] * 100
    )
    
    return ComparisonMetrics(
        throughput_speedup=throughput_speedup,
        latency_improvement_percent=latency_improvement,
        memory_reduction_percent=memory_reduction,
        first_token_improvement_percent=first_token_improvement,
    )


def create_comparison_plots(baseline: Dict, nano_batch: Dict):
    """Create visualization comparing both engines."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("nano_batch vs HuggingFace Baseline Comparison", fontsize=16)
    
    # Throughput
    axes[0, 0].bar(
        ["HuggingFace", "nano_batch"],
        [baseline["tokens_per_second"], nano_batch["tokens_per_second"]],
        color=["#3498db", "#e74c3c"]
    )
    axes[0, 0].set_title("Throughput (tokens/s)")
    axes[0, 0].set_ylabel("Tokens per Second")
    
    # Latency
    axes[0, 1].bar(
        ["HuggingFace", "nano_batch"],
        [baseline["average_latency_per_token_ms"], nano_batch["average_latency_per_token_ms"]],
        color=["#3498db", "#e74c3c"]
    )
    axes[0, 1].set_title("Avg Latency per Token (ms)")
    axes[0, 1].set_ylabel("Milliseconds")
    
    # Memory
    axes[1, 0].bar(
        ["HuggingFace", "nano_batch"],
        [baseline["memory_allocated_gb"], nano_batch["memory_allocated_gb"]],
        color=["#3498db", "#e74c3c"]
    )
    axes[1, 0].set_title("Memory Allocated (GB)")
    axes[1, 0].set_ylabel("Gigabytes")
    
    # First Token Latency
    axes[1, 1].bar(
        ["HuggingFace", "nano_batch"],
        [baseline["first_token_latency_ms"], nano_batch["first_token_latency_ms"]],
        color=["#3498db", "#e74c3c"]
    )
    axes[1, 1].set_title("First Token Latency (ms)")
    axes[1, 1].set_ylabel("Milliseconds")
    
    plt.tight_layout()
    plt.savefig("benchmark_comparison.png", dpi=300)
    print("Comparison plot saved to benchmark_comparison.png")


def print_comparison_report(baseline: Dict, nano_batch: Dict, metrics: ComparisonMetrics):
    """Print detailed comparison report."""
    print("\n" + "="*80)
    print("BENCHMARK COMPARISON REPORT")
    print("="*80)
    
    print("\n--- HuggingFace Baseline ---")
    print(f"  Throughput: {baseline['tokens_per_second']:.2f} tokens/s")
    print(f"  Avg Latency: {baseline['average_latency_per_token_ms']:.2f} ms/token")
    print(f"  First Token Latency: {baseline['first_token_latency_ms']:.2f} ms")
    print(f"  Memory: {baseline['memory_allocated_gb']:.2f} GB")
    
    print("\n--- nano_batch Engine ---")
    print(f"  Throughput: {nano_batch['tokens_per_second']:.2f} tokens/s")
    print(f"  Avg Latency: {nano_batch['average_latency_per_token_ms']:.2f} ms/token")
    print(f"  First Token Latency: {nano_batch['first_token_latency_ms']:.2f} ms")
    print(f"  Memory: {nano_batch['memory_allocated_gb']:.2f} GB")
    
    print("\n--- Performance Improvements ---")
    print(f"  Throughput Speedup: {metrics.throughput_speedup:.2f}x")
    print(f"  Latency Improvement: {metrics.latency_improvement_percent:+.1f}%")
    print(f"  First Token Improvement: {metrics.first_token_improvement_percent:+.1f}%")
    print(f"  Memory Reduction: {metrics.memory_reduction_percent:+.1f}%")
    
    print("\n" + "="*80)
    print("\nINTERPRETATION:")
    if metrics.throughput_speedup > 1.0:
        print(f"✓ nano_batch is {metrics.throughput_speedup:.2f}x FASTER than baseline!")
    else:
        print(f"⚠ nano_batch is {1/metrics.throughput_speedup:.2f}x slower than baseline.")
    
    if metrics.memory_reduction_percent > 0:
        print(f"✓ nano_batch uses {metrics.memory_reduction_percent:.1f}% LESS memory!")
    else:
        print(f"⚠ nano_batch uses {abs(metrics.memory_reduction_percent):.1f}% MORE memory.")
    
    print("\n" + "="*80)


def main():
    """Run comparison."""
    try:
        baseline, nano_batch = load_results("hf_baseline_results.json", "nano_batch_results.json")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run both benchmarks first:")
        print("  1. python baseline_hf.py")
        print("  2. python nano_batch_benchmark.py")
        return
    
    metrics = calculate_comparison(baseline, nano_batch)
    print_comparison_report(baseline, nano_batch, metrics)
    
    # Create plots
    try:
        create_comparison_plots(baseline, nano_batch)
    except ImportError:
        print("\nNote: matplotlib not available, skipping plots")


if __name__ == "__main__":
    main()
