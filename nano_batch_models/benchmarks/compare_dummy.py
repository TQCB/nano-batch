"""
Compare dummy benchmark results between HuggingFace baseline and nano_batch engine.

This script loads the results from both dummy benchmarks and generates a comparison
report with visualizations and speedup calculations.
"""

import json
import os
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
    
    # Handle division by zero for memory (both are 0.0 in dummy benchmarks)
    if baseline["memory_allocated_gb"] > 0:
        memory_reduction = (
            (baseline["memory_allocated_gb"] - nano_batch["memory_allocated_gb"]) /
            baseline["memory_allocated_gb"] * 100
        )
    else:
        memory_reduction = 0.0
    
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
    fig.suptitle("nano_batch vs HuggingFace Baseline Comparison (Dummy Models)", fontsize=16)
    
    # Throughput
    axes[0, 0].bar(
        ["HuggingFace", "nano_batch"],
        [baseline["tokens_per_second"], nano_batch["tokens_per_second"]],
        color=["#3498db", "#e74c3c"]
    )
    axes[0, 0].set_title("Throughput (tokens/s)")
    axes[0, 0].set_ylabel("Tokens per Second")
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Latency
    axes[0, 1].bar(
        ["HuggingFace", "nano_batch"],
        [baseline["average_latency_per_token_ms"], nano_batch["average_latency_per_token_ms"]],
        color=["#3498db", "#e74c3c"]
    )
    axes[0, 1].set_title("Avg Latency per Token (ms)")
    axes[0, 1].set_ylabel("Milliseconds")
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Token count comparison
    axes[1, 0].bar(
        ["HuggingFace", "nano_batch"],
        [baseline["total_tokens_generated"], nano_batch["total_tokens_generated"]],
        color=["#3498db", "#e74c3c"]
    )
    axes[1, 0].set_title("Total Tokens Generated")
    axes[1, 0].set_ylabel("Tokens")
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # First Token Latency
    axes[1, 1].bar(
        ["HuggingFace", "nano_batch"],
        [baseline["first_token_latency_ms"], nano_batch["first_token_latency_ms"]],
        color=["#3498db", "#e74c3c"]
    )
    axes[1, 1].set_title("First Token Latency (ms)")
    axes[1, 1].set_ylabel("Milliseconds")
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("dummy_benchmark_comparison.png", dpi=300)
    print("\nComparison plot saved to dummy_benchmark_comparison.png")


def print_comparison_report(baseline: Dict, nano_batch: Dict, metrics: ComparisonMetrics):
    """Print detailed comparison report."""
    print("\n" + "="*80)
    print("DUMMY BENCHMARK COMPARISON REPORT")
    print("="*80)
    
    print("\n--- HuggingFace Baseline ---")
    print(f"  Model: {baseline['model_name']}")
    print(f"  Prompts: {baseline['num_prompts']}")
    print(f"  Total Tokens: {baseline['total_tokens_generated']}")
    print(f"  Throughput: {baseline['tokens_per_second']:.2f} tokens/s")
    print(f"  Avg Latency: {baseline['average_latency_per_token_ms']:.2f} ms/token")
    print(f"  First Token Latency: {baseline['first_token_latency_ms']:.2f} ms")
    print(f"  Total Time: {baseline['total_time_seconds']:.2f} s")
    
    print("\n--- nano_batch Engine ---")
    print(f"  Model: {nano_batch['model_name']}")
    print(f"  Prompts: {nano_batch['num_prompts']}")
    print(f"  Total Tokens: {nano_batch['total_tokens_generated']}")
    print(f"  Throughput: {nano_batch['tokens_per_second']:.2f} tokens/s")
    print(f"  Avg Latency: {nano_batch['average_latency_per_token_ms']:.2f} ms/token")
    print(f"  First Token Latency: {nano_batch['first_token_latency_ms']:.2f} ms")
    print(f"  Total Time: {nano_batch['total_time_seconds']:.2f} s")
    
    print("\n--- Performance Comparison ---")
    print(f"  Throughput Speedup: {metrics.throughput_speedup:.2f}x")
    print(f"  Latency Improvement: {metrics.latency_improvement_percent:+.1f}%")
    print(f"  First Token Improvement: {metrics.first_token_improvement_percent:+.1f}%")
    
    print("\n" + "="*80)
    print("\nINTERPRETATION:")
    if metrics.throughput_speedup > 1.0:
        print(f"✓ nano_batch is {metrics.throughput_speedup:.2f}x FASTER than baseline!")
    else:
        print(f"⚠ nano_batch is {1/metrics.throughput_speedup:.2f}x slower than baseline.")
    
    if metrics.latency_improvement_percent > 0:
        print(f"✓ nano_batch has {metrics.latency_improvement_percent:.1f}% LOWER latency!")
    else:
        print(f"⚠ nano_batch has {abs(metrics.latency_improvement_percent):.1f}% HIGHER latency.")
    
    if metrics.first_token_improvement_percent > 0:
        print(f"✓ nano_batch has {metrics.first_token_improvement_percent:.1f}% FASTER first token!")
    else:
        print(f"⚠ nano_batch has {abs(metrics.first_token_improvement_percent):.1f}% SLOWER first token.")
    
    print("\n" + "="*80)
    print("\nNOTE: These are dummy benchmarks with random weights.")
    print("Results show engine overhead and scheduling efficiency,")
    print("not model quality or actual inference performance.")
    print("="*80)


def main():
    """Run comparison."""
    # Try to find result files in current directory or benchmarks directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(script_dir))
    
    baseline_candidates = [
        "dummy_hf_baseline_results.json",
        os.path.join(script_dir, "dummy_hf_baseline_results.json"),
        os.path.join(parent_dir, "dummy_hf_baseline_results.json"),
    ]
    
    nano_batch_candidates = [
        "dummy_nano_batch_results.json",
        os.path.join(script_dir, "dummy_nano_batch_results.json"),
        os.path.join(parent_dir, "dummy_nano_batch_results.json"),
    ]
    
    baseline_file = None
    nano_batch_file = None
    
    for candidate in baseline_candidates:
        if os.path.exists(candidate):
            baseline_file = candidate
            break
    
    for candidate in nano_batch_candidates:
        if os.path.exists(candidate):
            nano_batch_file = candidate
            break
    
    if not baseline_file or not nano_batch_file:
        print("Error: Could not find result files.")
        print("\nPlease run both benchmarks first:")
        print("  1. python nano_batch_models/benchmarks/dummy_baseline_hf.py")
        print("  2. python nano_batch_models/benchmarks/dummy_nano_batch_benchmark.py")
        return
    
    try:
        baseline, nano_batch = load_results(baseline_file, nano_batch_file)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    metrics = calculate_comparison(baseline, nano_batch)
    print_comparison_report(baseline, nano_batch, metrics)
    
    # Create plots
    try:
        create_comparison_plots(baseline, nano_batch)
    except ImportError:
        print("\nNote: matplotlib not available, skipping plots")
    except Exception as e:
        print(f"\nNote: Could not create plots: {e}")


if __name__ == "__main__":
    main()
