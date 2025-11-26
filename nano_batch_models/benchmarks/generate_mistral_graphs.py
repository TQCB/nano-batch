"""
Generate professional, Mistral-style benchmark graphs.

This script runs benchmarks at different concurrency levels and generates
high-quality visualizations using seaborn.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import os
import torch
import numpy as np
from typing import List, Dict

# Import benchmark functions
# We need to add the parent directory to path to import from sibling scripts if needed
# But since we are in the same package, relative imports might work if run as module
# Or we can just import assuming we run from root
from nano_batch_models.benchmarks.dummy_nano_batch_benchmark import run_nano_batch_benchmark
from nano_batch_models.benchmarks.dummy_baseline_hf import run_hf_baseline_benchmark
from nano_batch_models.benchmarks.dummy_baseline_hf import load_test_prompts

# Set Mistral-style theme
def set_mistral_style():
    sns.set_theme(style="white", context="talk")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
    plt.rcParams["grid.linestyle"] = "--"

# Colors
COLOR_NANO = "#F5A623"  # Mistral Orange/Gold
COLOR_BASE = "#BDC3C7"  # Light Grey

def run_scaling_benchmarks():
    """Run benchmarks at different concurrency levels."""
    concurrency_levels = [1, 10, 25, 50]
    results = []
    
    # Load prompts
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompts_file = os.path.join(script_dir, "test_prompts.json")
    all_prompts = load_test_prompts(prompts_file)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Running scaling benchmarks on {device}...")
    
    for n in concurrency_levels:
        print(f"\n--- Concurrency: {n} ---")
        current_prompts = all_prompts[:n]
        
        # Baseline
        print("Running Baseline...")
        base_res = run_hf_baseline_benchmark(
            prompts=current_prompts,
            max_new_tokens=100, # Use 100 for faster scaling test
            device=device
        )
        results.append({
            "System": "HuggingFace Baseline",
            "Concurrency": n,
            "Throughput": base_res.tokens_per_second,
            "Avg Latency": base_res.average_latency_per_token_ms,
            "First Token Latency": base_res.first_token_latency_ms,
        })
        
        # Nano Batch
        print("Running Nano Batch...")
        nano_res = run_nano_batch_benchmark(
            prompts=current_prompts,
            max_new_tokens=100,
            num_blocks=1000,
            block_size=32,
            device=device
        )
        results.append({
            "System": "nano_batch",
            "Concurrency": n,
            "Throughput": nano_res.tokens_per_second,
            "Avg Latency": nano_res.average_latency_per_token_ms,
            "First Token Latency": nano_res.first_token_latency_ms,
        })
        
    return pd.DataFrame(results)

def plot_throughput_comparison(df):
    """Plot throughput comparison at max concurrency."""
    max_n = df["Concurrency"].max()
    data = df[df["Concurrency"] == max_n]
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=data,
        x="System",
        y="Throughput",
        palette=[COLOR_BASE, COLOR_NANO],
        hue="System",
        legend=False
    )
    
    # Annotate
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.1f}",
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='bottom',
            fontsize=14, fontweight='bold',
            xytext=(0, 5), textcoords='offset points'
        )
        
    plt.title(f"Throughput Comparison (Concurrency={max_n})", fontweight='bold', pad=20)
    plt.ylabel("Tokens per Second")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig("mistral_throughput.png")
    print("Saved mistral_throughput.png")

def plot_scaling(df):
    """Plot throughput scaling vs concurrency."""
    plt.figure(figsize=(12, 7))
    
    sns.lineplot(
        data=df,
        x="Concurrency",
        y="Throughput",
        hue="System",
        palette=[COLOR_BASE, COLOR_NANO],
        style="System",
        markers=True,
        dashes=False,
        linewidth=3,
        markersize=10
    )
    
    plt.title("Throughput Scaling vs Concurrency", fontweight='bold', pad=20)
    plt.ylabel("Throughput (tokens/s)")
    plt.xlabel("Number of Concurrent Requests")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig("mistral_scaling.png")
    print("Saved mistral_scaling.png")

def plot_latency_comparison(df):
    """Plot latency comparison (Avg and TTFT)."""
    max_n = df["Concurrency"].max()
    data = df[df["Concurrency"] == max_n]
    
    # Melt for grouped bar chart
    melted = data.melt(
        id_vars=["System"],
        value_vars=["Avg Latency", "First Token Latency"],
        var_name="Metric",
        value_name="Latency (ms)"
    )
    
    # Create two subplots because scales are very different
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Avg Latency
    sns.barplot(
        data=melted[melted["Metric"] == "Avg Latency"],
        x="System",
        y="Latency (ms)",
        palette=[COLOR_BASE, COLOR_NANO],
        hue="System",
        legend=False,
        ax=ax1
    )
    ax1.set_title("Average Latency per Token", fontweight='bold')
    ax1.set_xlabel("")
    
    for p in ax1.patches:
        ax1.annotate(
            f"{p.get_height():.1f} ms",
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='bottom',
            fontsize=12, fontweight='bold',
            xytext=(0, 5), textcoords='offset points'
        )

    # TTFT
    sns.barplot(
        data=melted[melted["Metric"] == "First Token Latency"],
        x="System",
        y="Latency (ms)",
        palette=[COLOR_BASE, COLOR_NANO],
        hue="System",
        legend=False,
        ax=ax2
    )
    ax2.set_title("Time to First Token (TTFT)", fontweight='bold')
    ax2.set_xlabel("")
    
    for p in ax2.patches:
        ax2.annotate(
            f"{p.get_height():.0f} ms",
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='bottom',
            fontsize=12, fontweight='bold',
            xytext=(0, 5), textcoords='offset points'
        )
        
    plt.tight_layout()
    plt.savefig("mistral_latency.png")
    print("Saved mistral_latency.png")

def main():
    set_mistral_style()
    
    # Run benchmarks
    df = run_scaling_benchmarks()
    
    # Generate plots
    plot_throughput_comparison(df)
    plot_scaling(df)
    plot_latency_comparison(df)
    
    print("\nAll graphs generated successfully!")

if __name__ == "__main__":
    main()
