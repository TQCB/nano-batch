# Benchmarking Scripts

This directory contains benchmarking scripts to compare nano_batch with HuggingFace baseline.

## Files

- `baseline_hf.py` - HuggingFace baseline benchmark
- `nano_batch_benchmark.py` - nano_batch engine benchmark
- `compare.py` - Comparison and visualization script
- `test_prompts.json` - Standard test prompts

## Usage

1. **Run HuggingFace baseline:**
   ```bash
   cd python/benchmarks
   python baseline_hf.py
   ```

2. **Run nano_batch benchmark:**
   ```bash
   python nano_batch_benchmark.py
   ```

3. **Compare results:**
   ```bash
   python compare.py
   ```

## Metrics

Both benchmarks measure:
- **Throughput**: Tokens generated per second
- **Latency**: Average milliseconds per token
- **First Token Latency**: Time to generate first token
- **Memory Usage**: GPU memory allocated and reserved
