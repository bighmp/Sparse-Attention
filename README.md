# Sparse Attention from Scratch

A from-scratch PyTorch implementation of dense causal attention and sliding-window sparse attention.

## Implemented

- Manual dense causal scaled dot-product attention
- Sliding-window sparse attention
- Correctness tests
- NaN handling for masked attention
- Forward-pass benchmarks for sequence lengths 512 to 8192
- 2-layer character-level GPT on TinyShakespeare
- Dense vs. sparse training-loss comparison

## Files

- `attention.py` — dense and sliding-window attention
- `train.py` — TinyGPT model and text generation
- `test_correctness.py` — correctness checks
- `benchmark.py` — attention benchmark
- `benchmark_plot.png` — benchmark results
- `loss_comparison.png` — training loss comparison
- `WRITEUP.md` — detailed writeup

## Installation

```bash
pip install torch matplotlib
