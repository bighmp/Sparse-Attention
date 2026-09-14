# Sparse Attention from Scratch

A from-scratch PyTorch implementation of dense causal attention and sliding-window sparse attention.

## What I Built

- Manual dense causal self-attention using matrix multiplication, masking, and softmax
- Sliding-window sparse attention
- Correctness tests comparing sparse attention with dense attention
- Benchmarking across sequence lengths from 512 to 8192
- A small 2-layer character-level GPT for comparing dense and sparse attention
- Training-loss and benchmark plots
- NaN handling for masked attention

## Files

- `attention.py` — dense and sliding-window attention implementations
- `train.py` — TinyGPT model and text generation
- `test_correctness.py` — correctness checks
- `benchmark.py` — attention benchmark and plots
- `benchmark_plot.png` — benchmark results
- `loss_comparison.png` — dense vs. sparse training loss
- `WRITEUP.md` — detailed implementation notes and findings

## Running

Install PyTorch and matplotlib, then run:

```bash
python test_correctness.py
python benchmark.py
