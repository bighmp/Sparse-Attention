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

## Running

Install dependencies with `pip install torch matplotlib`.

Run the correctness tests with `python test_correctness.py`.

Run the benchmark with `python benchmark.py`.

The benchmark tests sequence lengths of 512, 1024, 2048, 4096, and 8192.

## Results

### Benchmark

The benchmark compares dense causal attention with sliding-window attention as sequence length increases.

The score matrix grows quadratically with sequence length. For float32 scores:

| Sequence length | Score matrix |
|---:|---:|
| 512 | 1 MB |
| 1024 | 4 MB |
| 2048 | 16 MB |
| 4096 | 64 MB |
| 8192 | 256 MB |

See `benchmark_plot.png` for the benchmark results.

An important limitation is that the current sparse implementation still constructs the full `Q @ K^T` matrix before applying the mask. Therefore, it demonstrates the sparse attention pattern but does not eliminate the underlying quadratic computation.

### Training

See `loss_comparison.png` for the training-loss comparison.

A 2-layer character-level GPT was trained with dense and sliding-window attention for 49,000 steps on Colab CPU.

| Model | Final loss |
|---|---:|
| Dense | 1.9101 |
| Sliding Window | 1.8003 |

Each run took approximately 6 minutes.

The sparse model was not worse in this experiment. Since this is a small character-level model with a relatively short context, much of the useful signal is local, so restricting attention to a window did not remove enough information to hurt this run.

## Limitations

The sparse implementation currently uses a dense score matrix followed by masking, so it does not provide the computational savings of a true sparse attention kernel.

I also attempted a BigBird-style local + global + random attention pattern but did not complete it. The completed implementation therefore focuses on dense and sliding-window attention.

See `WRITEUP.md` for the implementation details, debugging process, numerical issues, benchmark analysis, and conclusions.
