import torch
import time
import matplotlib.pyplot as plt
from attention import dense_attention, sparse_attention

def benchmark(seq_lengths, window_size=32, d_model=64, num_repeats=3):
    results = []
    for seq_len in seq_lengths:
        torch.manual_seed(0)
        Q = torch.randn(seq_len, d_model)
        K = torch.randn(seq_len, d_model)
        V = torch.randn(seq_len, d_model)

        start = time.time()
        for _ in range(num_repeats):
            dense_attention(Q, K, V, causal=True)
        dense_time = (time.time() - start) / num_repeats

        start = time.time()
        for _ in range(num_repeats):
            sparse_attention(Q, K, V, window_size)
        sparse_time = (time.time() - start) / num_repeats

        score_matrix_mb = (seq_len * seq_len * 4) / (1024 ** 2)

        results.append({
            "seq_len": seq_len,
            "dense_time": dense_time,
            "sparse_time": sparse_time,
            "score_matrix_mb": score_matrix_mb,
        })
        print(f"seq_len={seq_len}: dense={dense_time:.4f}s, sparse={sparse_time:.4f}s, score matrix={score_matrix_mb:.1f}MB")

    return results


def plot_results(results):
    seq_lens = [r["seq_len"] for r in results]
    dense_times = [r["dense_time"] for r in results]
    sparse_times = [r["sparse_time"] for r in results]
    memory_mb = [r["score_matrix_mb"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(seq_lens, dense_times, marker='o', label='Dense')
    ax1.plot(seq_lens, sparse_times, marker='o', label='Sparse (sliding window)')
    ax1.set_xlabel('Sequence length')
    ax1.set_ylabel('Time per forward pass (s)')
    ax1.set_title('Wall-clock time vs sequence length')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(seq_lens, memory_mb, marker='o', color='red')
    ax2.set_xlabel('Sequence length')
    ax2.set_ylabel('Score matrix size (MB)')
    ax2.set_title('Memory (score matrix) vs sequence length')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('benchmark_plot.png', dpi=150)
    plt.show()
    print("Saved plot to benchmark_plot.png")


if __name__ == "__main__":
    seq_lengths = [512, 1024, 2048, 4096, 8192]
    results = benchmark(seq_lengths, num_repeats=3)
    plot_results(results)
