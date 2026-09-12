import torch
from attention import dense_attention, sparse_attention

def test_sparse_matches_dense(seq_len=5, d_model=8, window_size=2, tolerance=1e-5):
    torch.manual_seed(1)
    x = torch.randn(seq_len, d_model)
    W_Q, W_K, W_V = torch.randn(d_model, d_model), torch.randn(d_model, d_model), torch.randn(d_model, d_model)
    Q, K, V = x @ W_Q, x @ W_K, x @ W_V

    dense_out, _ = dense_attention(Q, K, V, causal=True)
    sparse_out, _ = sparse_attention(Q, K, V, window_size)

    all_passed = True
    for i in range(seq_len):
        if i <= window_size:
            match = torch.allclose(dense_out[i], sparse_out[i], atol=tolerance)
            status = "PASS" if match else "FAIL"
            if not match:
                all_passed = False
            print(f"word {i}: {status} (expected match)")
        else:
            diff = (dense_out[i] - sparse_out[i]).abs().max().item()
            print(f"word {i}: differs from dense (expected) — max diff {diff:.4f}")

    print("\nOVERALL:", "PASS" if all_passed else "FAIL")

if __name__ == "__main__":
    test_sparse_matches_dense()
