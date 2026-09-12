import torch

def dense_attention(Q, K, V, causal=False):
    """
    Scaled dot-product attention. Optionally causal (word i can't see future words).
    """
    d_k = Q.shape[-1]
    seq_len = Q.shape[0]
    scores = Q @ K.T
    scores = scores / (d_k ** 0.5)
    if causal:
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
    attn_weights = torch.softmax(scores, dim=-1)
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
    output = attn_weights @ V
    return output, attn_weights


def sliding_window_mask(seq_len, window_size):
    """
    True = masked. Word i can only see itself and the previous window_size words.
    """
    mask = torch.ones(seq_len, seq_len).bool()
    for i in range(seq_len):
        start = max(0, i - window_size)
        end = i
        mask[i, start:end+1] = False
    return mask


def sparse_attention(Q, K, V, window_size):
    """
    Sliding-window attention: word i only attends to itself and the previous window_size words.
    """
    d_k = Q.shape[-1]
    seq_len = Q.shape[0]
    scores = Q @ K.T
    scores = scores / (d_k ** 0.5)
    mask = sliding_window_mask(seq_len, window_size)
    scores = scores.masked_fill(mask, float('-inf'))
    attn_weights = torch.softmax(scores, dim=-1)
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
    output = attn_weights @ V
    return output, attn_weights
