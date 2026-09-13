import torch

def dense_attention(Q, K, V, causal=False):
    """
    Q, K, V: tensors of shape (seq_len, d_k)
    causal: if True, apply causal masking (word i can't see future words)
    Returns: output tensor of shape (seq_len, d_k), and the attention weights (seq_len, seq_len)
    """
    d_k = Q.shape[-1]  # last dimension size, used for scaling
    seq_len = Q.shape[0]

    # 1. Compute raw attention scores (Q against K)
    scores = Q @ K.T  # match by transposing and multipying

    # 2. Scale by sqrt(d_k)
    scores = scores / (d_k**0.5)

    # 3. If causal, mask out future positions
    if causal:
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()  # hint: torch.triu(...)
        scores = scores.masked_fill(mask, float('-inf'))  # hint: masked_fill

    # 4. Softmax to get attention weights
    attn_weights = torch.softmax(scores, dim=-1)

    # 5. Weighted blend of V
    output = attn_weights @ V

    return output, attn_weights


def sliding_window_mask(seq_len, window_size):
    """
    Returns a boolean mask (True = masked/not allowed).
    Word i can only attend to itself and up to `window_size` previous words.
    """
    mask = torch.ones(seq_len, seq_len).bool()  # start with everything masked (True)
    for i in range(seq_len):
        start = max(0, i - window_size)
        end = i  #last allowed column
        mask[i, start:end+1] = False
    return mask
    
