import torch
import torch.nn as nn
from attention import dense_attention, sparse_attention

class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, causal=True, sparse=False, window_size=None):
        super().__init__()
        self.sparse = sparse
        self.window_size = window_size
        self.causal = causal

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        self.feedforward = FeedForward(d_model)

    def forward(self, x):
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        if self.sparse:
            attn_out, _ = sparse_attention(Q, K, V, self.window_size)
        else:
            attn_out, _ = dense_attention(Q, K, V, causal=self.causal)

        x = x + attn_out
        x = x + self.feedforward(x)
        return x

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model, block_size, sparse=False, window_size=None):
        super().__init__()
        self.block_size = block_size

        self.token_embedding_table = nn.Embedding(vocab_size, d_model)
        self.position_embedding_table = nn.Embedding(block_size, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model,
                causal=True,
                sparse=sparse,
                window_size=window_size
            )
            for _ in range(2)
        ])

        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_emb = self.token_embedding_table(idx)
        position_emb = self.position_embedding_table(
            torch.arange(T, device=idx.device)
        )

        x = token_emb + position_emb

        for block in self.blocks:
            x = block(x)

        logits = self.output_head(x)

        loss = None
        if targets is not None:
            logits_flat = logits.reshape(B * T, -1)
            targets_flat = targets.reshape(B * T)
            loss = nn.functional.cross_entropy(logits_flat, targets_flat)

        return logits, loss

@torch.no_grad()
def generate_text(model, start_text, num_chars, char_to_idx, idx_to_char):
    model.eval()
    device = next(model.parameters()).device

    context = torch.tensor(
        [[char_to_idx[c] for c in start_text]],
        dtype=torch.long,
        device=device
    )

    for _ in range(num_chars):
        idx_cond = context[:, -model.block_size:]
        logits, _ = model(idx_cond)
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        next_idx = torch.multinomial(probs, num_samples=1)
        context = torch.cat((context, next_idx), dim=1)

    return ''.join(idx_to_char[i.item()] for i in context[0])
