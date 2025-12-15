import torch
import torch.nn as nn


class CharPolicy(nn.Module):
    """
    轻量字符级策略/语言模型：嵌入 -> GRU -> 线性输出 logits。
    既可用于 teacher forcing 预热，也可扩展 RL。
    """

    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor):
        # input_ids: (batch, seq_len)
        x = self.embed(input_ids)
        h, hidden = self.gru(x)
        logits = self.head(h)
        return logits, hidden

    def step(self, input_ids: torch.Tensor, hidden=None):
        """
        单步前向，用于自回归解码。
        input_ids: (batch, seq_len=1)
        """
        x = self.embed(input_ids)
        h, hidden = self.gru(x, hidden)
        logits = self.head(h)
        return logits, hidden
