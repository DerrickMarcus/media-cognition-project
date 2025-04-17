import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)].to(x.device)


# ----- 填空题 1: 实现 MultiHeadSelfAttention 模块 -----
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim 必须能整除 num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    # 请完成forward函数
    def forward(self, x, mask=None):
        # x: [B, T, D]，D=embed_dim
        B, T, D = x.size()

        # my code begin
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
        attn_weight = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weight, v).transpose(1, 2).reshape(B, T, D)
        # my code end

        # 通过输出投影层输出结果
        return self.out_proj(attn_output)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ln1 = nn.LayerNorm(embed_dim)
        # 初始化前馈网络，先 Linear(embed_dim, embed_dim*4) -> ReLU -> Linear(embed_dim*4, embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        # 残差连接：先 LayerNorm，再 MultiHeadSelfAttention，再加上输入
        # 计算自注意力块输出，并加上原输入，实现残差连接
        x = x + self.attn(self.ln1(x), mask)
        # 同理，对于前馈网络：先 LayerNorm，再前馈计算，再加上输入
        # 计算前馈网络输出，并加上残差
        x = x + self.ff(self.ln2(x))
        return x


# ----- 填空题 2: 实现 TransformerTextEncoder 模块 -----
class TransformerTextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=256,
        num_heads=4,
        num_layers=4,
        padding_idx=0,
        max_len=100,
    ):
        super().__init__()
        # 初始化嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        # 初始化位置编码器，参数为 embed_dim 和 max_len
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=max_len)
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)]
        )
        self.fc = nn.Linear(embed_dim, embed_dim)

    # 请完成forward函数
    def forward(self, captions):
        # captions: [B, T]

        # my code begin
        B, T = captions.size()
        x = self.embedding(captions) * math.sqrt(self.embedding.weight.shape[1])
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=1)
        out = self.fc(x)

        # my code end

        return F.normalize(out, p=2, dim=1)


# ----- 测试单元 -----
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 1000
    # 随机生成输入：形状为 [batch=4, T=20]
    dummy_captions = torch.randint(0, vocab_size, (4, 20)).to(device)
    # 实例化 TransformerTextEncoder，要求使用 4 层（num_layers=4）
    model = TransformerTextEncoder(
        vocab_size, embed_dim=256, num_heads=4, num_layers=4, padding_idx=0, max_len=100
    ).to(device)
    output = model(dummy_captions)
    print("输出嵌入张量的形状：", output.shape)  # 期望输出：[4, 256]
