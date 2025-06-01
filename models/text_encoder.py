import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

from .lstm_custom import LSTMTextEncoder
from .transformer_encoder import TransformerTextEncoder


class TextEncoder(nn.Module):
    def __init__(
        self, vocab_size, embed_dim=256, hidden_dim=512, model_name="transformer"
    ):
        """
        参数说明：
          - vocab_size: 词表大小
          - embed_dim: 嵌入维度（同时也是目标输出维度）
          - hidden_dim: LSTM 隐藏层维度（仅在 model_name='lstm' 时有效）
          - model_name: 指定使用哪种编码器，可选 'lstm' 或 'transformer'
        """
        super().__init__()

        if model_name == "lstm":
            # 使用手写版 LSTMTextEncoder，注意此处 LSTMTextEncoder 内部已包含词嵌入层和全连接映射
            self.encoder = LSTMTextEncoder(vocab_size, embed_dim, hidden_dim)
        elif model_name == "transformer":
            print("true")
            # 使用手写版 TransformerTextEncoder，注意内部参数可根据需求调整，如头数、最大序列长度等
            self.encoder = TransformerTextEncoder(
                vocab_size, embed_dim, num_heads=4, padding_idx=0, max_len=100
            )
        else:
            raise ValueError(
                "Unsupported model_name: choose either 'lstm' or 'transformer'"
            )

    def forward(self, captions):
        # 直接调用内部 encoder 模块进行前向传播
        return self.encoder(captions)  # 输出形状：[batch, embed_dim]


class PretrainedBert(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        model_name="bert-base-uncased",
        freeze_backbone=True,
        num_unfrozen_layers=4,
    ):
        super().__init__()
        self.backbone = BertModel.from_pretrained(
            model_name,
            local_files_only=True,
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            # 解冻最后 N 层 encoder
            for p in self.backbone.encoder.layer[-num_unfrozen_layers:].parameters():
                p.requires_grad = True
            # 解冻 pooler 层
            for p in self.backbone.pooler.parameters():
                p.requires_grad = True

        self.proj = nn.Linear(self.backbone.config.hidden_size, embed_dim, bias=False)

    def forward(self, caption_ids, attn_mask):
        out = self.backbone(
            input_ids=caption_ids,
            attention_mask=attn_mask,
            return_dict=True,
        )
        cls = out.pooler_output  # (B, 768)
        # or: cls = out.last_hidden_state[:, 0]
        embeddings = self.proj(cls)  # (B, embed_dim)
        return F.normalize(embeddings, p=2, dim=1)
