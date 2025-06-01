import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, init_temperature=0.07):
        super().__init__()
        # 初始化温度参数，并设置为可学习参数
        self.temperature = nn.Parameter(torch.tensor(init_temperature))

    def forward(self, image_embeds, text_embeds):
        """
        image_embeds, text_embeds: [batch, embed_dim]
        """
        # L2归一化
        image_embeds = F.normalize(image_embeds, p=2, dim=1)
        text_embeds = F.normalize(text_embeds, p=2, dim=1)

        # 计算相似度矩阵：图像到文本方向、文本到图像方向
        logits_image_to_text = image_embeds @ text_embeds.t() / self.temperature

        # 标签是每个样本对应的索引
        batch_size = image_embeds.size(0)
        labels = torch.arange(batch_size, device=image_embeds.device)

        # InfoNCE 损失
        loss_image_to_text = F.cross_entropy(logits_image_to_text, labels)
        loss_text_to_image = F.cross_entropy(logits_image_to_text.t(), labels)

        # 双向损失
        loss = (loss_image_to_text + loss_text_to_image) / 2

        return loss
