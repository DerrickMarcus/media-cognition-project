import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import CLIPVisionModel, ViTModel

from .resnet_custom import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152


class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=256, model_name="resnet18"):
        super().__init__()
        if model_name == "resnet18":
            self.encoder = ResNet18(embed_dim)
        elif model_name == "resnet34":
            self.encoder = ResNet34(embed_dim)
        elif model_name == "resnet50":
            self.encoder = ResNet50(embed_dim)
        elif model_name == "resnet101":
            self.encoder = ResNet101(embed_dim)
        elif model_name == "resnet152":
            self.encoder = ResNet152(embed_dim)
        else:
            raise ValueError(
                "Unsupported model_name: choose either 'resnet18', 'resnet34', 'resnet50', 'resnet101', or 'resnet152'"
            )

    def forward(self, images):
        embeddings = self.encoder(images)  # [batch, embed_dim]
        return embeddings


class PretrainedResNet(nn.Module):
    def __init__(self, embed_dim=256, model_name="resnet18", freeze_backbone=True):
        super().__init__()
        if model_name == "resnet18":
            self.backbone = models.resnet18(
                weights=models.ResNet18_Weights.IMAGENET1K_V1
            )
        elif model_name == "resnet34":
            self.backbone = models.resnet34(
                weights=models.ResNet34_Weights.IMAGENET1K_V1
            )
        elif model_name == "resnet50":
            self.backbone = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V2
            )
        elif model_name == "resnext50_32x4d":
            self.backbone = models.resnext50_32x4d(
                weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2
            )
        elif model_name == "resnet101":
            self.backbone = models.resnet101(
                weights=models.ResNet101_Weights.IMAGENET1K_V2
            )
        elif model_name == "resnext101_32x8d":
            self.backbone = models.resnext101_32x8d(
                weights=models.ResNeXt101_32X8D_Weights.IMAGENET1K_V2
            )
        elif model_name == "resnext101_64x4d":
            self.backbone = models.resnext101_64x4d(
                weights=models.ResNeXt101_64X4D_Weights.IMAGENET1K_V1
            )
        elif model_name == "resnet152":
            self.backbone = models.resnet152(
                weights=models.ResNet152_Weights.IMAGENET1K_V2
            )
        else:
            raise ValueError("Unsupported model_name.")

        num_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # 移除最后的分类层

        # 冻结全部参数，只解冻最后一层 layer4
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            for p in self.backbone.layer4.parameters():
                p.requires_grad = True

        self.proj = nn.Linear(num_feats, embed_dim)

    def forward(self, images):
        features = self.backbone(images)
        embeddings = self.proj(features)
        return F.normalize(embeddings, p=2, dim=1)


class PretrainedViT(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        model_name="google/vit-base-patch16-224-in21k",
        freeze_backbone=True,
        num_unfrozen_layers=3,
    ):
        super().__init__()
        self.backbone = ViTModel.from_pretrained(
            model_name,
            local_files_only=True,
        )

        num_feats = self.backbone.config.hidden_size  # 768 for ViT-B/16

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            for p in self.backbone.encoder.layer[-num_unfrozen_layers].parameters():
                p.requires_grad = True

        self.proj = nn.Linear(num_feats, embed_dim)

    def forward(self, images):
        # ViT 模型的输入需要是一个字典，包含 `pixel_values`
        outputs = self.backbone(pixel_values=images)
        # 取 CLS token 的输出作为特征表示
        features = outputs.last_hidden_state[:, 0, :]
        embeddings = self.proj(features)
        return F.normalize(embeddings, p=2, dim=1)


class PretrainedCLIPViT(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        model_name="openai/clip-vit-base-patch16",
        freeze_backbone=True,
        num_unfrozen_layers=1,
    ):
        super().__init__()
        self.backbone = CLIPVisionModel.from_pretrained(
            model_name,
            local_files_only=True,
        )
        num_feats = self.backbone.config.hidden_size

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            # 解冻最后几层
            for p in self.backbone.vision_model.encoder.layers[
                -num_unfrozen_layers
            ].parameters():
                p.requires_grad = True
            # 解冻视觉投影层
            for p in self.backbone.vision_model.post_layernorm.parameters():
                p.requires_grad = True

        # 把 CLIP-ViT 输出的特征（通常是 512 维）再投到 embed_dim
        self.proj = nn.Linear(num_feats, embed_dim, bias=False)

    def forward(self, pixel_values):
        # pixel_values: 需预处理到 CLIP 要求的输入格式 (batch, 3, 224, 224)，
        outputs = self.backbone(pixel_values=pixel_values, return_dict=True)
        features = outputs.pooler_output  # [batch, hidden_size], 如 512
        embeddings = self.proj(features)  # [batch, embed_dim]
        return F.normalize(embeddings, p=2, dim=1)
