# train_and_eval.py

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader import Flickr8kDataset
from loss import contrastive_loss
from models.image_encoder import ImageEncoder
from models.text_encoder import TextEncoder
from utils import SimpleTokenizer


def evaluate(img_encoder, txt_encoder, dataloader, device):
    """
    计算给定数据集上的损失。

    Args:
        img_encoder (nn.Module): 图像编码器模型。
        txt_encoder (nn.Module): 文本编码器模型。
        dataloader (DataLoader): 数据加载器。
        device (torch.device): 设备（CPU 或 GPU）。

    Returns:
        float: 平均损失。
    """
    img_encoder.eval()
    txt_encoder.eval()

    total_loss = 0.0
    with torch.no_grad():
        for images, captions_ids in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            captions_ids = captions_ids.to(device)

            image_embeds = img_encoder(images)  # [batch, embed_dim]
            text_embeds = txt_encoder(captions_ids)  # [batch, embed_dim]

            loss = contrastive_loss(image_embeds, text_embeds)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def evaluate_top_k(img_encoder, txt_encoder, dataloader, device, topk=(1, 5, 10)):
    """
    评估模型在给定数据集上的 Top-K 准确率。

    Args:
        img_encoder (nn.Module): 图像编码器模型。
        txt_encoder (nn.Module): 文本编码器模型。
        dataloader (DataLoader): 数据加载器。
        device (torch.device): 设备（CPU 或 GPU）。
        topk (tuple, optional): 要评估的 Top-K 值。默认为 (1, 5, 10)。

    Returns:
        tuple: 文本到图像和图像到文本的 Top-K 准确率。
    """
    img_encoder.eval()
    txt_encoder.eval()

    all_image_embeds = []
    all_text_embeds = []

    with torch.no_grad():
        for images, captions_ids in tqdm(dataloader, desc="Extracting embeddings"):
            images = images.to(device)
            captions_ids = captions_ids.to(device)

            image_embed = img_encoder(images)  # [batch, dim]
            text_embed = txt_encoder(captions_ids)  # [batch, dim]

            all_image_embeds.append(image_embed.cpu())
            all_text_embeds.append(text_embed.cpu())

    all_image_embeds = torch.cat(all_image_embeds, dim=0)  # [N, D]
    all_text_embeds = torch.cat(all_text_embeds, dim=0)  # [N, D]

    # 归一化
    all_image_embeds = F.normalize(all_image_embeds, dim=1)
    all_text_embeds = F.normalize(all_text_embeds, dim=1)

    # 文本 -> 图像检索
    sim_matrix = torch.matmul(all_text_embeds, all_image_embeds.T)  # [N, N]
    txt2img_ranks = torch.argsort(sim_matrix, dim=1, descending=True)

    # 图像 -> 文本检索
    sim_matrix_T = sim_matrix.T  # [N, N]
    img2txt_ranks = torch.argsort(sim_matrix_T, dim=1, descending=True)

    def recall_at_k(ranks, topk):
        recalls = []
        for k in topk:
            match = [i in ranks[i][:k] for i in range(len(ranks))]
            recalls.append(np.mean(match))
        return recalls

    r_txt2img = recall_at_k(txt2img_ranks, topk)
    r_img2txt = recall_at_k(img2txt_ranks, topk)

    print("\n📈 Text → Image Retrieval:")
    for i, k in enumerate(topk):
        print(f"Recall@{k}: {r_txt2img[i] * 100:.2f}%")

    print("\n📈 Image → Text Retrieval:")
    for i, k in enumerate(topk):
        print(f"Recall@{k}: {r_img2txt[i] * 100:.2f}%")

    return r_txt2img, r_img2txt


def main():
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 文件路径，根据实际调整
    token_file = "Flickr8k/captions.txt"  # 总的 captions 文件，用于构建词表
    train_token_file = "Flickr8k/train_captions.txt"  # 训练集，格式： image,caption
    val_token_file = "Flickr8k/val_captions.txt"  # 验证集
    test_token_file = "Flickr8k/test_captions.txt"  # 测试集

    # 读取所有 caption 用于构建总词表（假设以逗号分隔，如果不是，请修改 split 参数）
    with open(token_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    captions = [line.strip().split(",")[1] for line in lines if line.strip()]

    # 构建统一的 tokenizer
    tokenizer = SimpleTokenizer(captions, min_freq=1)
    vocab_size = len(tokenizer)
    print(f"Vocabulary size: {vocab_size}")

    # 构建数据集与 DataLoader：训练集、验证集、测试集
    train_dataset = Flickr8kDataset(
        root_dir="Flickr8k/images",  # 图片所在目录
        captions_file=train_token_file,  # 训练集 captions 文件，格式： image,caption
        tokenizer=tokenizer,
    )
    val_dataset = Flickr8kDataset(
        root_dir="Flickr8k/images",
        captions_file=val_token_file,  # 验证集 captions 文件
        tokenizer=tokenizer,
    )
    test_dataset = Flickr8kDataset(
        root_dir="Flickr8k/images",
        captions_file=test_token_file,  # 测试集 captions 文件
        tokenizer=tokenizer,
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=4, drop_last=True
    )
    # 为保证评估稳定，每个 batch 使用 batch_size=1
    val_dataloader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False
    )

    # 构造模型（设定 embed_dim=256）
    embed_dim = 256
    img_encoder = ImageEncoder(embed_dim=embed_dim).to(device)
    txt_encoder = TextEncoder(vocab_size, embed_dim=embed_dim).to(device)

    # 优化器
    optimizer = torch.optim.Adam(
        list(img_encoder.parameters()) + list(txt_encoder.parameters()), lr=1e-4
    )

    best_val_loss = float("inf")
    best_val_recall = 0.0  # 用于记录最佳验证集 Recall@1
    epochs = 40
    for epoch in range(epochs):
        img_encoder.train()
        txt_encoder.train()
        epoch_loss = 0.0

        for images, captions_ids in train_dataloader:
            images = images.to(device)
            captions_ids = captions_ids.to(device)

            image_embeds = img_encoder(images)  # [batch, embed_dim]
            text_embeds = txt_encoder(captions_ids)  # [batch, embed_dim]
            loss = contrastive_loss(image_embeds, text_embeds)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_dataloader)

        # 在验证集上评估
        val_loss = evaluate(img_encoder, txt_encoder, val_dataloader, device)
        val_recall, _ = evaluate_top_k(img_encoder, txt_encoder, val_dataloader, device)
        val_recall_at_1 = val_recall[0]  # 获取 Recall@1

        print(
            f"Epoch [{epoch + 1}/{epochs}]: Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Recall@1: {val_recall_at_1 * 100:.2f}%"
        )

        # 如果验证集 Recall@1 提升，则保存最佳模型
        if val_recall_at_1 > best_val_recall:
            best_val_recall = val_recall_at_1
            checkpoint = {
                "epoch": epoch + 1,
                "img_encoder_state_dict": img_encoder.state_dict(),
                "txt_encoder_state_dict": txt_encoder.state_dict(),
                "tokenizer_vocab": tokenizer.word2idx,
                "best_val_recall": best_val_recall,
            }
            torch.save(checkpoint, "best_clip_model.pth")
            print(
                f"    > Best model updated at epoch {epoch + 1} with Recall@1: {best_val_recall * 100:.2f}%"
            )

    # 训练完成，最终在测试集上评估
    final_test_loss = evaluate(img_encoder, txt_encoder, test_dataloader, device)
    print(f"Final Test Loss: {final_test_loss:.4f}")

    # 最终评估 top-k 准确率
    r_txt2img, r_img2txt = evaluate_top_k(
        img_encoder, txt_encoder, test_dataloader, device
    )
    print("\n📈 Final Test Recall@k:")
    for i, k in enumerate([1, 5, 10]):
        print(f"Text → Image Recall@{k}: {r_txt2img[i] * 100:.2f}%")
        print(f"Image → Text Recall@{k}: {r_img2txt[i] * 100:.2f}%")


if __name__ == "__main__":
    main()
