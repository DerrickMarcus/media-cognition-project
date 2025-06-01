import math
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import BertTokenizer, get_cosine_schedule_with_warmup

from data_loader import Flickr8kDatasetV2

# from loss_learned import ContrastiveLoss
from loss import contrastive_loss
from models.image_encoder import PretrainedResNet
from models.text_encoder import PretrainedBert


# 供同学们参考
def evaluate_top_k(img_encoder, txt_encoder, dataloader, device, topk=(1, 5, 10)):
    img_encoder.eval()
    txt_encoder.eval()

    all_image_embeds = []
    all_text_embeds = []

    with torch.no_grad():
        for images, captions_ids, attn_mask in tqdm(
            dataloader, desc="Extracting embeddings"
        ):
            images = images.to(device)
            captions_ids = captions_ids.to(device)
            attn_mask = attn_mask.to(device)

            image_embed = img_encoder(images)  # [1, dim]
            text_embed = txt_encoder(captions_ids, attn_mask)  # [1, dim]

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

    BATCH_TRAIN = 128  # 物理 batch
    BATCH_EVAL = 512
    ACCUM_STEPS = 1  # =1 表示“不累计”；以后显存紧张时可设 2
    EPOCHS = 50
    EMBED_DIM = 256
    BASE_LR = 5e-5  # 配合 scheduler 的初始 LR

    # 文件路径，根据实际调整
    # token_file = "Flickr8k/captions.txt"  # 总的 captions 文件，用于构建词表
    train_token_file = "Flickr8k/train_captions.txt"  # 训练集，格式： image,caption
    val_token_file = "Flickr8k/val_captions.txt"  # 验证集
    test_token_file = "Flickr8k/test_captions.txt"  # 测试集

    # 读取所有 caption 用于构建总词表（假设以 tab 分隔，如果不是，请修改 split 参数）
    # with open(token_file, "r", encoding="utf-8") as f:
    #     captions = [line.strip().split(",")[1] for line in f if line.strip()]

    # 构建统一的 tokenizer
    bert_tokenizer = BertTokenizer.from_pretrained(
        "cache/bert-base-uncased",
        local_files_only=True,
    )
    vocab_size = len(bert_tokenizer)
    print(f"Vocabulary size: {vocab_size}")

    # 图像变换
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # 构建数据集与 DataLoader
    # captions 文件格式： image<TAB>caption
    train_dataloader = DataLoader(
        Flickr8kDatasetV2(
            root_dir="Flickr8k/images",
            captions_file=train_token_file,
            bert_tokenizer=bert_tokenizer,
            # transform=train_tf,
        ),
        batch_size=BATCH_TRAIN,
        shuffle=True,
        num_workers=12,
        pin_memory=True,
        drop_last=True,
    )
    val_dataloader = DataLoader(
        Flickr8kDatasetV2(
            root_dir="Flickr8k/images",
            captions_file=val_token_file,
            bert_tokenizer=bert_tokenizer,
            # transform=eval_tf,
        ),
        batch_size=BATCH_EVAL,
        shuffle=False,
        num_workers=12,
        pin_memory=True,
        drop_last=False,
    )
    test_dataloader = DataLoader(
        Flickr8kDatasetV2(
            root_dir="Flickr8k/images",
            captions_file=test_token_file,
            bert_tokenizer=bert_tokenizer,
            # transform=eval_tf,
        ),
        batch_size=BATCH_EVAL,
        shuffle=False,
        num_workers=12,
        pin_memory=True,
        drop_last=False,
    )

    # 构造模型（设定 embed_dim=256）
    # img_encoder = ImageEncoder(embed_dim=EMBED_DIM, model_name="resnet50").to(device)
    img_encoder = PretrainedResNet(
        embed_dim=EMBED_DIM,
        model_name="resnet50",
        freeze_backbone=True,
    ).to(device)
    # txt_encoder = TextEncoder(
    #     vocab_size, embed_dim=EMBED_DIM, model_name="transformer"
    # ).to(device)
    txt_encoder = PretrainedBert(
        embed_dim=EMBED_DIM,
        model_name="cache/bert-base-uncased",
        freeze_backbone=True,
        num_unfrozen_layers=4,
    ).to(device)
    # contra_loss = ContrastiveLoss(init_temperature=0.07).to(device)

    # 优化器和调度器
    optimizer = torch.optim.AdamW(
        [
            {"params": img_encoder.parameters(), "lr": BASE_LR},
            {"params": txt_encoder.parameters(), "lr": BASE_LR},
            # {"params": contra_loss.parameters(), "lr": 1e-4},
        ],
        weight_decay=1e-4,
    )
    total_iters = EPOCHS * math.ceil(len(train_dataloader) / ACCUM_STEPS)
    warm_iters = int(0.1 * total_iters)  # 5% 预热
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warm_iters,  # 先线性升到基准 LR
        num_training_steps=total_iters,  # 之后余弦退火降到 0
    )
    scaler = GradScaler(device="cuda")

    # 使用之前训练好的 ResNet50 模型参数
    ckpt = torch.load(
        "exp/ckpts/pre-resnet50-trans.pth", map_location=device, weights_only=False
    )
    # img_encoder.load_state_dict(ckpt["img_encoder_state"])
    txt_encoder.load_state_dict(ckpt["txt_encoder_state"])

    pathlib.Path("exp/ckpts").mkdir(parents=True, exist_ok=True)
    pathlib.Path("exp/images").mkdir(parents=True, exist_ok=True)
    best_ckpt = "exp/ckpts/pre-resnet50-bert.pth"
    loss_curve = "exp/images/pre-resnet50-bert.png"
    best_recall_i2t, best_recall_t2i = 0.0, 0.0
    train_losses, valid_losses = [], []
    recall_i2t, recall_t2i = [], []
    topk = (1, 5, 10)
    for epoch in range(EPOCHS):
        # train
        img_encoder.train()
        txt_encoder.train()
        # contra_loss.train()

        epoch_loss = 0.0
        pbar = tqdm(
            train_dataloader, desc=f"Train epoch {epoch + 1}/{EPOCHS}", unit="batch"
        )

        for step, (images, captions_ids, attn_mask) in enumerate(pbar):
            images = images.to(device)
            captions_ids = captions_ids.to(device)
            attn_mask = attn_mask.to(device)

            image_embeds = img_encoder(images)  # [batch, embed_dim]
            text_embeds = txt_encoder(captions_ids, attn_mask)  # [batch, embed_dim]

            with autocast(device_type="cuda"):
                loss = contrastive_loss(image_embeds, text_embeds) / ACCUM_STEPS
                # loss = contra_loss(image_embeds, text_embeds) / ACCUM_STEPS
            # loss.backward()
            scaler.scale(loss).backward()

            if (step + 1) % ACCUM_STEPS == 0:
                # optimizer.step()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()  # 更新学习率

            epoch_loss += loss.item() * ACCUM_STEPS
            pbar.set_postfix(loss=epoch_loss / (step + 1))

        avg_train_loss = epoch_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        # validation
        img_encoder.eval()
        txt_encoder.eval()
        # contra_loss.eval()

        total_val_loss = 0.0
        pbar = tqdm(
            val_dataloader, desc=f"Validation epoch {epoch + 1}/{EPOCHS}", unit="batch"
        )
        with torch.no_grad(), autocast(device_type="cuda"):
            for images, captions_ids, attn_mask in pbar:
                images = images.to(device)
                captions_ids = captions_ids.to(device)
                attn_mask = attn_mask.to(device)

                image_embeds = img_encoder(images)
                text_embeds = txt_encoder(captions_ids, attn_mask)
                val_loss = contrastive_loss(image_embeds, text_embeds)
                # val_loss = contra_loss(image_embeds, text_embeds)
                total_val_loss += val_loss.item()
                pbar.set_postfix(loss=val_loss.item())

        avg_val_loss = total_val_loss / len(val_dataloader)
        valid_losses.append(avg_val_loss)

        print(
            f"Epoch {epoch + 1}/{EPOCHS}: Train loss: {avg_train_loss:.4f}, Val loss: {avg_val_loss:.4f}"
        )

        # 以 Recall@1 为评价指标
        r_txt2img, r_img2txt = evaluate_top_k(
            img_encoder, txt_encoder, val_dataloader, device, topk=topk
        )
        if r_img2txt[0] > best_recall_i2t and r_txt2img[0] > best_recall_t2i:
            best_recall_i2t, best_recall_t2i = r_img2txt[0], r_txt2img[0]
            torch.save(
                {
                    "epoch": epoch + 1,
                    "img_encoder_state": img_encoder.state_dict(),
                    "txt_encoder_state": txt_encoder.state_dict(),
                    # "contra_loss_state": contra_loss.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    # "vocab": bert_tokenizer.word2idx,
                    "recall_i2t": best_recall_i2t,
                    "recall_t2i": best_recall_t2i,
                },
                best_ckpt,
            )
            print(f"    > Best model updated at epoch {epoch + 1}")
        recall_i2t.append(r_img2txt)
        recall_t2i.append(r_txt2img)

    # 绘制 loss 曲线和 Recall 曲线
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 3, 1)
    plt.plot(range(1, EPOCHS + 1), train_losses, label="Train Loss")
    plt.plot(range(1, EPOCHS + 1), valid_losses, label="Valid Loss")
    plt.title("Train & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    for i, k in enumerate(topk):
        plt.plot(
            range(1, EPOCHS + 1), [r[i] * 100 for r in recall_i2t], label=f"Recall@{k}"
        )
    plt.title("Image to Text Recall")
    plt.xlabel("Epoch")
    plt.ylabel("Recall (%)")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    for i, k in enumerate(topk):
        plt.plot(
            range(1, EPOCHS + 1), [r[i] * 100 for r in recall_t2i], label=f"Recall@{k}"
        )
    plt.title("Text to Image Recall")
    plt.xlabel("Epoch")
    plt.ylabel("Recall (%)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(loss_curve)
    plt.show()

    # test
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    img_encoder.load_state_dict(ckpt["img_encoder_state"])
    txt_encoder.load_state_dict(ckpt["txt_encoder_state"])
    # contra_loss.load_state_dict(ckpt["contra_loss_state"])
    print(
        f"Loaded best model from epoch {ckpt['epoch']}\n"
        f"Best Recall@1 of image to text = {ckpt['recall_i2t']:.4f}\n"
        f"Best Recall@1 of text to image = {ckpt['recall_t2i']:.4f}\n"
    )

    img_encoder.eval()
    txt_encoder.eval()
    # contra_loss.eval()

    total_test_loss = 0.0
    pbar = tqdm(test_dataloader, desc="Final test", unit="batch")

    with torch.no_grad(), autocast(device_type="cuda"):
        for images, captions_ids, attn_mask in pbar:
            images = images.to(device)
            captions_ids = captions_ids.to(device)
            attn_mask = attn_mask.to(device)

            image_embeds = img_encoder(images)
            text_embeds = txt_encoder(captions_ids, attn_mask)
            test_loss = contrastive_loss(image_embeds, text_embeds)
            # test_loss = contra_loss(image_embeds, text_embeds)

            total_test_loss += test_loss.item()
            pbar.set_postfix(loss=test_loss.item())

    avg_test_loss = total_test_loss / len(test_dataloader)
    print(f"Final test loss: {avg_test_loss:.4f}")

    evaluate_top_k(img_encoder, txt_encoder, test_dataloader, device, topk=topk)


if __name__ == "__main__":
    main()
