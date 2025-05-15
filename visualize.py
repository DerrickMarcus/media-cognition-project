import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image

from data_loader import Flickr8kDataset
from models.image_encoder import ImageEncoder
from models.text_encoder import TextEncoder
from utils import SimpleTokenizer


def visualize_text_to_image(
    query_caption: str,
    img_encoder: torch.nn.Module,
    txt_encoder: torch.nn.Module,
    dataset: Flickr8kDataset,
    tokenizer: SimpleTokenizer,
    device: torch.device,
    topk: int = 5,
    image_embeds_cache: torch.Tensor = None,
):
    img_encoder.eval()
    txt_encoder.eval()

    # 1. 计算或加载所有图片的嵌入
    if image_embeds_cache is None:
        all_embeds = []
        with torch.no_grad():
            for fn, _ in dataset.pairs:
                img = Image.open(os.path.join(dataset.root_dir, fn)).convert("RGB")
                x = dataset.transform(img).unsqueeze(0).to(device)
                all_embeds.append(img_encoder(x).cpu())
        image_embeds = torch.cat(all_embeds, dim=0)  # [N, D]
    else:
        image_embeds = image_embeds_cache

    image_embeds = F.normalize(image_embeds, dim=1)

    # 2. 编码查询文本
    tok_ids = tokenizer.encode(query_caption)
    tok_ids = tok_ids[: dataset.max_len] + [tokenizer.pad_token_id] * max(
        0, dataset.max_len - len(tok_ids)
    )
    cap = torch.tensor(tok_ids, device=device).unsqueeze(0)
    with torch.no_grad():
        txt_embed = txt_encoder(cap).cpu()
    txt_embed = F.normalize(txt_embed, dim=1)  # [1, D]

    # 3. 计算相似度，取 Top-K
    sims = (txt_embed @ image_embeds.t()).squeeze(0)  # [N]
    top_vals, top_idxs = sims.topk(topk, largest=True)

    # 4. 可视化
    fig, axes = plt.subplots(1, topk, figsize=(3 * topk, 3))
    fig.suptitle(f'Query: "{query_caption}"', fontsize=14)
    for rank, idx in enumerate(top_idxs.tolist(), 1):
        fn, _ = dataset.pairs[idx]
        img = Image.open(os.path.join(dataset.root_dir, fn)).convert("RGB")
        axes[rank - 1].imshow(img)
        axes[rank - 1].set_title(f"#{rank} Score={top_vals[rank - 1]:.2f}")
        axes[rank - 1].axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # ---- 配置 ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 构建 tokenizer（需与训练时保持一致）
    captions = [...]  # 所有原始 caption 列表
    tokenizer = SimpleTokenizer(captions, min_freq=1)
    vocab_size = len(tokenizer)

    # 加载模型
    ckpt = torch.load("chpts/best_clip_model.pth", map_location=device)
    img_enc = ImageEncoder(embed_dim=256).to(device)
    txt_enc = TextEncoder(vocab_size, embed_dim=256, encoder_type="transformer").to(
        device
    )
    img_enc.load_state_dict(ckpt["img_encoder_state_dict"])
    txt_enc.load_state_dict(ckpt["txt_encoder_state_dict"])

    # 构建 dataset
    dataset = Flickr8kDataset(
        root_dir="Flickr8k/images",
        captions_file="Flickr8k/val_captions.txt",
        tokenizer=tokenizer,
    )

    # 可选：预先计算图片嵌入并缓存
    # image_embeds_cache = ...
    image_embeds_cache = None

    # 输入一个句子，查看 Top-5 检索结果
    query = "A man riding a bicycle on a country road."
    visualize_text_to_image(
        query_caption=query,
        img_encoder=img_enc,
        txt_encoder=txt_enc,
        dataset=dataset,
        tokenizer=tokenizer,
        device=device,
        topk=5,
        image_embeds_cache=image_embeds_cache,
    )
