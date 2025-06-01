import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import BertTokenizer, CLIPImageProcessor

from data_loader import Flickr8kDatasetV3
from models.image_encoder import PretrainedCLIPViT
from models.text_encoder import PretrainedBert


def main():
    query = "Young girl hanging on a vine ."
    topk = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(
        "exp/ckpts/pre-clipvit-bert.pth",
        map_location=device,
        weights_only=False,
    )
    bert_tokenizer = BertTokenizer.from_pretrained(
        "cache/bert-base-uncased", local_files_only=True
    )
    clip_processor = CLIPImageProcessor.from_pretrained(
        "cache/clip-vit-base-patch16", local_files_only=True
    )

    img_encoder = PretrainedCLIPViT(
        embed_dim=256,
        model_name="cache/clip-vit-base-patch16",
        freeze_backbone=True,
        num_unfrozen_layers=1,
    ).to(device)

    txt_encoder = PretrainedBert(
        embed_dim=256,
        model_name="cache/bert-base-uncased",  # 若离线，请指向本地缓存
        freeze_backbone=True,
        num_unfrozen_layers=4,
    ).to(device)

    img_encoder.load_state_dict(ckpt["img_encoder_state"])
    txt_encoder.load_state_dict(ckpt["txt_encoder_state"])
    img_encoder.eval()
    txt_encoder.eval()

    dataset = Flickr8kDatasetV3(
        "Flickr8k/images",
        "Flickr8k/val_captions.txt",
        bert_tokenizer=bert_tokenizer,
        clip_processor=clip_processor,
    )

    # all_embeds = []
    # with torch.no_grad():
    #     for fn, _ in dataset.pairs:
    #         img = Image.open(os.path.join(dataset.root_dir, fn)).convert("RGB")
    #         x = dataset.transform(img).unsqueeze(0).to(device)
    #         all_embeds.append(img_encoder(x).cpu())
    # image_embeds = torch.cat(all_embeds, dim=0)
    # image_embeds = F.normalize(image_embeds, dim=1)

    # 对图片去重
    all_fns = [fn for fn, _ in dataset.pairs]
    unique_fns = list(dict.fromkeys(all_fns))

    all_embeds = []
    with torch.no_grad():
        for fn in unique_fns:
            img = Image.open(os.path.join(dataset.root_dir, fn)).convert("RGB")
            enc = clip_processor(images=img, return_tensors="pt")
            pixel_values = enc["pixel_values"].squeeze(0).to(device)  # [3,224,224]
            pixel_values = pixel_values.unsqueeze(0)  # [1,3,224,224]

            embed_i = img_encoder(pixel_values)  # [1,256]
            all_embeds.append(embed_i.cpu())

    image_embeds = torch.cat(all_embeds, dim=0)  # [N_unique, D]
    image_embeds = F.normalize(image_embeds, dim=1)

    encoding = bert_tokenizer(
        query,
        padding="max_length",
        truncation=True,
        max_length=32,  # 与训练时 max_len 一致
        return_tensors="pt",
    ).to(device)
    input_ids = encoding["input_ids"]  # [1, 32]
    attn_mask = encoding["attention_mask"]  # [1, 32]

    with torch.no_grad():
        text_embeds = txt_encoder(input_ids, attn_mask).cpu()
    text_embeds = F.normalize(text_embeds, dim=1)

    sims = (text_embeds @ image_embeds.t()).squeeze(0)
    vals, idxs = sims.topk(topk, largest=True)

    # 绘制图像
    fig, axes = plt.subplots(1, topk, figsize=(3 * topk, 3))
    fig.suptitle(f"Caption: {query}", fontsize=12)
    for i, (score, idx) in enumerate(zip(vals, idxs), start=1):
        fn = unique_fns[idx]
        img = Image.open(os.path.join(dataset.root_dir, fn)).convert("RGB")
        axes[i - 1].imshow(img)
        axes[i - 1].axis("off")
        axes[i - 1].set_title(f"Rank{i}: {score:.2f}")
    plt.tight_layout()
    plt.savefig("exp/images/visualization_final.png")
    plt.show()


if __name__ == "__main__":
    main()
