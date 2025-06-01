import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image

from data_loader import Flickr8kDataset
from models.image_encoder import ImageEncoder
from models.text_encoder import TextEncoder
from utils import SimpleTokenizer


def main():
    query = "A man riding a bike wearing flannel shirt , plaid pants and a blue backpack while riding on a street near buildings and parked cars ."
    topk = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(
        "exp/chpts/best_clip_model_transformer3.pth",
        map_location=device,
        weights_only=False,
    )
    vocab_dict = ckpt.get("tokenizer_vocab")
    tokenizer = SimpleTokenizer([], min_freq=1)
    if vocab_dict is not None:
        tokenizer.word2idx = vocab_dict
    pad_id = tokenizer.word2idx.get("<PAD>", 0)

    img_encoder = ImageEncoder(embed_dim=256).to(device)
    txt_encoder = TextEncoder(
        len(tokenizer), embed_dim=256, encoder_type="transformer"
    ).to(device)
    img_encoder.load_state_dict(ckpt["img_encoder_state_dict"])
    txt_encoder.load_state_dict(ckpt["txt_encoder_state_dict"])
    img_encoder.eval()
    txt_encoder.eval()

    dataset = Flickr8kDataset("Flickr8k/images", "Flickr8k/val_captions.txt", tokenizer)

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
            x = dataset.transform(img).unsqueeze(0).to(device)
            all_embeds.append(img_encoder(x).cpu())

    image_embeds = torch.cat(all_embeds, dim=0)  # [N_unique, D]
    image_embeds = F.normalize(image_embeds, dim=1)

    token_ids = tokenizer.encode(query, max_len=dataset.max_len)
    if len(token_ids) < dataset.max_len:
        token_ids += [pad_id] * (dataset.max_len - len(token_ids))
    cap = torch.tensor(token_ids, device=device).unsqueeze(0)
    with torch.no_grad():
        text_emb = txt_encoder(cap).cpu()
    text_emb = F.normalize(text_emb, dim=1)

    sims = (text_emb @ image_embeds.t()).squeeze(0)
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
    plt.savefig("exp/images/visualization.png")
    plt.show()


if __name__ == "__main__":
    main()
