import csv
import os

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from transformers import BertTokenizer, CLIPImageProcessor

from utils import SimpleTokenizer

# 针对 ResNet + Transformer 的数据集类


class Flickr8kDataset(Dataset):
    def __init__(
        self,
        root_dir,
        captions_file,
        tokenizer: SimpleTokenizer,
        transform=None,
        max_len=32,
    ):
        self.root_dir = root_dir
        self.transform = transform or T.Compose([T.Resize((224, 224)), T.ToTensor()])
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pairs = self._load_pairs(captions_file)

    def _load_pairs(self, captions_file):
        pairs = []
        with open(captions_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                image_filename = row[0].strip()
                caption = row[1].strip()
                pairs.append((image_filename, caption))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        filename, caption = self.pairs[idx]
        image_path = os.path.join(self.root_dir, filename)
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        caption_ids = self.tokenizer.encode(caption)
        return image, torch.tensor(caption_ids)


# 针对 ResNet + Bert 模型的数据集类


class Flickr8kDatasetV2(Dataset):
    def __init__(
        self,
        root_dir,
        captions_file,
        bert_tokenizer: BertTokenizer,
        transform=None,
        max_len=32,
    ):
        self.root_dir = root_dir
        self.transform = transform or T.Compose([T.Resize((224, 224)), T.ToTensor()])
        self.tokenizer = bert_tokenizer
        self.max_len = max_len
        self.pairs = self._load_pairs(captions_file)

    def _load_pairs(self, captions_file):
        pairs = []
        with open(captions_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                image_filename = row[0].strip()
                caption = row[1].strip()
                pairs.append((image_filename, caption))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        filename, caption = self.pairs[idx]
        image_path = os.path.join(self.root_dir, filename)
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)  # [3, 224, 224]

        encodings = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        caption_ids = encodings["input_ids"].squeeze(0)  # [max_len]
        attn_mask = encodings["attention_mask"].squeeze(0)  # [max_len]

        return image, caption_ids, attn_mask


# 针对 CLIP-ViT + BERT 模型的数据集类
class Flickr8kDatasetV3(Dataset):
    def __init__(
        self,
        root_dir,
        captions_file,
        bert_tokenizer: BertTokenizer,
        clip_processor: CLIPImageProcessor,
        max_len: int = 32,
    ):
        self.root_dir = root_dir
        self.tokenizer = bert_tokenizer
        self.processor = clip_processor
        self.max_len = max_len
        self.pairs = self._load_pairs(captions_file)

    def _load_pairs(self, captions_file):
        pairs = []
        with open(captions_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                image_filename = row[0].strip()
                caption = row[1].strip()
                pairs.append((image_filename, caption))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        filename, caption = self.pairs[idx]
        image_path = os.path.join(self.root_dir, filename)
        image = Image.open(image_path).convert("RGB")

        enc_img = self.processor(images=image, return_tensors="pt")
        pixel_values = enc_img["pixel_values"].squeeze(0)  # [3, 224, 224]

        enc_txt = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        caption_ids = enc_txt["input_ids"].squeeze(0)  # [max_len]
        attn_mask = enc_txt["attention_mask"].squeeze(0)  # [max_len]

        return pixel_values, caption_ids, attn_mask
