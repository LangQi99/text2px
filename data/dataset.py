"""
Dataset for Minecraft item textures.
Expects:
  data/minecraft_items/
    ├── images/       # 16x16 PNG files (RGBA)
    └── labels.json   # {"filename": "description", ...}
"""
import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class MinecraftItemDataset(Dataset):
    def __init__(self, root_dir, tokenizer, image_size=16, augment=False):
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.augment = augment

        labels_path = os.path.join(root_dir, 'labels.json')
        with open(labels_path) as f:
            self.labels = json.load(f)

        self.items = list(self.labels.items())

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        filename, description = self.items[idx]
        img_path = os.path.join(self.root_dir, 'images', filename)

        img = Image.open(img_path).convert('RGBA')
        img = img.resize((self.image_size, self.image_size), Image.NEAREST)

        img_array = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        img_tensor = img_tensor * 2.0 - 1.0  # normalize to [-1, 1]

        if self.augment:
            if torch.rand(1).item() > 0.5:
                img_tensor = img_tensor.flip(-1)

        tokens = torch.tensor(self.tokenizer.encode(description), dtype=torch.long)
        mask = torch.tensor(self.tokenizer.get_mask(description), dtype=torch.bool)

        return img_tensor, tokens, mask
