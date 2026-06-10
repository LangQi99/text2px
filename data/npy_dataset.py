import numpy as np
import torch
from torch.utils.data import Dataset


class NpyRGBA16Dataset(Dataset):
    """Reference-style packed NumPy dataset: token ids + 16x16 RGBA arrays."""

    def __init__(self, token_ids, img_data, mean=None, std=None):
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        img = img_data.astype(np.float32)
        if mean is None:
            mean = img.mean(axis=(0, 1, 2))
        if std is None:
            std = img.std(axis=(0, 1, 2))
        std = np.asarray(std, dtype=np.float32)
        std[std == 0] = 1.0

        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = std
        img = (img - self.mean) / self.std
        self.img_data = torch.tensor(img, dtype=torch.float32)

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, idx):
        return self.token_ids[idx], self.img_data[idx]


def denormalize_rgba(tensor, mean, std):
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    arr = tensor.permute(0, 2, 3, 1).detach().cpu().numpy()
    arr = arr * np.asarray(std, dtype=np.float32) + np.asarray(mean, dtype=np.float32)
    return np.clip(arr, 0, 255).astype(np.uint8)
