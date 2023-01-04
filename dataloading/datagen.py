from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataset import Subset


class CustomDataGen(Dataset):
    def __init__(
        self,
        df: Subset[pd.DataFrame],
        base_path: Path,
        transform=None,
    ):
        self.base_path = base_path
        self.transform = transform

        subset = np.take(df.dataset, df.indices, axis=0) # type: ignore
        self.img_paths = subset["filename"].to_numpy()
        self.labels = subset["label"].to_numpy()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]

        # the image might be in any subfolder of the base path
        # so we need to search for it first
        img_path = next(self.base_path.glob(f"**/{img_path}"))
        img = Image.open(img_path)
        img = TF.to_tensor(img)
        if self.transform:
            img = self.transform(img)

        return img, self.labels[idx]
