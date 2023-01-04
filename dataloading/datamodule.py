from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.transforms
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, random_split

from dataloading.datagen import CustomDataGen


class MyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        train_val_ratio,
        base_path,
        num_workers=0,
        img_size=None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.train_val_ratio = train_val_ratio
        self.num_workers = num_workers
        self.base_path = Path(base_path)
        self.img_size = img_size

    def prepare_data(self):
        f = self.base_path / "metadata.csv"
        self.df = pd.read_csv(f)
        self.filenames = self.df["filename"]
        self.labels = self.df["label"]

        assert len(self.filenames) == len(self.labels)

        self.num_classes = len(set(self.labels))
        self.df = pd.DataFrame({"filename": self.filenames, "label": self.labels})

    def get_preprocessing_transform(self):
        transforms = nn.Sequential(
            torchvision.transforms.Normalize(self.mean, self.std),
            torchvision.transforms.Resize(self.img_size)
            if self.img_size
            else nn.Identity(),
        )
        return transforms

    def setup(self, stage=None):
        num_subjects = len(self.df)
        num_train_subjects = int(round(num_subjects * self.train_val_ratio))
        num_val_subjects = num_subjects - num_train_subjects
        splits = num_train_subjects, num_val_subjects
        self.train_subjects, self.val_subjects = random_split(
            self.df, splits, generator=torch.Generator().manual_seed(42)  # type: ignore
        )
        self.calc_mean_std()

    def train_dataloader(self):
        return DataLoader(
            CustomDataGen(
                self.train_subjects,
                self.base_path,
                transform=self.get_preprocessing_transform(),
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            CustomDataGen(
                self.val_subjects,
                self.base_path,
                transform=self.get_preprocessing_transform(),
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def calc_mean_std(self):
        pixel_sum, pixel_squared_sum, num_pixels = (
            torch.tensor([0.0, 0.0, 0.0]),
            torch.tensor([0.0, 0.0, 0.0]),
            0,
        )

        print("Calculating mean and std...")
        for i, row in self.df.iterrows():
            filename = row["filename"]
            filepath = next(self.base_path.glob(f"**/{filename}"))
            img = Image.open(filepath)
            img = TF.to_tensor(img)
            pixel_sum += img.sum(dim=(1, 2))
            pixel_squared_sum += (img**2).sum(dim=(1, 2))
            num_pixels += img.shape[1] * img.shape[2]

        self.mean = pixel_sum / num_pixels
        self.std = torch.sqrt(pixel_squared_sum / num_pixels - self.mean**2)
        print("Done calculating mean and std.")
