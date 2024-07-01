import os

import numpy as np
import pytorch_lightning as pl
import scipy.io
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import transforms


class CrackForest(pl.LightningDataModule):
    def __init__(
        self, data_dir="./data/CrackForest-dataset", batch_size=4, num_workers=19
    ):
        super().__init__()
        self.cf_train = None
        self.cf_val = None
        self.cf_test = None
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = CrackForestDataset()
        self.cf_train, self.cf_val, self.cf_test = random_split(
            self.dataset, [100, 10, 8]
        )

    def train_dataloader(self):
        return DataLoader(self.cf_train, shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.cf_val, shuffle=False, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.cf_test, shuffle=False, batch_size=self.batch_size)


class CrackForestDataset(Dataset):
    def __init__(self, data_dir="./data/CrackForest-dataset"):
        self.data_dir = data_dir

        # len(dataset) == 118
        self.number_strings = [i for i in range(1, 119)]
        self.images = [
            Image.open(data_dir + "/img/" + f"{i}.png").convert("RGB")
            for i in self.number_strings
        ]
        self.labels = [
            Image.open(data_dir + "/gt/" + f"{i}.png").convert("L")
            for i in self.number_strings
        ]

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.number_strings)

    def __getitem__(self, idx):
        image = np.array(self.images[idx])
        label = np.expand_dims(np.array(self.labels[idx]), axis=2) / 255
        image = self.transform(image)
        label = self.transform(label)
        return image, label
