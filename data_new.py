import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image


class CrackForestDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


class CrackForestDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir="./data/CrackForest-dataset",
        batch_size=8,
        num_workers=4,
        transform=None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform

    def setup(self, stage=None):
        self.train_dataset = CrackForestDataset(
            image_dir=os.path.join(self.data_dir, "image"),
            mask_dir=os.path.join(self.data_dir, "groundTruth"),
            transform=self.transform,
        )
        # Split the dataset if needed for train/val/test

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        # Implement this method if you have a validation set
        pass

    def test_dataloader(self):
        # Implement this method if you have a test set
        pass
