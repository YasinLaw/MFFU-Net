import torch.nn as nn
import lightning as pl
from schedulefree import AdamWScheduleFree

from modules import DoubleConv, Down, Up


class UNet(pl.LightningModule):
    def __init__(self, num_classes=1, in_channel=3, bilinear=True):
        super().__init__()
        self.bilinear = bilinear
        self.num_classes = num_classes
        self.layer1 = DoubleConv(in_channel, 64)
        self.layer2 = Down(64, 128)
        self.layer3 = Down(128, 256)
        self.layer4 = Down(256, 512)
        self.factor = 2 if self.bilinear else 1
        self.layer5 = Down(512, 1024 // self.factor)
        self.layer6 = Up(1024, 512 // self.factor, bilinear=self.bilinear)
        self.layer7 = Up(512, 256 // self.factor, bilinear=self.bilinear)
        self.layer8 = Up(256, 128 // self.factor, bilinear=self.bilinear)
        self.layer9 = Up(128, 64, bilinear=self.bilinear)

        self.layer10 = nn.Conv2d(64, self.num_classes, kernel_size=1)

        self.criterion = (
            nn.CrossEntropyLoss() if num_classes > 1 else nn.BCEWithLogitsLoss()
        )

        self.save_hyperparameters()

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)

        x6, f1 = self.layer6(x5, x4)
        x6, f2 = self.layer7(x6, x3)
        x6, f3 = self.layer8(x6, x2)
        x6, f4 = self.layer9(x6, x1)

        # x = f1 + f2 + f3 + f4
        return self.layer10(x6)

    def shared_step(self, batch):
        image, label = batch
        target = self(image)
        loss = self.criterion(target, label)
        self.log_dict({f"{'train' if self.training else 'val'}/loss": loss})
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch)

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch)

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch)

    def configure_optimizers(self):
        return AdamWScheduleFree(self.parameters(), lr=1e-5)

    def on_train_epoch_start(self):
        optimizer = self.optimizers().optimizer
        optimizer.train()

    def on_validation_epoch_start(self):
        optimizer = self.optimizers().optimizer
        optimizer.eval()

    def on_test_epoch_start(self):
        optimizer = self.optimizers().optimizer
        optimizer.eval()
