import torch.nn as nn
import pytorch_lightning as pl
import torch.optim
from schedulefree import AdamWScheduleFree

from modules import Down, Up, ResBlock, AttnBlock


class UNet(pl.LightningModule):
    def __init__(self, num_classes=1, in_channel=3, bicubic=True):
        super().__init__()
        self.bicubic = bicubic
        self.num_classes = num_classes
        self.layer1 = nn.Sequential(ResBlock(in_channel, 64, 64), ResBlock(64, 64, 64))

        # Encoder
        self.layer2 = Down(64, 128)
        self.layer3 = Down(128, 256)
        self.layer4 = Down(256, 512)
        self.factor = 2 if self.bicubic else 1
        self.layer5 = Down(512, 1024 // self.factor)

        # Decoder
        self.layer6 = Up(1024, 512 // self.factor, bicubic=self.bicubic)
        self.layer7 = Up(512, 256 // self.factor, bicubic=self.bicubic)
        self.layer8 = Up(256, 128 // self.factor, bicubic=self.bicubic)
        self.layer9 = Up(128, 64, bicubic=self.bicubic)

        self.layer10 = nn.Conv2d(64, self.num_classes, kernel_size=1)

        self.criterion = (
            nn.CrossEntropyLoss() if num_classes > 1 else nn.BCEWithLogitsLoss()
        )

        self.adversarial_loss = nn.BCEWithLogitsLoss()

        self.discriminator = nn.Sequential(
            ResBlock(in_size=1, hidden_size=64, out_size=64),
            ResBlock(in_size=64, hidden_size=128, out_size=128),
            ResBlock(in_size=128, hidden_size=64, out_size=64),
            AttnBlock(in_ch=64),
            nn.Conv2d(64, self.num_classes, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        self.automatic_optimization = False
        self.save_hyperparameters()

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)

        x6 = self.layer6(x5, x4)
        x6 = self.layer7(x6, x3)
        x6 = self.layer8(x6, x2)
        x6 = self.layer9(x6, x1)

        return self.layer10(x6)

    def shared_step(self, batch):
        image, label = batch
        output = self(image)
        loss = self.criterion(output, label)
        self.log_dict({f"{'train' if self.training else 'val'}/loss": loss})
        return loss

    def training_step(self, batch, batch_idx):
        g_opt, e_opt, d_opt = self.optimizers()
        image, label = batch

        output = self(image)
        loss = self.criterion(output, label)

        g_opt.zero_grad()
        self.manual_backward(loss)
        g_opt.step()

        d_opt.zero_grad()
        d_real = self.adversarial_loss(
            self.discriminator(label), torch.ones_like(label)
        )
        d_fake = self.adversarial_loss(
            self.discriminator(output), torch.zeros_like(label)
        )
        self.manual_backward(d_real + d_fake)
        d_opt.step()

        e_opt.zero_grad()
        d_real = self.adversarial_loss(
            self.discriminator(output), torch.ones_like(label)
        )
        self.manual_backward(d_real)
        d_opt.step()

        self.log_dict({f"{'train' if self.training else 'val'}/loss": loss})

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch)

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch)

    def configure_optimizers(self):
        g_opt = torch.optim.AdamW(self.parameters(), lr=1e-5)

        e_params = (
            list(self.layer2.parameters())
            + list(self.layer3.parameters())
            + list(self.layer4.parameters())
            + list(self.layer5.parameters())
        )
        e_opt = torch.optim.AdamW(e_params, lr=1e-5)

        # todo: figure out what params should be there
        d_params = (
            list(self.layer6.parameters())
            + list(self.layer7.parameters())
            + list(self.layer8.parameters())
            + list(self.layer9.parameters())
            + list(self.discriminator.parameters())
        )
        d_opt = torch.optim.AdamW(d_params, lr=1e-5)
        return g_opt, e_opt, d_opt
