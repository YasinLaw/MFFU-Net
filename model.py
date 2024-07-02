import schedulefree
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
            nn.Conv2d(self.num_classes, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.LazyLinear(256),
            nn.ReLU(inplace=True),
            nn.LazyLinear(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.LazyLinear(1),
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

        batch_size = label.size(dim=0)

        d_real = self.adversarial_loss(
            self.discriminator(label),
            torch.ones(batch_size, 1).to(self.device),
        )
        d_fake = self.adversarial_loss(
            self.discriminator(output.detach()),
            torch.zeros(batch_size, 1).to(self.device),
        )

        err_d = d_real + d_fake
        d_opt.zero_grad()
        self.manual_backward(err_d)
        d_opt.step()

        err_e = self.adversarial_loss(
            self.discriminator(self(image)),
            torch.ones(batch_size, 1).to(self.device),
        )
        e_opt.zero_grad()
        self.manual_backward(err_e)
        e_opt.step()

        self.log_dict({f"{'train' if self.training else 'val'}/loss": loss})

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch)

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch)

    def on_train_epoch_start(self):
        opts = self.optimizers()
        for opt in opts:
            opt.train()

    def on_validation_epoch_start(self):
        opts = self.optimizers()
        for opt in opts:
            opt.eval()

    def on_test_start(self):
        opts = self.optimizers()
        for opt in opts:
            opt.eval()

    def configure_optimizers(self):
        g_opt = schedulefree.AdamWScheduleFree(self.parameters(), lr=1e-5)

        e_params = (
            list(self.layer6.parameters())
            + list(self.layer7.parameters())
            + list(self.layer8.parameters())
            + list(self.layer9.parameters())
            + list(self.layer10.parameters())
        )
        e_opt = schedulefree.AdamWScheduleFree(e_params, lr=1e-5)

        d_params = list(self.discriminator.parameters())
        d_opt = schedulefree.AdamWScheduleFree(d_params, lr=1e-5)
        return g_opt, e_opt, d_opt
