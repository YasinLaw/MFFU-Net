from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl

from data import CrackForest
from model import UNet

if __name__ == "__main__":
    model = UNet()
    data_module = CrackForest()

    trainer = pl.Trainer(
        accelerator="cpu",
        max_epochs=100,
        # logger=WandbLogger(project="MFFU-Net", name="res-ca-bicubic"),
    )
    trainer.fit(
        model,
        train_dataloaders=data_module.train_dataloader(),
        val_dataloaders=data_module.val_dataloader(),
    )

    trainer.test(model, dataloaders=data_module.test_dataloader())
