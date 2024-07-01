import lightning as pl
from lightning.pytorch.loggers import WandbLogger

from data import CrackForest
from model import UNet

if __name__ == "__main__":
    model = UNet()
    data_module = CrackForest()

    trainer = pl.Trainer(
        accelerator="cpu",
        max_epochs=30,
        logger=WandbLogger(log_model="all", project="MFFU-Net"),
        log_every_n_steps=2,
    )
    trainer.fit(
        model,
        train_dataloaders=data_module.train_dataloader(),
        val_dataloaders=data_module.val_dataloader(),
    )

    trainer.test(model, dataloaders=data_module.test_dataloader())