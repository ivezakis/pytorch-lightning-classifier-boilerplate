import os

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics as tm


class Net(pl.LightningModule):
    def __init__(self, model, criterion, num_classes):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.num_classes = num_classes

        self.confusion_matrix = tm.ConfusionMatrix(
            task="multiclass", num_classes=num_classes
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=3e-4)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y = F.one_hot(y, num_classes=self.num_classes).float()
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        self.confusion_matrix.update(y_hat, y)
        y = F.one_hot(y, num_classes=self.num_classes).float()
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def validation_epoch_end(self, outs):
        cm = self.confusion_matrix.compute()
        self.confusion_matrix.reset()

        if self.logger:
            run_name = self.logger.name + "/" + str(self.logger.version) # type: ignore
        else:
            run_name = "default"
        os.makedirs(f"confusion_matrices/{run_name}", exist_ok=True)
        torch.save(
            cm, os.path.join(f"confusion_matrices/{run_name}", "confusion_matrix.pt")
        )
