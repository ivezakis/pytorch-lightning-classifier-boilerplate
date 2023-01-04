import pytorch_lightning as pl
import torch.nn as nn
from pytorch_lightning.loggers import TensorBoardLogger

from dataloading.datamodule import MyDataModule
from network_module import Net

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", "-e", type=str, default="task01")
    parser.add_argument("--run_name", "-r", type=str, default="efficientnetb0")
    parser.add_argument("--base_path", "-b", type=str, default="./dataset/")
    parser.add_argument("--img_size", "-i", type=tuple, default=None)

    args = parser.parse_args()

    run_name = f"{args.experiment_name}/{args.run_name}/{args.img_size}"

    tensorboard_logger = TensorBoardLogger(
        save_dir="logs",
        name=run_name,
    )

    dm = MyDataModule(
        batch_size=16,
        train_val_ratio=0.7,
        base_path=args.base_path,
        num_workers=12,
        img_size=args.img_size,
    )
    dm.prepare_data()

    net = Net(
        model=EfficientNetBN(
            model_name="efficientnet-b0",
            num_classes=dm.num_classes,
            pretrained=False,
        ),
        criterion=nn.CrossEntropyLoss(weight=dm.class_weights), # type: ignore
        num_classes=dm.num_classes,
    )

    trainer = pl.Trainer(
        gpus=1, max_epochs=100, log_every_n_steps=1, logger=tensorboard_logger
    )
    trainer.fit(net, dm)
