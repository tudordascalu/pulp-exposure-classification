"""
This script trains a model and saves its outputs.
"""
import numpy as np
import torch
import yaml
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from src.data.data import PulpExposureDataset
from src.model.resnet.resnet import ResNet

if __name__ == "__main__":
    with open("./src/model/resnet/scripts/config.yml", "r") as f:
        config = yaml.safe_load(f)

    # Find out whether gpu is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    data_train = np.load(f"data/final/data_train.npy")
    data_val = np.load(f"data/final/data_val.npy")
    data_test = np.load(f"data/final/data_test.npy")

    # Define transforms
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=20, translate=(.10, .10),
                                scale=(.8, 1.2)),
        transforms.RandomPerspective(p=.5, interpolation=InterpolationMode.NEAREST,
                                     distortion_scale=.1),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
    ])

    # Load dataset
    dataset_train = PulpExposureDataset(data=data_train, transform=transform)
    dataset_val = PulpExposureDataset(data_val)
    dataset_test = PulpExposureDataset(data_test)

    loader_args = dict(batch_size=config["batch_size"])
    loader_train = DataLoader(dataset_train, shuffle=True, batch_size=config["batch_size"])
    loader_val = DataLoader(dataset_val, shuffle=False, batch_size=len(data_val))
    loader_test = DataLoader(dataset_test, shuffle=False, batch_size=len(data_test))

    # Define model
    model = ResNet(config)
    logger = loggers.TensorBoardLogger(save_dir=config["checkpoints_path"], name=None)
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_auc", mode="max",
                                          filename="epoch={epoch:02d}-val_loss={val_auc:.2f}")
    trainer_args = dict(max_epochs=config["epochs"],
                        callbacks=[checkpoint_callback],
                        logger=logger,
                        log_every_n_steps=10)
    if device.type == "cpu":
        trainer = Trainer(**trainer_args)
    else:
        trainer = Trainer(accelerator="gpu", strategy="ddp", devices=1, **trainer_args)

    # Train model
    trainer.fit(model=model, train_dataloaders=loader_train, val_dataloaders=loader_val)

    # Test
    trainer.test(ckpt_path="best", dataloaders=loader_test)
