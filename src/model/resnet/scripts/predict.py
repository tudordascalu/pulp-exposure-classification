"""
This script trains a model and saves its outputs.
"""
import numpy as np
import torch
import yaml
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from src.data.data import PulpExposureDataset
from src.model.resnet.resnet import ResNet

if __name__ == "__main__":
    with open("./src/model/resnet/scripts/config.yml", "r") as f:
        config = yaml.safe_load(f)

    # Find out whether gpu is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    data_test = np.load(f"data/final/data_test.npy")

    # Load dataset
    dataset_test = PulpExposureDataset(data_test)

    loader_test = DataLoader(dataset_test, shuffle=False, batch_size=config["batch_size"])

    # Define model
    model = ResNet.load_from_checkpoint(
        f"{config['checkpoints_path']}version_2/checkpoints/epoch=epoch=47-val_loss=val_f1=0.92.ckpt", map_location=device)
    trainer_args = dict(max_epochs=config["epochs"],
                        log_every_n_steps=10)
    if device.type == "cpu":
        trainer = Trainer(**trainer_args)
    else:
        trainer = Trainer(accelerator="gpu", strategy="ddp", devices=1, **trainer_args)

    # Test
    trainer.predict(model, dataloaders=loader_test)
