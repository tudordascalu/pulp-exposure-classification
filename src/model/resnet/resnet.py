import torch
from pytorch_lightning import LightningModule
import torchvision.models as models
from sklearn.metrics import f1_score, roc_auc_score
from torch.nn import BCELoss, Linear, Conv2d
from torch.optim import RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import ResNet50_Weights
from torch import sigmoid


class ResNet(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.__dict__.update(locals())
        self.save_hyperparameters()
        self.config = config

        # Utils
        self.bce_loss = BCELoss()
        self.predictions = []
        self.targets = []

        # Model
        resnets = {
            18: models.resnet18, 34: models.resnet34,
            50: models.resnet50, 101: models.resnet101,
            152: models.resnet152
        }
        self.model = resnets[self.config["resnet_version"]](weights=ResNet50_Weights.DEFAULT)
        self.conv1 = Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1, bias=False)
        linear_size = list(self.model.children())[-1].in_features
        self.model.fc = Linear(linear_size, 6)
        self.fc1 = Linear(8, config["n_classes"])

        # Freeze the early layers (everything but the final layer)
        for name, param in self.model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False

    def forward(self, image, extra):
        # Convert to 3 channels image
        x = self.conv1(image)

        # Compute image features
        x = self.model(x)

        # Concatenate clinical features
        x = torch.concat((x, extra), dim=1)

        # Compute output
        x = self.fc1(x)

        x = sigmoid(x)

        return x

    def configure_optimizers(self):
        optimizer = RMSprop(self.parameters(),
                            lr=float(self.config["learning_rate"]),
                            weight_decay=float(self.config["weight_decay"]),
                            momentum=0.9)
        scheduler = ReduceLROnPlateau(optimizer=optimizer, mode="min", patience=self.config["scheduler_patience"],
                                      verbose=True)
        return {"optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",  # scheduler.step() is called after each epoch
                    "frequency": 1,  # scheduler.step() is called once every "frequency" times.
                    "strict": True,
                }}

    def training_step(self, batch, batch_idx):
        # Extract data
        image, extra, target = batch["image"], batch["extra"], batch["target"]

        prediction = self.forward(image, extra)

        # Compute metrics
        loss = self.bce_loss(prediction, target)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Extract data
        image, extra, target = batch["image"], batch["extra"], batch["target"]

        # Predict
        with torch.no_grad():
            prediction = self.forward(image, extra)
        prediction_binary = (prediction > .5).type(torch.float32)

        # Compute metrics
        target_np = target.cpu().numpy()
        loss = self.bce_loss(prediction, target)
        f1 = f1_score(prediction_binary.cpu().numpy(), target_np, average="macro")
        auc = roc_auc_score(target_np, prediction.cpu().numpy())

        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("val_f1", f1, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_auc", auc, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        # Extract data
        image, extra, target = batch["image"], batch["extra"], batch["target"]

        # Predict
        with torch.no_grad():
            prediction = self.forward(image, extra)
        prediction_binary = (prediction > .5).type(torch.float32)

        # Compute metrics
        target_np = target.cpu().numpy()
        loss = self.bce_loss(prediction, target)
        f1 = f1_score(prediction_binary.cpu().numpy(), target_np, average="macro")
        auc = roc_auc_score(target_np, prediction.cpu().numpy())

        self.log("test_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("test_f1", f1, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_auc", auc, prog_bar=True, on_step=False, on_epoch=True)

    def predict_step(self, batch, batch_idx):
        # Extract data
        image, extra, target = batch["image"], batch["extra"], batch["target"]

        # Predict
        y_pred = self.forward(image, extra)

        self.predictions.append(y_pred)
        self.targets.append(target)

    def on_predict_epoch_end(self):
        y_pred = torch.concat(self.predictions)
        target = torch.concat(self.targets)

        torch.save(y_pred, f"output/predictions.pt")
        torch.save(target, f"output/targets.pt")
