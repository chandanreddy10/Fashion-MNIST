import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import utils

class LightningModel(pl.LightningModule):
    def __init__(self, model: nn.Module, lr=0.001,batch_size=32):
        super().__init__()
        self.model = model
        self.lr = lr
        self.batch_size=batch_size
        self.criterion = nn.CrossEntropyLoss()
        self.save_hyperparameters()
        self.preds = []

    def training_step(self, batch, batch_idx):
        X, y = batch
        logits = self.model(X)
        train_loss = self.criterion(logits, y)
        
        # Log the training loss
        self.log('train_loss', train_loss)
        return {'loss':train_loss}
        
    def validation_step(self, batch, batch_idx):
        X, y = batch
        logits = self.model(X)
        val_loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        val_acc = (preds == y).float().mean()

        # Log the validation loss and accuracy
        self.log('val_loss', val_loss, prog_bar=True,logger=True)
        self.log("val_acc", val_acc, prog_bar=True, logger=True)
        return {'val_loss':val_loss,"val_acc":val_acc}
    
    def test_step(self, batch, batch_idx):
        X, y = batch
        logits = self.model(X)
        test_loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        test_acc = (preds == y).float().mean()

        # Log the test loss and accuracy
        self.log('test_loss', test_loss,prog_bar=True,logger=True)
        self.log("test_acc", test_acc, prog_bar=True, logger=True)
        self.preds.append({'preds': preds, 'labels': y})
        return {'preds': preds, 'labels': y}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = {
            "scheduler": lr_scheduler.ReduceLROnPlateau(optimizer,'max',patience=5, factor=0.5),
            "monitor": "val_acc",
            "frequency": 1,
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

class TestEpochEndCallback(Callback):
    def on_test_epoch_end(self,trainer,pl_module):
        accuracy = utils.compute_classwise_accuracy(pl_module.preds, class_names=["T-shirt/top", "Trouser", "Pullover", "Sneaker", "Bag", "Ankle boot"])
        self.log('classwise_accuracy', accuracy)
