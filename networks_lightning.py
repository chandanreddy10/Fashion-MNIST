import torch
from torch import nn
import pytorch_lightning as pl

class LightningModel(pl.LightningModule):
    def __init__(self, model: nn.Module, lr=0.001):
        super().__init__()
        self.model = model
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x):
        return self.model(x)
        
    def training_step(self, batch, batch_idx):
        X, y = batch
        logits = self.model(X)
        train_loss = self.criterion(logits, y)
        
        # Log the training loss
        self.log('train_loss', train_loss)
        
        return train_loss
        
    def validation_step(self, batch, batch_idx):
        X, y = batch
        logits = self.model(X)
        val_loss = self.criterion(logits, y)
        
        # Log the validation loss
        self.log('val_loss', val_loss, prog_bar=True)  # prog_bar=True will display it in the progress bar

        return val_loss
    
    def test_step(self, batch, batch_idx):
        X, y = batch
        logits = self.model(X)
        test_loss = self.criterion(logits, y)
        
        # Log the test loss
        self.log('test_loss', test_loss)

        return test_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
