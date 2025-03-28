import torch
from torch import nn
import pytorch_lightning as pl

class LightningModel(pl.LightningModule):
    def __init__(self, model:nn.Module, lr=0.001):
        super().__init__()
        self.model = model
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x):
        return self.model(x)
        
    def training_step(self,batch,batch_idx):
        X, y = batch
        logits = self.model(X)
        loss = self.criterion(logits,y)
        
    def validation_step(self,batch,batch_idx):
        X,y = batch
        logits = self.model(X)
        loss = self.criterion(logits,y)
        
    def test_step(self,batch,batch_idx):
        X,y = batch
        logits = self.model(X)
        loss = self.criterion(logits,y)
        
    def configure_optimizers(self):
        optimizer= torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    