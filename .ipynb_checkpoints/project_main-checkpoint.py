import argparse
import numpy as np
import torch
import random
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from models import CNN_v1, CNN_v2, MLP
from fmnist import FMNIST
from networks_lightning import LightningModule
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
#from torchmetrics.classification import Accuracy

seed=42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate models on FashionMNIST")
    
    # Flags for training and evaluation
    parser.add_argument('--train', action='store_true', help="Flag to train the model")
    parser.add_argument('--evaluate', action='store_true', help="Flag to evaluate the model")
    parser.add_argument('--test', action='store_true', help='Optional test flag')
    parser.add_argument('--epochs', type=int, default=5, help="Number of epochs to train")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to checkpoint file for evaluation")
    parser.add_argument('--model_type', type=str, choices=['cnn_v1','cnn_v2', 'mlp'], required=True, help="Model type to use")
    parser.add_argument('--change_classes', type=str, default=None, help="Comma-separated list of selected classes")

    # Parse the arguments
    args = parser.parse_args()

    if args.change_classes:
        selected_classes = [cls.strip() for cls in args.change_classes.split(',')]
    else:
        selected_classes = ["T-shirt/top", "Trouser", "Pullover", "Sneaker", "Bag", "Ankle boot"]

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = FMNIST(selected_classes=selected_classes,
                           train=True,
                           transform=transform,
                           download=True)

    train_dataset, val_dataset = train_dataset.split()
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    #model based on the flag
    if args.model_type == 'mlp':
        model = MLP(in_channels=28*28, 
                    hidden_units=128,
                    out_channels=len(selected_classes))
        
    elif args.model_type == 'cnn_v1':
        model = CNN(in_channels=1, 
                    hidden_units=32, 
                    out_channels=len(selected_classes))
        
    elif args.model_type == 'cnn_v2':
        model = CNN_v2(in_channels=1, 
                       hidden=32, o
                       ut_channels=len(selected_classes))
    
    # Initialize the Lightning module
    lightning_model = LightningModel(model=model, lr=args.lr)

    # Early stopping callback to stop training when validation accuracy stops improving
    early_stop_callback = EarlyStopping(monitor="val_loss", 
                                        patience=3, 
                                        mode="min", 
                                        verbose=True)

    # Create the Trainer
    trainer = pl.Trainer(max_epochs=args.epochs,
                         callbacks=[early_stop_callback],
                         progress_bar_refresh_rate=20,
                         gpus=1 if torch.cuda.is_available() else 0)

    if args.train:
        # Train the model
        trainer.fit(lightning_model, train_loader, val_loader)
        if args.evaluate:  # If we want to evaluate after training
            print("Evaluating the model...")
            trainer.test(lightning_model, val_loader)  # Test on validation set

    elif args.evaluate:
        # Evaluate an existing model from checkpoint
        if not args.checkpoint:
            print("Error: --checkpoint must be specified for evaluation.")
            return
        
        # Load the model from the checkpoint
        lightning_model = LightningModel.load_from_checkpoint(args.checkpoint)
        
        # Test the model
        print(f"Evaluating model from checkpoint: {args.checkpoint}")
        trainer.test(lightning_model, val_loader)  # Test on validation set


if __name__ == "__main__":
    main()
