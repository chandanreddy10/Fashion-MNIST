import argparse
import numpy as np
import torch
import random
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from networks import CNN_v1, CNN_v2, CNN_v3, MLP
from fmnist import FMNIST
from networks_lightning import LightningModel, TestEpochEndCallback
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os

seed=42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate models on FashionMNIST")
    
    # Flags for training and evaluation
    parser.add_argument('--train', action='store_true', help="Flag to train the model")
    parser.add_argument('--evaluate', action='store_true', help="Flag to evaluate the model")
    parser.add_argument('--rotate_train',action='store_true',help="Flag to  Rotate Train and Validation dataset.")
    parser.add_argument('--rotate_test',action="store_true",help="Flag to Rotate only test dataset.")
    parser.add_argument('--use_augmentations', action="store_true", help="Apply additional augmentations during training.")
    parser.add_argument('--plot_error',action="store_true",help="Plot Confusion Matrix and Bar Plot (only in evaluation).")
    parser.add_argument('--epochs', type=int, default=5, help="Number of epochs to train")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to checkpoint file for evaluation")
    parser.add_argument('--model_type', type=str, choices=['cnn_v1','cnn_v2', 'cnn_v3','mlp'],help="Model type to use")
    parser.add_argument('--change_classes', type=str, default=None, help="Comma-separated list of selected classes")
    parser.add_argument('--train_val_split',type=float,default=0.8,help="Set Train and Validation data split.(Optional)")
    # Parse the arguments
    args = parser.parse_args()
    try:
        if args.change_classes:
            selected_classes = [cls.strip() for cls in args.change_classes.split(',')]
        else:
            selected_classes = ["T-shirt/top", "Trouser", "Pullover", "Sneaker", "Bag", "Ankle boot"]

        base_transform = [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]
        if args.use_augmentations:
            print("Using additional augmentations...")
            augmentations = [
                transforms.RandomHorizontalFlip(), 
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))  # Random shift
            ]
            transform = transforms.Compose(base_transform+augmentations)
        else:
            transform = transforms.Compose(base_transform)
        split_ratio = args.train_val_split
        train_dataset = FMNIST(selected_classes=selected_classes,
                            train=True,
                            transform=transform,
                            rotate=args.rotate_train,
                            download=True)
        
        train_dataset, val_dataset = train_dataset.split(ratio=split_ratio)
        test_dataset = FMNIST(selected_classes=selected_classes,
                                    train=False,
                                    transform=transforms.Compose(base_transform),
                                    rotate_test=args.rotate_test,
                                    download=True)
        #Dataloaders for train, val and test.
        train_loader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True)
        val_loader = DataLoader(val_dataset,batch_size=args.batch_size,shuffle=False)
        test_loader = DataLoader(test_dataset,batch_size=args.batch_size,shuffle=False)

        if args.train:
        #model based on the flag
            model_dict = {
                'mlp': MLP(in_channels=28*28, hidden_units=128, out_channels=len(selected_classes)),
                'cnn_v1': CNN_v1(in_channels=1, hidden_units=32, out_channels=len(selected_classes)),
                'cnn_v2': CNN_v2(in_channels=1, hidden_units=32, out_channels=len(selected_classes)),
                'cnn_v3': CNN_v3(in_channels=1, hidden_units=32, out_channels=len(selected_classes))
            }
            model = model_dict.get(args.model_type)

            if model is None:
                print("Please flag the Model Type. Ignore for Evaluation.")
                return
            
            # Initialize the Lightning module
            lightning_model = LightningModel(model=model, lr=args.lr)
            model_class_name = model.__class__.__name__

            # Early stopping callback to stop training when validation accuracy stops improving
            early_stop_callback = EarlyStopping(monitor="val_acc", 
                                                patience=3, 
                                                mode="max", 
                                                verbose=True)
            
            # Model checkpoint for saving the model under custom name.
            checkpoint_callback = ModelCheckpoint(
                                                dirpath='checkpoints/',
                                                filename=f'{model_class_name}'+'_{epoch}_{val_acc:.2f}',
                                                monitor='val_acc',  
                                                save_top_k=1, 
                                                mode='max',  
                                                verbose=True)
            
            # Create the Trainer
            # TensorBoard Logger
            tb_logger = TensorBoardLogger("tensorboard_logs", name=model_class_name)
            tb_logger.log_hyperparams({'lr': args.lr, 'batch_size': args.batch_size, 'model_name': model_class_name})

            trainer = pl.Trainer(logger=tb_logger,
                                max_epochs=args.epochs,
                                callbacks=[early_stop_callback,checkpoint_callback],
                                gpus=1 if torch.cuda.is_available() else 0)

        
            # Train the model
            trainer.fit(lightning_model, train_loader, val_loader)
            if args.evaluate:  # If we want to evaluate after training
                print("Evaluating the model...")
                best_model_path = checkpoint_callback.best_model_path
                lightning_model = LightningModel.load_from_checkpoint(best_model_path)
                trainer.test(lightning_model, test_loader)

        elif args.evaluate:
            # Evaluate an existing model from checkpoint
            csv_logger = CSVLogger("logs", name="lightning_logs")
            if not args.checkpoint:
                print("Error: --checkpoint must be specified for evaluation.")
                return
            
            # Load the model from the checkpoint
            model = LightningModel.load_from_checkpoint(args.checkpoint)
            if args.plot_error:
                evaluate_trainer = pl.Trainer(
                                max_epochs=1,
                                callbacks=[TestEpochEndCallback()],
                                gpus=1 if torch.cuda.is_available() else 0)
                outputs= evaluate_trainer.test(model, test_loader)
            else:
                evaluate_trainer = pl.Trainer(
                                max_epochs=1,
                                gpus=1 if torch.cuda.is_available() else 0)
                outputs= evaluate_trainer.test(model, test_loader)
           

    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()
