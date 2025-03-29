import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms
import numpy as np
import random

seed = 42
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

class FMNIST(Dataset):
    def __init__(self,selected_classes,train=True, transform=None,download=True,rotate=False,rotate_test=False):
        """
        Args:
            train : If True, loads the training dataset, otherwise loads the test dataset.
            transform: Optional transform to be applied on a sample.
            download: Download the Dataset (Default:True)
            rotate : Rotate the training data.
            rotate_test: Rotate the dataset at evaluation.
        """
        self.__train = train
        self.transform = transform
        self.__label_dict = {index:label for index, label in enumerate(selected_classes)}
        self.__class_to_index = {class_name: idx for idx, class_name in self.__label_dict.items()}
        self.rotate=rotate
        self.rotate_test = rotate_test

        # Load FashionMNIST dataset
        full_dataset = datasets.FashionMNIST(root='./data', train=train, download=download)
        
        #extracting data points that only belong to the selected classes.
        index_to_labels_classes = {index:label for index, label in enumerate(full_dataset.classes)}
        indices = [i for i, label in enumerate(full_dataset.targets) if index_to_labels_classes[label.item()] in selected_classes]
        
        # Update dataset and targets
        self.data = full_dataset.data[indices]
        #mapping all the targets to 0 to 5.
        self.targets = torch.tensor([self.__class_to_index[index_to_labels_classes[label.item()]] for label in full_dataset.targets[indices]])
        
    @property
    def train(self):
        return self.__train

    @property
    def label_dict(self):
        return self.__label_dict
    @property
    def class_to_index(self):
        return self.__class_to_index
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """returns a single item (image and label) from the dataset."""
        img, label = self.data[idx], self.targets[idx]
        
        # apply the transformation
        if self.transform:
            img = self.transform(img)
        
        if self.rotate:
           angle = random.choice([0, 90, 180, 270])
           img = transforms.functional.rotate(img,angle)

        if self.rotate_test and not self.train:
           angle = random.choice([0, 90, 180, 270])
           img = transforms.functional.rotate(img,angle)

        return img, label
    
    def split(self,ratio=0.9):
        self.ratio=ratio
        if self.__train:
            train_size = int(len(self) * self.ratio)
            val_size = len(self) - train_size

            indices = torch.randperm(len(self)).tolist() 
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]
            
            train_dataset = Subset(self, train_indices)
            val_dataset = Subset(self, val_indices)
            
            return train_dataset, val_dataset
        else:
            raise ValueError("Cannot split test data, only training data can be split.")