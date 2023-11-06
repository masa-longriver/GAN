import json
import os
import pickle
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

class Dataset():
    def __init__(self, dataset):
        self._dataset = dataset
        self.check_dataset()
        self._dataset_dir = os.path.join(os.getcwd(),
                                         'data/datasets',
                                         f'{self._dataset.lower()}.pkl')
        config_nm = os.path.join(os.getcwd(),
                                 'configs',
                                 f'{self._dataset.lower()}.json')
        with open(config_nm, 'r') as f:
            self._config = json.load(f)
    
    def check_dataset(self):
        supported_dataset = ['food-101', 'food-101-small']
        if self._dataset.lower() not in supported_dataset:
            raise NotImplementedError("Dataset is not supported.")
    
    def save_dataset(self, ds):
        with open(self._dataset_dir, 'wb') as f:
            pickle.dump(ds, f)
    
    def create_dataset(self):
        data_dir = self._config['path']
        transform = transforms.Compose([
            transforms.Resize((self._config['height'], self._config['width'])),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        ds = datasets.ImageFolder(root=data_dir, transform=transform)
        self.save_dataset(ds)

        return ds
    
    def get_dataloader(self):
        if os.path.exists(self._dataset_dir):
            with open(self._dataset_dir, 'rb') as f:
                ds = pickle.load(f)
        else:
            ds = self.create_dataset()
        
        dl = DataLoader(ds,
                        batch_size=self._config['batch_size'],
                        shuffle=True)
        
        return dl
    
    def get_config(self):
        
        return self._config