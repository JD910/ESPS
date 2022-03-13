import h5py
import numpy as np
import argparse
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms.transforms import CenterCrop, RandomResizedCrop, Resize

class H5Dataset(Dataset):
    def __init__(self, h5_path,train = True):
        self.h5_path = h5_path
        self.train = train
        print(h5_path)
        with h5py.File(self.h5_path, 'r') as record:
            keys = list(record.keys())   

    def __getitem__(self, index):

        with h5py.File(self.h5_path, 'r') as record:
            
            CIFAR100_MEANS = (0.485, 0.456, 0.406) #precomputed channel means of CIFAR100(train) for normalization [0.485, 0.456, 0.406]
            CIFAR100_STDS = (0.229, 0.224, 0.225) #precomputed standard deviations [0.229, 0.224, 0.225]
    
            transformations = {
                'train': transforms.Compose([
                    #transforms.ToPILImage(),
                    #transforms.RandomResizedCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(CIFAR100_MEANS, CIFAR100_STDS)
                ]),
                'val': transforms.Compose([
                    #transforms.ToPILImage(),
                    #transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(CIFAR100_MEANS, CIFAR100_STDS)
                ])
            }
            keys = list(record.keys())
            train_data = np.array(record[keys[index]]['train']).astype(np.float32)

            if self.train:
                train_data1 = np.expand_dims(train_data, axis = -1)
                train_data2 = np.concatenate((train_data1, train_data1, train_data1), axis = -1)
                train_data3 = transformations['train'](train_data2)
            else:
                train_data1 = np.expand_dims(train_data, axis = -1)
                train_data2 = np.concatenate((train_data1, train_data1, train_data1), axis = -1)
                train_data3 = transformations['val'](train_data2)
                
        target_data = np.array(record[keys[index]]['target'])
        
        return train_data3, target_data, keys[index]

    def __len__(self):
        with h5py.File(self.h5_path,'r') as record:
            return len(record)

