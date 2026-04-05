################################################################################
# FOLDER: data
# FILE:   imagenet.py
# PATH:   .\data\imagenet.py
################################################################################

import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from utils.logger import get_logger

logger = get_logger("imagenet_loader", logfile="logs/imagenet.log")

def get_imagenet_loaders(batch_size=32, num_workers=0, split_test=False, fast_dev_mode=True):
    # ImageNet requires 224x224 input sizes and specific normalization
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pin_mem = torch.cuda.is_available() and num_workers > 0
    data_dir = './data/imagenet'
    
    # FALLBACK: If ImageNet is not downloaded (it is 150GB), use FakeData to test the pipeline safely
    use_fake_data = not os.path.exists(os.path.join(data_dir, 'train'))
    if use_fake_data:
        logger.warning("Real ImageNet files not found in ./data/imagenet/train. Using FakeData for architecture testing.")
        dataset_class = lambda is_train, transform: torchvision.datasets.FakeData(
            size=5000 if is_train else 1000, image_size=(3, 224, 224), num_classes=1000, transform=transform
        )
    else:
        dataset_class = lambda is_train, transform: torchvision.datasets.ImageFolder(
            root=os.path.join(data_dir, 'train' if is_train else 'val'), transform=transform
        )

    if split_test:
        train_data_full = dataset_class(is_train=True, transform=transform_train)
        val_data_full = dataset_class(is_train=True, transform=transform_test)
        
        indices = list(range(len(train_data_full)))
        np.random.seed(42)  
        np.random.shuffle(indices)
        
        if fast_dev_mode:
            train_idx, val_idx = indices[:1000], indices[1000:1200]
        else:
            # Standard 90/10 split of training data if real ImageNet
            split = int(0.9 * len(indices))
            train_idx, val_idx = indices[:split], indices[split:]
            
        train_subset = Subset(train_data_full, train_idx)
        val_subset = Subset(val_data_full, val_idx)
        
        trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_mem)
        valloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)
        
        testset = dataset_class(is_train=False, transform=transform_test)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)
        
        return trainloader, valloader, testloader

    else:
        trainset = dataset_class(is_train=True, transform=transform_train)
        testset = dataset_class(is_train=False, transform=transform_test)
        
        if fast_dev_mode:
            num_train = len(trainset)
            indices = list(range(num_train))
            trainset = Subset(trainset, indices[:int(0.1 * num_train)])

        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_mem)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)

        return trainloader, testloader