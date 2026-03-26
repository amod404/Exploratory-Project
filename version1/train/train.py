# train/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from .trainer import Trainer

# Assuming you have a function to get data in data/cifar10.py
# from data.cifar10 import get_dataloaders

def train_architecture(model, train_loader, test_loader, epochs=10, lr=0.025):
    """
    The main function called by your evolutionary algorithm to evaluate a new model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Standard setup for image classification
    criterion = nn.CrossEntropyLoss()
    
    # SGD with momentum is standard for NAS on CIFAR-10
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=3e-4)
    
    # Initialize our engine
    trainer = Trainer(model, criterion, optimizer, device)
    
    print(f"Starting training on {device} for {epochs} epochs...")
    
    best_accuracy = 0.0
    
    for epoch in range(epochs):
        train_loss, train_acc = trainer.train_epoch(train_loader)
        test_loss, test_acc = trainer.evaluate(test_loader)
        
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
              
    return best_accuracy