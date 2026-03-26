# train/trainer.py
import torch
import time

class Trainer:
    """
    The core training engine. It handles the math of backpropagation
    for ANY architecture passed to it.
    """
    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.model.to(self.device)

    def train_epoch(self, train_loader):
        """Trains the model for one single pass over the dataset."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # 1. Clear old gradients
            self.optimizer.zero_grad()
            
            # 2. Forward pass (push images through the architecture)
            outputs = self.model(inputs)
            
            # 3. Calculate the error (loss)
            loss = self.criterion(outputs, targets)
            
            # 4. Backward pass (calculate gradients)
            loss.backward()
            
            # 5. Update the weights
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        accuracy = 100. * correct / total
        return total_loss / len(train_loader), accuracy

    def evaluate(self, test_loader):
        """Tests the model on unseen data to get its true accuracy."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad(): # No training happens here!
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        accuracy = 100. * correct / total
        return total_loss / len(test_loader), accuracy