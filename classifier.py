import numpy as np
from dataset import generate_dataloaders

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter




class CNNModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = torch.nn.Sequential(
                torch.nn.Conv2d(3, 8, 5),
                torch.nn.ReLU(),
                torch.nn.Conv2d(8, 16, 5),
                torch.nn.ReLU(),
                # torch.nn.Conv2d(16, 16, 7),
                # torch.nn.ReLU(),
                # torch.nn.Conv2d(16, 16, 7),
                # torch.nn.ReLU(),
                # torch.nn.Conv2d(16, 16, 7),
                # torch.nn.ReLU(),
                # torch.nn.Conv2d(16, 16, 7),
                # torch.nn.ReLU(),
                # torch.nn.Conv2d(16, 16, 7),
                # torch.nn.ReLU(),
                # torch.nn.Conv2d(16, 16, 7),
                # torch.nn.ReLU(),
                torch.nn.Flatten(),
                torch.nn.Linear(50176, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 13),
                # torch.nn.Softmax()
            )

    def forward(self, features):
        """Takes in features and makes a prediction"""
        return self.layers(features)

def train(model, train_loader, val_loader, lr=1e-3, epochs=1000, optimiser=torch.optim.SGD):
    writer = SummaryWriter()
    optimiser = optimiser(model.parameters(), lr=lr)
    batch_idx = 0
    model.train()

    for epoch in range(epochs):
        for batch in train_loader:
            print(f'Batch Idx: {batch_idx}')
            features, labels = batch
            predictions = model(features)
            # print(predictions.type(), labels.type())
            loss = F.cross_entropy(predictions, labels)
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
            writer.add_scalar('loss-train', loss.item(), batch_idx)
            batch_idx += 1
            if batch_idx % 50 == 0:
                val_loss = evaluate(model, val_loader)
                writer.add_scalar('loss-val', val_loss, batch_idx)

def evaluate(model, data_loader):
    losses = []
    model.eval()

    for batch in data_loader:
        features, labels = batch
        predictions = model(features)
        loss = F.cross_entropy(predictions, labels)
        losses.append(loss.detach())
        
    avg_loss = np.mean(losses)
    return avg_loss

if __name__ == '__main__':
    
    print('Building model')
    model = CNNModel()
    
    print('Generating dataloaders')
    dataloaders = generate_dataloaders()
    train_dataloader, val_dataloader = dataloaders['train'], dataloaders['validation']
    
    print('Training model')
    train(model, train_dataloader, val_dataloader, epochs = 800)