import numpy as np
from dataset import generate_dataloaders
from classifier import train

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet50
from torchvision.models.resnet import ResNet50_Weights

class TransferLearning(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = resnet50(weights=ResNet50_Weights.DEFAULT)
        for param in self.layers.parameters():
            param.requires_grad = False
        fc_inputs = self.layers.fc.in_features
        fc_outputs = self.layers.fc.out_features
        # print(self.feature_extractor)
        linear_layers = torch.nn.Sequential(
            torch.nn.Linear(fc_inputs, fc_outputs),
            torch.nn.ReLU(),
            torch.nn.Linear(fc_outputs, 13)
        )
        self.layers.fc = linear_layers
    
    def forward(self, x):
        return self.layers(x)

if __name__ == '__main__':
    print('Building model')
    resnet_model = TransferLearning()
    
    print('Generating dataloaders')
    dataloaders = generate_dataloaders()
    train_dataloader, val_dataloader = dataloaders['train'], dataloaders['validation']
    
    print('Training model')
    train(resnet_model, train_dataloader, val_dataloader, epochs = 65)
    # print(model.state_dict())
    torch.save(resnet_model.state_dict(), 'resnet_model_weights.pth')
