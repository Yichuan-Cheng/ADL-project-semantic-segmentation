import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models import get_model
from dataset import get_dataloaders

color_encoding = [
    ('sky', (128, 128, 128)),
    ('building', (128, 0, 0)),
    ('pole', (192, 192, 128)),
    ('road', (128, 64, 128)),
    ('pavement', (60, 40, 222)),
    ('tree', (128, 128, 0)),
    ('sign_symbol', (192, 128, 128)),
    ('fence', (64, 64, 128)),
    ('car', (64, 0, 128)),
    ('pedestrian', (64, 64, 0)),
    ('bicyclist', (0, 128, 192)),
]
num_classes = len(color_encoding)
# ignore_index = num_classes

models=get_model()
optimizer=optim.Adam(models.parameters(), lr=1e-3)
criterion=nn.CrossEntropyLoss(ignore_index=num_classes)
train_loader, val_loader, test_loader=get_dataloaders()

def train_epoch(model, train_loader, criterion, optimizer):
    model.train()
    running_loss=0.0
    cnt=0
    for i, data in enumerate(train_loader):
        inputs, labels=data
        optimizer.zero_grad()
        outputs=model(inputs)
        outputs=F.interpolate(outputs, size=labels.shape[1:], mode='bilinear', align_corners=True)
        loss=criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()*inputs.size(0)
        cnt+=inputs.size(0)
    return running_loss/cnt

