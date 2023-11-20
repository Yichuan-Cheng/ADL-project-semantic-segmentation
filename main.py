import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models import get_model
from dataset import get_dataloaders
from utils import cal_metric

def train(model, train_loader, criterion, optimizer):
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

def eval(model, val_loader, criterion):
    model.eval()
    running_loss=0.0
    preds_list=[]
    labels_list=[]
    cnt=0
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            inputs, labels=data
            outputs=model(inputs)
            outputs=F.interpolate(outputs, size=labels.shape[1:], mode='bilinear', align_corners=True)
            loss=criterion(outputs, labels)
            running_loss+=loss.item()*inputs.size(0)
            cnt+=inputs.size(0)
            preds=outputs.argmax(dim=-1).view(-1).cpu().numpy().tolist()
            labels=labels.view(-1).cpu().numpy().tolist()
            preds_list.extend(preds)
            labels_list.extend(labels)
    metric=cal_metric(preds_list, labels_list)
    return running_loss/cnt, metric

EPOCH=100
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

model=get_model()
optimizer=optim.Adam(model.parameters(), lr=1e-3)
criterion=nn.CrossEntropyLoss(ignore_index=num_classes)
train_loader, val_loader, test_loader=get_dataloaders()
best_metric=0.0
for i in range(EPOCH):
    train_loss=train(model, train_loader, criterion, optimizer)
    val_loss, val_metric=eval(model, val_loader, criterion)
    test_loss, test_metric=eval(model, test_loader, criterion)
    print("Epoch: {0}, Train Loss: {1:.4f}, Val Loss: {2:.4f}, Val Metric: {3:.4f}, Test Loss: {4:.4f}, Test Metric: {5:.4f}".format(i, train_loss, val_loss, val_metric, test_loss, test_metric))
    if val_metric>best_metric:
        best_metric=val_metric
        torch.save(model.state_dict(), 'best_model.pth')

model.load_state_dict(torch.load('best_model.pth'))
test_loss, test_metric=eval(model, test_loader, criterion)
print("-"*20)
print("Final Test Loss: {0:.4f}, Test Metric: {1:.4f}".format(test_loss, test_metric))





