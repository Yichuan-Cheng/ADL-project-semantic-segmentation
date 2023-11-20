import os
from collections import OrderedDict
import torch.utils.data as data
import utils


class CamVid(data.Dataset):
    color_encoding = OrderedDict([
        ('Bicyclist', (0,128, 192)),
        ('Building', (128, 0, 0)),
        ('Car', (64, 0, 128)),
        ('Column_Pole', (192, 192, 128)),
        ('Fence', (64, 64, 128)),
        ('Pedestrian', (64, 64, 0)),
        ('Road', (128, 64, 128)),
        ('Sidewalk', (0, 0, 192)),
        ('SignSymbol', (192, 128, 128)),
        ('Sky', (128, 128, 128)),
        ('Tree', (128, 128, 0)),
    ])
    def __init__(self, root_dir, mode='train', transform=None, label_transform=None, loader=utils.pil_loader):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.label_transform = label_transform
        self.loader = loader
        self.data_folder=os.path.join(root_dir, mode)
        self.label_folder=os.path.join(root_dir, mode+'_labels')
        self.data=utils.get_files(self.data_folder, extension_filter='.png')
        self.labels=utils.get_files(self.label_folder, extension_filter='.png')

    def __getitem__(self, index):
        data_path, label_path = self.data[index], self.labels[index]
        img, label = self.loader(data_path, label_path)
        if self.transform is not None:
            img = self.transform(img)
        if self.label_transform is not None:
            label = self.label_transform(label)
        return img, label

    def __len__(self):
        return len(self.data)
