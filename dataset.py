import os
from collections import OrderedDict
import torch.utils.data as data
import utils
from torchvision import transforms


class CamVid(data.Dataset):
    def __init__(self, root_dir, mode='train', transform=None, label_transform=None, loader=utils.pil_loader):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.label_transform = label_transform
        self.loader = loader
        self.data_folder=os.path.join(root_dir, mode)
        self.label_folder=os.path.join(root_dir, mode+'annot')
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

def get_dataloaders(root_dir='CamVid',batch_size=8, num_workers=0, height=720, width=960):
    train_data_transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((height, width)), transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),])
    val_data_transform=train_data_transform
    test_data_transform=train_data_transform
    label_transform=transforms.Compose([transforms.PILToTensor(),transforms.Resize((height, width),transforms.InterpolationMode.NEAREST)])
    train_set=CamVid(root_dir=root_dir, mode='train',transform=train_data_transform, label_transform=label_transform)
    val_set=CamVid(root_dir=root_dir, mode='val',transform=val_data_transform, label_transform=label_transform)
    test_set=CamVid(root_dir=root_dir, mode='test',transform=test_data_transform, label_transform=label_transform)
    train_loader=data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader=data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader=data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


