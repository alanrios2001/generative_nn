from torch.utils.data import DataLoader, Dataset
import torch
import torchvision
import PIL
import os
import numpy as np
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

class CelebADataset(Dataset):
    def __init__(self, path, size = 256):
        self.sizes = [size, size]
        items, labels = [], []

        #i = 0
        for data in os.listdir(path):
            item = os.path.join(path, data)
            items.append(item)
            labels.append(0)
            #i += 1
            #if i == 1000:
                #break
        self.items = items
        self.labels = labels

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        data = PIL.Image.open(self.items[idx]).convert("RGB")
        data = np.asarray(torchvision.transforms.Resize(self.sizes)(data))
        data = np.transpose(data, (2, 0, 1)).astype(np.float32, copy=False)
        data = torch.from_numpy(data).div(255)
        return data, self.labels[idx]

def load_dataset(path, BATCH_SIZE):
    # Dataset
    dataset = CelebADataset(path, size=128)

    # dataloader
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader