import pickle
from sklearn.model_selection import train_test_split
import torch
from PIL import Image
import numpy as np

class MNIST(torch.utils.data.Dataset):
    def __init__(self, file_path, mode: str, transform=None):
        self.transform = transform
        self.mode = mode

        with open(file_path, "rb") as f:
            trainX, trainLabel, testX, test_label = pickle.load(f)

        if mode == 'train' or mode == 'val':
            train_data, val_data, train_label, val_label = train_test_split(trainX, trainLabel, test_size=0.1, random_state=42)
        if mode == 'train':
            self.X = train_data
            self.y = train_label
        elif mode == 'val':
            self.X = val_data
            self.y = val_label
        else:
            self.X = testX
            self.y = test_label

    def __getitem__(self, index):
        img, target = self.X[index], self.y[index]

        img = Image.fromarray(img.squeeze().astype(np.uint8), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.from_numpy(target.copy()).to(torch.long)

    def __len__(self):
        return len(self.X)