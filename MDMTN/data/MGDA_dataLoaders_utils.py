import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim
from os import path
from torch import Tensor
from typing import Iterable, Union
from PIL import Image
from torchvision import transforms
import pickle

from typing import Iterable, Union
_params_t = Union[Iterable[Tensor], Iterable[dict]]
from data.get_multimnist_dataset import get_dataset
from data.cifar10mnist_dataset import PaddedDataset, create_cifar10mnist

class MGDA_Data:
    """
    This class bundles methods regarding the datasets for MGDA method.

    Example:
        >> X, y = MGDA_Data.get_data(path="")
        >> MGDA_Data.visualize_training_data(X, y)
    """

    @staticmethod
    def get_data(path: str, train_loader, test_loader):
        """
        Returns a list X which contains N 32x32 images and a corresponding list y with the target values.

        :param path: root path, where the folder 'processed' is located
        :return: X and y (both as np.array). X[i] has a shape of (32, 32), y[i] has a shape of (2, 10)
        """

        raw_X = []
        raw_y = []

        # Combine data in train and test_loader
        for dat in train_loader:
            ims = dat[0].numpy()
            ims = [item for sublist in ims for item in sublist]
            raw_X.extend(ims)
            labels = zip(dat[1], dat[2])
            raw_y.extend(labels)

        for dat in test_loader:
            ims = dat[0].numpy()
            ims = [item for sublist in ims for item in sublist]
            raw_X.extend(ims)
            labels = zip(dat[1], dat[2])
            raw_y.extend(labels)

        raw_X = np.array(raw_X)
        raw_y = np.array(raw_y)
        raw_X.reshape(len(raw_X), 28, 28)

        # Enlargen image for better compatibility with CNNs
        X = MGDA_Data.create_bigger_image(raw_X)

        # One hot encode all labels
        y = np.array([[MGDA_Data.one_hot_encode(label[0]), MGDA_Data.one_hot_encode(label[1])]
                      for label in raw_y])
        return X, y

    @staticmethod
    def one_hot_encode(num, length=10):
        return [int(num == index) for index in range(length)]

def Cifar10Mnist_dataset():

    data_path = "Data/Cifar10Mnist"

    if not path.exists(path.join(data_path, "test_10k_CIFAR_MNIST.pkl")) or not path.exists(path.join(data_path, "train_50k_CIFAR_MNIST.pkl")):
         _, train_50k_images, test_10k_images = create_cifar10mnist()
    else:
        with open(f'{data_path}/train_50k_CIFAR_MNIST.pkl', 'rb') as f:
            train_50k_images = pickle.load(f)
        with open(f'{data_path}/test_10k_CIFAR_MNIST.pkl', 'rb') as f:
            test_10k_images = pickle.load(f)

    transform_train = transforms.Compose([transforms.ToTensor(),
                                            transforms.Lambda(lambda x: np.transpose(x, (1, 2, 0))),
                                            transforms.Lambda(lambda x: np.array(x))  # Convert to NumPy array
                                            ])

    transform_test = transforms.Compose([transforms.ToTensor(),
                                            transforms.Lambda(lambda x: np.transpose(x, (1, 2, 0))),
                                            transforms.Lambda(lambda x: np.array(x))  # Convert to NumPy array
                                            ])


    # Create PyTorch datasets for the padded images
    train_50k_dataset = PaddedDataset(train_50k_images, transform = transform_train)
    test_10k_dataset = PaddedDataset(test_10k_images, transform = transform_test)

    return train_50k_dataset, test_10k_dataset

def MultiMnist_dataset():
    
    data_path = "data"
    batch_size = [256, 100]

    train_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,)),
                                        transforms.Lambda(lambda x: np.transpose(x, (1, 2, 0))),
                                        transforms.Lambda(lambda x: np.array(x))  # Convert to NumPy array
                                        ])

    test_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,)),
                                        transforms.Lambda(lambda x: np.transpose(x, (1, 2, 0))),
                                        transforms.Lambda(lambda x: np.array(x))  # Convert to NumPy array
                                        ])

    transformers = [train_transform, test_transform] # [None, None]

    configs = {
    "mnist": {
        "path": data_path,
        "all_tasks": ["L", "R"]
        },
        }
    
    params = {
        "optimizer": "Adam", 
        "batch_size": batch_size,
        "lr": 0.0001,
        "dataset": "mnist",
        "tasks": ["0", "1"],
        "scales": {"0":0.025, "1":0.025},
        "parallel": True
    }

    def Transform(transform):
        if transform is None:
            return transforms.Compose([ transforms.ToTensor()])
        else:
            return transform
        
    transformers = [Transform(transformers[0]), Transform(transformers[1])]
    
    _, mm_train_dst, _, mm_test_dst = get_dataset(params, configs, transformers)

    return mm_train_dst, mm_test_dst
